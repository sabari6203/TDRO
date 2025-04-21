import argparse
import os
import time
import numpy as np
import torch
import random
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Dataset import data_load
from Full_rank import full_ranking
from Metric import print_results
from torch.utils.data import default_collate

# Custom collate function for GAR only (user and item tensors)
def custom_collate(batch):
    user_tensor = torch.cat([item[0] for item in batch], dim=0)  # [batch_size]
    item_tensor = torch.stack([item[1] for item in batch], dim=0)  # [batch_size, 1 + num_neg]
    return user_tensor, item_tensor

# GAR Dataset class (simplified without groups or periods)
class GAR_Dataset(torch.utils.data.Dataset):
    def __init__(self, num_user, num_item, user_item_dict, cold_set, train_data, num_neg, pretrained_emb, dataset='amazon'):
        self.num_user = num_user
        self.num_item = num_item
        self.num_neg = num_neg
        self.user_item_dict = user_item_dict
        self.cold_set = cold_set
        self.all_set = set(range(num_user, num_user + num_item)) - self.cold_set  # all warm items
        self.train_data = []
        self.pretrained_emb = pretrained_emb
        self.dataset = dataset

        # Generate user-item pairs with negatives
        for u_id, i_ids in train_data.items():
            for pos_item in i_ids:
                self.train_data.append((u_id, pos_item))
        
        self.user_tensor = []
        self.item_tensor = []
        
        for user, pos_item in self.train_data:
            # Add positive item
            self.user_tensor.append(user)
            self.item_tensor.append(pos_item)
            
            # Add negative items
            neg_items = self._get_neg_items(pos_item, user)
            self.user_tensor.append(user)
            self.item_tensor.append(torch.tensor([pos_item] + neg_items))

        self.user_tensor = torch.tensor(self.user_tensor, dtype=torch.long)
        self.item_tensor = torch.stack(self.item_tensor, dim=0)

    def _get_neg_items(self, pos_item, user):
        neg_items = []
        while len(neg_items) < self.num_neg:
            neg_item = np.random.randint(self.num_user, self.num_user + self.num_item)
            if neg_item not in self.user_item_dict.get(user, []) and neg_item not in neg_items:
                neg_items.append(neg_item)
        return neg_items

    def __len__(self):
        return len(self.user_tensor)

    def __getitem__(self, idx):
        return self.user_tensor[idx], self.item_tensor[idx]

# GAR Model (without TDRO)
class GARModel(torch.nn.Module):
    def __init__(self, num_user, num_item, dim_E, feature_dim, alpha=0.5, beta=0.6):
        super(GARModel, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.dim_E = dim_E
        self.feature_dim = feature_dim
        self.alpha = alpha
        self.beta = beta
        self.result = torch.zeros(num_user + num_item, dim_E, device='cuda')
        self.emb_id = torch.arange(num_user + num_item, device='cuda')
    
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.BatchNorm1d(feature_dim),
            torch.nn.Linear(feature_dim, 200),
            torch.nn.Tanh(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(200, dim_E)
        )
        self.generator = torch.nn.Sequential(
            torch.nn.BatchNorm1d(feature_dim),
            torch.nn.Linear(feature_dim, 200),
            torch.nn.Tanh(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(200, dim_E)
        )
        self.discriminator = torch.nn.Sequential(
            torch.nn.BatchNorm1d(dim_E),
            torch.nn.Linear(dim_E, 200),
            torch.nn.Tanh(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(200, 1)
        )
        self.user_embedding = torch.nn.Embedding(num_user, dim_E)
        self.item_embedding = torch.nn.Embedding(num_item, dim_E)

    def forward(self, user_ids, item_ids, features, training=False):
        user_emb = self.user_embedding(user_ids)  # [batch_size, dim_E]
        batch_size, num_items = item_ids.size()  # [batch_size, 1 + num_neg]
        item_emb = self.item_embedding(item_ids - self.num_user)  # [batch_size, 1 + num_neg, dim_E]
        features_flat = features.view(-1, self.feature_dim)  # [batch_size * (1 + num_neg), feature_dim]
        feature_reps_flat = self.feature_extractor(features_flat)  # [batch_size * (1 + num_neg), dim_E]
        feature_reps = feature_reps_flat.view(batch_size, num_items, self.dim_E)  # Reshape back
        
        gen_reps_flat = self.generator(features_flat)
        gen_reps = gen_reps_flat.view(batch_size, num_items, self.dim_E)
        
        disc_input = torch.cat([item_emb.mean(dim=1), gen_reps.mean(dim=1)], dim=0)  # [2 * batch_size, dim_E]
        disc_output = self.discriminator(disc_input)  # [2 * batch_size, 1]
        real_output = disc_output[:batch_size]  # [batch_size, 1]
        fake_output = disc_output[batch_size:]  # [batch_size, 1]
        return user_emb, item_emb, feature_reps, gen_reps, real_output, fake_output

    def loss(self, user_tensor, item_tensor, features):
        batch_size = user_tensor.size(0)
        user_emb, item_emb, feature_reps, gen_reps, real_output, fake_output = self.forward(
            user_tensor, item_tensor, features
        )
        
        # Prediction loss
        pred_loss = torch.mean(torch.pow(user_emb.unsqueeze(1) - feature_reps.mean(dim=1), 2))
        
        # Discriminator and generator losses
        d_loss = torch.mean(
            torch.nn.functional.binary_cross_entropy_with_logits(real_output, torch.ones_like(real_output)) +
            torch.nn.functional.binary_cross_entropy_with_logits(fake_output, torch.zeros_like(fake_output))
        )
        g_loss = torch.mean(
            torch.nn.functional.binary_cross_entropy_with_logits(fake_output, torch.ones_like(fake_output))
        )
        sim_loss = torch.mean(torch.abs(gen_reps.mean(dim=1) - item_emb.mean(dim=1)))
        
        # Total loss with balancing coefficients
        total_loss = self.beta * pred_loss + self.alpha * (d_loss + (1 - self.alpha) * g_loss + self.alpha * sim_loss)
        return total_loss, torch.tensor(0.0)

# Argument parser setup
def init():
    parser = argparse.ArgumentParser(description="Run GAR on Amazon dataset")
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--data_path', default='amazon', help='Dataset path (set to amazon)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--num_epoch', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of data loader workers')
    parser.add_argument('--topK', default='[10, 20, 50, 100]', help='Top-K recommendation list')
    parser.add_argument('--step', type=int, default=2000, help='Step size for ranking')
    parser.add_argument('--l_r', type=float, default=5e-4, help='Initial learning rate')
    parser.add_argument('--dim_E', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--num_neg', type=int, default=128, help='Number of negative samples')
    parser.add_argument('--alpha', type=float, default=0.5, help='Coefficient for adversarial loss')
    parser.add_argument('--beta', type=float, default=0.6, help='Coefficient for interaction prediction loss')
    parser.add_argument('--gpu', default='0', help='GPU ID')
    parser.add_argument('--save_path', default='./models/', help='Model save path')
    parser.add_argument('--pretrained_emb', default='./pretrained_emb/', help='Path to pretrained embeddings')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = init()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    global num_user, num_item
    print('Loading Amazon dataset...')
    num_user, num_item, num_warm_item, train_data, val_data, val_warm_data, val_cold_data, test_data, test_warm_data, test_cold_data, v_feat, a_feat, t_feat = data_load(args.data_path)
    dir_str = f'../data/{args.data_path}'

    user_item_all_dict = {u_id: train_data[u_id] + val_data[u_id] + test_data[u_id] for u_id in train_data}
    train_dict = {u_id: train_data[u_id] for u_id in train_data}
    tv_dict = {u_id: train_data[u_id] + val_data[u_id] for u_id in train_data}

    warm_item = set([i_id.item() + num_user for i_id in torch.tensor(list(np.load(dir_str + '/warm_item.npy', allow_pickle=True).item()))])
    cold_item = set([i_id.item() + num_user for i_id in torch.tensor(list(np.load(dir_str + '/cold_item.npy', allow_pickle=True).item()))])

    pretrained_emb = torch.FloatTensor(np.load(args.pretrained_emb + args.data_path + '/all_item_feature.npy', allow_pickle=True)).cuda()
    feature_dim = pretrained_emb.size(1)

    # Use GAR_Dataset instead of DRO_Dataset
    train_dataset = GAR_Dataset(num_user, num_item, user_item_all_dict, cold_item, train_data, args.num_neg, pretrained_emb, dataset='amazon')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=custom_collate, drop_last=True)
    print('Dataset loaded.')

    model = GARModel(num_user, num_item, args.dim_E, feature_dim, alpha=args.alpha, beta=args.beta).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.l_r)

    # Add ReduceLROnPlateau scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)

    max_recall = 0.0
    num_decreases = 0
    best_epoch = 0
    topK = eval(args.topK)
    max_val_result = max_test_result = max_test_result_warm = max_test_result_cold = None

    torch.cuda.empty_cache()
    for epoch in range(args.num_epoch):
        epoch_start_time = time.time()
        total_loss = 0.0
        for user_tensor, item_tensor in train_dataloader:
            user_tensor, item_tensor = user_tensor.to(device), item_tensor.to(device)
            item_indices = item_tensor - num_user
            if (item_indices < 0).any() or (item_indices >= pretrained_emb.size(0)).any():
                print(f"Invalid item indices: min {item_indices.min()}, max {item_indices.max()}, pretrained_emb size {pretrained_emb.size(0)}")
                raise ValueError("Item indices out of bounds")
            features = pretrained_emb[item_indices].to(device)
            loss, _ = model.loss(user_tensor, item_tensor, features)
            optimizer.zero_grad()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        torch.cuda.empty_cache()
        elapsed_time = time.time() - epoch_start_time
        print(f"Epoch {epoch:03d}: Average Loss = {total_loss/len(train_dataloader):.4f}, Time = {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}")
        if torch.isnan(torch.tensor(total_loss)):
            print("Loss is NaN. Exiting.")
            break

        if (epoch + 1) % 1 == 0:
            with torch.no_grad():
                model.result[:num_user] = model.user_embedding.weight
                model.result[num_user:] = model.item_embedding.weight
                val_result = full_ranking(model, val_data, train_dict, None, False, args.step, topK)
                test_result = full_ranking(model, test_data, tv_dict, None, False, args.step, topK)
                test_result_warm = full_ranking(model, test_warm_data, tv_dict, cold_item, False, args.step, topK)
                test_result_cold = full_ranking(model, test_cold_data, tv_dict, warm_item, False, args.step, topK)

            print('---' * 18)
            print(f"Epoch {epoch:03d} Results")
            print_results(None, val_result, test_result)
            print('Warm Items:')
            print_results(None, None, test_result_warm)
            print('Cold Items:')
            print_results(None, None, test_result_cold)

            # Update scheduler based on validation Recall@10
            scheduler.step(val_result[1][0])

            if val_result[1][0] > max_recall:
                best_epoch = epoch
                max_recall = val_result[1][0]
                max_val_result = val_result
                max_test_result = test_result
                max_test_result_warm = test_result_warm
                max_test_result_cold = test_result_cold
                num_decreases = 0
                if not os.path.exists(args.save_path):
                    os.makedirs(args.save_path)
                torch.save(model, f'{args.save_path}GAR_amazon.pth')
            else:
                num_decreases += 1
                if num_decreases > 20:
                    print('Early stopping triggered.')
                    break

    model = torch.load(f'{args.save_path}GAR_amazon.pth')
    model.eval()
    with torch.no_grad():
        model.result[:num_user] = model.user_embedding.weight
        model.result[num_user:] = model.item_embedding.weight
        test_result = full_ranking(model, test_data, tv_dict, None, False, args.step, topK)
        test_result_warm = full_ranking(model, test_warm_data, tv_dict, cold_item, False, args.step, topK)
        test_result_cold = full_ranking(model, test_cold_data, tv_dict, warm_item, False, args.step, topK)

    print('===' * 18)
    print(f"Training completed. Best epoch: {best_epoch}")
    print('Validation Results:')
    print_results(None, max_val_result, max_test_result)
    print('Test Results:')
    print('All Items:')
    print_results(None, None, test_result)
    print('Warm Items:')
    print_results(None, None, test_result_warm)
    print('Cold Items:')
    print_results(None, None, test_result_cold)
    print('---' * 18)
