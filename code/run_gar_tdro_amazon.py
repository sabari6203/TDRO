import argparse
import os
import time
import numpy as np
import torch
import random
from torch.utils.data import DataLoader
from Dataset import data_load, DRO_Dataset  # Updated to match file name
from Train import train_TDRO
from Full_rank import full_ranking
from Metric import print_results


# GAR Model with TDRO integration
class GARModel(torch.nn.Module):
    def __init__(self, num_user, num_item, dim_E, feature_dim, alpha=0.5, beta=0.9, K=3, E=4, lambda_=0.5, p=0.2, mu=0.5, eta_w=0.01):
        super(GARModel, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.dim_E = dim_E
        self.feature_dim = feature_dim
        self.alpha = alpha
        self.beta = beta
        self.K = K
        self.E = E
        self.lambda_ = lambda_
        self.p = p
        self.mu = mu
        self.eta_w = eta_w
        self.w = torch.ones(K, device='cuda') / K
        self.group_losses = torch.zeros(K, E, device='cuda')
        self.result = torch.zeros(num_user + num_item, dim_E, device='cuda')
        self.emb_id = torch.arange(num_user + num_item, device='cuda')

        # Feature extractor, generator, and discriminator
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, dim_E),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_E, dim_E)
        )
        self.generator = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, dim_E),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_E, dim_E)
        )
        self.discriminator = torch.nn.Sequential(
            torch.nn.Linear(dim_E, dim_E),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_E, 1),
            torch.nn.Sigmoid()
        )
        self.user_embedding = torch.nn.Embedding(num_user, dim_E)
        self.item_embedding = torch.nn.Embedding(num_item, dim_E)

    def forward(self, user_ids, item_ids, features, training=False):
        print(f"item_ids shape: {item_ids.shape}, min: {item_ids.min()}, max: {item_ids.max()}")
        print(f"pretrained_emb shape: {pretrained_emb.shape}, device: {pretrained_emb.device}")
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids - num_user)  # Ensure correct offset
        feature_reps = self.feature_extractor(features)
        gen_reps = self.generator(features)
        real_output = self.discriminator(item_emb)
        fake_output = self.discriminator(gen_reps)
        return user_emb, item_emb, feature_reps, gen_reps, real_output, fake_output

    def loss(self, user_tensor, item_tensor, group_tensor, period_tensor, features):
        batch_size = user_tensor.size(0)
        user_emb, item_emb, feature_reps, gen_reps, real_output, fake_output = self.forward(user_tensor, item_tensor, features)

        # Interaction prediction loss
        pred_loss = torch.mean(torch.pow(user_emb - feature_reps, 2))

        # Adversarial losses
        d_loss_real = torch.mean(torch.pow(real_output - torch.ones_like(real_output), 2))
        d_loss_fake = torch.mean(torch.pow(fake_output - torch.zeros_like(fake_output), 2))
        g_loss = torch.mean(torch.pow(fake_output - torch.ones_like(fake_output), 2))
        d_loss = d_loss_real + d_loss_fake

        # Total loss
        total_loss = self.beta * pred_loss + self.alpha * (d_loss + g_loss)

        # Group and period losses for TDRO
        group_losses = torch.zeros(self.K, device='cuda')
        for i in range(batch_size):
            group = group_tensor[i].item()
            period = period_tensor[i].item()
            loss_i = torch.pow(user_emb[i] - feature_reps[i], 2).mean()
            self.group_losses[group, period] = (1 - self.mu) * self.group_losses[group, period] + self.mu * loss_i
            group_losses[group] += loss_i
        group_losses /= torch.bincount(group_tensor, minlength=self.K).float().cuda()

        # Shifting trend
        period_grads = []
        for e in range(self.E):
            grads_e = torch.zeros(self.dim_E, device='cuda')
            for g in range(self.K):
                mask = (group_tensor == g) & (period_tensor == e)
                if mask.sum() > 0:
                    grads_e += torch.autograd.grad(loss_i, feature_reps, retain_graph=True)[0][mask].mean(dim=0).detach()
            period_grads.append(grads_e * torch.exp(self.p * (e + 1)))
        shifting_trend = sum(period_grads)

        # Group selection
        worst_case = (1 - self.lambda_) * group_losses
        shifting_factors = torch.tensor([torch.dot(grads_e.mean(dim=0), shifting_trend) for grads_e in period_grads], device='cuda')
        scores = worst_case - self.lambda_ * shifting_factors
        j_star = torch.argmin(scores).item()

        # Update group weights
        c_i = (1 - self.lambda_) * group_losses + self.lambda_ * shifting_factors
        self.w = self.w * torch.exp(self.eta_w * c_i)
        self.w /= self.w.sum()

        # Weighted loss
        loss_weightsum = torch.sum(self.w * group_losses) + total_loss
        return loss_weightsum, torch.tensor(0.0)  # Placeholder for reg_loss

# Argument parser setup
def init():
    parser = argparse.ArgumentParser(description="Run GAR+TDRO on Amazon dataset")
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--data_path', default='amazon', help='Dataset path (set to amazon)')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--num_epoch', type=int, default=200, help='Number of epochs')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of data loader workers')
    parser.add_argument('--topK', default='[10, 20, 50, 100]', help='Top-K recommendation list')
    parser.add_argument('--step', type=int, default=2000, help='Step size for ranking')
    parser.add_argument('--l_r', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--dim_E', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--num_neg', type=int, default=256, help='Number of negative samples')
    parser.add_argument('--num_group', type=int, default=3, help='Number of groups for GAR')
    parser.add_argument('--num_period', type=int, default=4, help='Number of time periods for TDRO')
    parser.add_argument('--split_mode', type=str, default='relative', choices=['relative', 'global'], help='Time split mode')
    parser.add_argument('--mu', type=float, default=0.5, help='Streaming learning rate for group DRO')
    parser.add_argument('--eta', type=float, default=0.01, help='Step size for group DRO')
    parser.add_argument('--lam', type=float, default=0.5, help='Coefficient for time-aware shifting trend')
    parser.add_argument('--p', type=float, default=0.2, help='Gradient strength for TDRO')
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

    train_dataset = DRO_Dataset(num_user, num_item, user_item_all_dict, cold_item, train_data, args.num_neg, args.num_group, args.num_period, args.split_mode, pretrained_emb, dataset='amazon')
    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
    print('Dataset loaded.')

    model = GARModel(num_user, num_item, args.dim_E, feature_dim, K=args.num_group, E=args.num_period, lambda_=args.lam, p=args.p, mu=args.mu, eta_w=args.eta).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.l_r)

    w_list = torch.ones(args.num_group).cuda()
    loss_list = torch.zeros(args.num_group).cuda()
    max_recall = 0.0
    num_decreases = 0
    best_epoch = 0
    topK = eval(args.topK)
    max_val_result = max_test_result = max_test_result_warm = max_test_result_cold = None

    for epoch in range(args.num_epoch):
        epoch_start_time = time.time()
        total_loss = 0.0
        for user_tensor, item_tensor, group_tensor, period_tensor in train_dataloader:
            user_tensor, item_tensor, group_tensor, period_tensor = user_tensor.to(device), item_tensor.to(device), group_tensor.to(device), period_tensor.to(device)
            features = pretrained_emb[item_tensor - num_user]  # Align features with item indices
            loss, _ = model.loss(user_tensor, item_tensor, group_tensor, period_tensor, features)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        elapsed_time = time.time() - epoch_start_time
        print(f"Epoch {epoch:03d}: Loss = {total_loss/len(train_dataloader):.4f}, Time = {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}")

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
                torch.save(model, f'{args.save_path}GAR_TDRO_amazon.pth')
            else:
                num_decreases += 1
                if num_decreases > 5:
                    print('Early stopping triggered.')
                    break

    model = torch.load(f'{args.save_path}GAR_TDRO_amazon.pth')
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
