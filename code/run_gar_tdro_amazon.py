import os
import time
import pickle
import argparse
import torch
import numpy as np
import pandas as pd
from pprint import pprint
from torch import nn, optim
from model import GAR
from utils import ndcg, Timer, set_seed_torch
import utils

# Other setting
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random Seed.')
parser.add_argument('--gpu_id', type=int, default=0)

# Dataset
parser.add_argument('--datadir', type=str, default="./data/", help='Directory of the dataset.')
parser.add_argument('--dataset', type=str, default="Amazon", help='Dataset to use.')

# Validation & testing
parser.add_argument('--val_interval', type=float, default=1)
parser.add_argument('--val_start', type=int, default=0, help='Validation per training batch.')
parser.add_argument('--test_batch_us', type=int, default=200)
parser.add_argument('--Ks', nargs='?', default='[20]', help='Output sizes of every layer')
parser.add_argument('--n_test_user', type=int, default=2000)

# Cold-start model training
parser.add_argument('--embed_meth', type=str, default='ncf', help='Recommender method')
parser.add_argument('--batch_size', type=int, default=1024, help='Normal batch size.')
parser.add_argument('--train_set', type=str, default='map', choices=['map', 'emb'])
parser.add_argument('--max_epoch', type=int, default=1000)
parser.add_argument('--restore', type=str, default="")
parser.add_argument('--patience', type=int, default=10, help='Early stop patience.')

# Cold-start model parameters
parser.add_argument('--model', type=str, default='gar')
parser.add_argument('--alpha', type=float, default=0.05, help='Parameter in GAR')
parser.add_argument('--beta', type=float, default=0.1, help='Parameter in GAR')

args, _ = parser.parse_known_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
args.Ks = eval(args.Ks)
args.model = args.model.upper()
set_seed_torch(args.seed)
pprint(vars(args))
timer = Timer(name='main')
ndcg.init(args)

# Load data
content_data = np.load(os.path.join(args.datadir, args.dataset, args.dataset + '_item_content.npy'))
content_data = np.concatenate([np.zeros([1, content_data.shape[-1]]), content_data], axis=0)
para_dict = pickle.load(open(args.datadir + args.dataset + '/convert_dict.pkl', 'rb'))
train_data = pd.read_csv(args.datadir + args.dataset + '/warm_{}.csv'.format(args.train_set), dtype=np.int64).values

# Load embedding
t0 = time.time()
emb_path = os.path.join(args.datadir, args.dataset, "{}.npy".format(args.embed_meth))
user_node_num = max(para_dict['user_array']) + 1
item_node_num = max(para_dict['item_array']) + 1
emb = np.load(emb_path)
user_emb = emb[:user_node_num]
item_emb = emb[user_node_num:]
timer.logging('Embeddings are loaded from {}'.format(emb_path))

# Load test set
def get_exclude_pair(u_pair, ts_nei):
    pos_item = np.array(sorted(list(set(para_dict['pos_user_nb'][u_pair[0]]) - set(ts_nei[u_pair[0]]))),
                        dtype=np.int64)
    pos_user = np.array([u_pair[1]] * len(pos_item), dtype=np.int64)
    return np.stack([pos_user, pos_item], axis=1)

def get_exclude_pair_count(ts_user, ts_nei, batch):
    exclude_pair_list = []
    exclude_count = [0]
    for i, beg in enumerate(range(0, len(ts_user), batch)):
        end = min(beg + batch, len(ts_user))
        batch_user = ts_user[beg:end]
        batch_range = list(range(end - beg))
        batch_u_pair = tuple(zip(batch_user.tolist(), batch_range))  # (org_id, map_id)

        specialize_get_exclude_pair = lambda x: get_exclude_pair(x, ts_nei)
        exclude_pair = list(map(specialize_get_exclude_pair, batch_u_pair))
        exclude_pair = np.concatenate(exclude_pair, axis=0)

        exclude_pair_list.append(exclude_pair)
        exclude_count.append(exclude_count[i] + len(exclude_pair))

    exclude_pair_list = np.concatenate(exclude_pair_list, axis=0)
    return [exclude_pair_list, exclude_count]

exclude_val_cold = get_exclude_pair_count(para_dict['cold_val_user'][:args.n_test_user], para_dict['cold_val_user_nb'],
                                          args.test_batch_us)
exclude_test_warm = get_exclude_pair_count(para_dict['warm_test_user'][:args.n_test_user],
                                           para_dict['warm_test_user_nb'],
                                           args.test_batch_us)
exclude_test_cold = get_exclude_pair_count(para_dict['cold_test_user'][:args.n_test_user],
                                           para_dict['cold_test_user_nb'],
                                           args.test_batch_us)
exclude_test_hybrid = get_exclude_pair_count(para_dict['hybrid_test_user'][:args.n_test_user],
                                             para_dict['hybrid_test_user_nb'],
                                             args.test_batch_us)
timer.logging("Loaded excluded pairs for validation and test.")

# Model configuration
device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
model = GAR(args, user_emb.shape[-1], content_data.shape[-1]).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)

save_dir = './GAR/model_save/'
os.makedirs(save_dir, exist_ok=True)
save_path = save_dir + args.dataset + '-' + args.model + '-'
param_file = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
save_file = save_path + param_file
args.param_file = param_file
timer.logging('Model will be stored in ' + save_file)

# Training
timer.logging("Training model...")
epoch = 0
best_va_metric = 0
patience_count = 0
train_time = 0
val_time = 0
stop_flag = 0
batch_count = 0
item_index = np.arange(item_node_num)

for epoch in range(1, args.max_epoch + 1):
    model.train()
    train_input = utils.bpr_neg_samp(para_dict['warm_user'], len(train_data),
                                     para_dict['emb_user_nb'], para_dict['warm_item'])
    n_batch = len(train_input) // args.batch_size
    for beg in range(0, len(train_input) - args.batch_size, args.batch_size):
        end = beg + args.batch_size
        batch_count += 1
        t_train_begin = time.time()
        batch_lbs = train_input[beg:end]

        d_loss = model.train_d(user_emb[batch_lbs[:, 0]].to(device),
                               item_emb[batch_lbs[:, 1]].to(device),
                               item_emb[batch_lbs[:, 2]].to(device),
                               content_data[batch_lbs[:, 1]].to(device))
        g_loss = model.train_g(user_emb[batch_lbs[:, 0]].to(device),
                               item_emb[batch_lbs[:, 1]].to(device),
                               content_data[batch_lbs[:, 1]].to(device))
        loss = d_loss + g_loss
        t_train_end = time.time()
        train_time += t_train_end - t_train_begin

        # Validation - interval can be float
        if (batch_count % int(n_batch * args.val_interval) == 0) and (epoch >= args.val_start):
            model.eval()
            t_val_begin = time.time()

            gen_user_emb = model.get_user_emb(user_emb)
            gen_item_emb = model.get_item_emb(content_data, item_emb,
                                              para_dict['warm_item'], para_dict['cold_item'])
            va_metric, _ = ndcg.test(model.get_ranked_rating,
                                     lambda u: model.get_user_rating(u, item_index, gen_user_emb, gen_item_emb),
                                     ts_nei=para_dict['cold_val_user_nb'],
                                     ts_user=para_dict['cold_val_user'][:args.n_test_user],
                                     masked_items=para_dict['warm_item'],
                                     exclude_pair_cnt=exclude_val_cold,
                                     )
            va_metric_current = va_metric['ndcg'][0]
            if va_metric_current > best_va_metric:
                best_va_metric = va_metric_current
                torch.save(model.state_dict(), save_file)  # Save model weights
                patience_count = 0
            else:
                patience_count += 1
                if patience_count > args.patience:
                    stop_flag = 1
                    break

            t_val_end = time.time()
            val_time += t_val_end - t_val_begin
            timer.logging('Epoch {}({}/{}) Loss: {:.4f}| Va_metric: {:.4f}| Best: {:.4f}| Time_Train: {:.2fs}| Val: {:.2fs}'.format(
                epoch, patience_count, args.patience, loss, va_metric_current, best_va_metric, train_time, val_time))

    if stop_flag:
        break

# Test
timer.logging("Testing model...")
model.load_state_dict(torch.load(save_file))  # Load the best model

# Cold recommendation
model.eval()
cold_res, _ = ndcg.test(model.get_ranked_rating,
                        lambda u: model.get_user_rating(u, item_index, gen_user_emb, gen_item_emb),
                        ts_nei=para_dict['cold_test_user_nb'],
                        ts_user=para_dict['cold_test_user'][:args.n_test_user],
                        masked_items=para_dict['warm_item'],
                        exclude_pair_cnt=exclude_test_cold,
                        )
timer.logging(f'Cold-start recommendation result@{args.Ks[0]}: PRE, REC, NDCG: {cold_res["precision"][0]:.4f}, {cold_res["recall"][0]:.4f}, {cold_res["ndcg"][0]:.4f}')

# Warm recommendation
warm_res, _ = ndcg.test(model.get_ranked_rating,
                        lambda u: model.get_user_rating(u, item_index, gen_user_emb, gen_item_emb),
                        ts_nei=para_dict['warm_test_user_nb'],
                        ts_user=para_dict['warm_test_user'][:args.n_test_user],
                        masked_items=para_dict['cold_item'],
                        exclude_pair_cnt=exclude_test_warm,
                        )
timer.logging(f"Warm recommendation result@{args.Ks[0]}: PRE, REC, NDCG: {warm_res['precision'][0]:.4f}, {warm_res['recall'][0]:.4f}, {warm_res['ndcg'][0]:.4f}")

# Hybrid recommendation
hybrid_res, _ = ndcg.test(model.get_ranked_rating,
                          lambda u: model.get_user_rating(u, item_index, gen_user_emb, gen_item_emb),
                          ts_nei=para_dict['hybrid_test_user_nb'],
                          ts_user=para_dict['hybrid_test_user'][:args.n_test_user],
                          masked_items=None,
                          exclude_pair_cnt=exclude_test_hybrid,
                          )
timer.logging(f"Hybrid recommendation result@{args.Ks[0]}: PRE, REC, NDCG: {hybrid_res['precision'][0]:.4f}, {hybrid_res['recall'][0]:.4f}, {hybrid_res['ndcg'][0]:.4f}")

# Save test results
result_file = './GAR/result/'
if not os.path.exists(result_file):
    os.makedirs(result_file)
with open(result_file + f'{args.model}.txt', 'a') as f:
    f.write(str(vars(args)))
    f.write(' | ')
    for i in range(len(args.Ks)):
        f.write(f'{cold_res["precision"][i]:.4f} {cold_res["recall"][i]:.4f} {cold_res["ndcg"][i]:.4f} ')
    f.write(' | ')
    for i in range(len(args.Ks)):
        f.write(f'{warm_res["precision"][i]:.4f} {warm_res["recall"][i]:.4f} {warm_res["ndcg"][i]:.4f} ')
    f.write(' | ')
    for i in range(len(args.Ks)):
        f.write(f'{hybrid_res["precision"][i]:.4f} {hybrid_res["recall"][i]:.4f} {hybrid_res["ndcg"][i]:.4f} ')
    f.write('\n')

