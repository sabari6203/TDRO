import time
import random
import numpy as np
from tqdm import tqdm
import scipy.sparse as sp
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from kmeans_pytorch import kmeans

def data_load(dataset):
    dir_str = '../data/' + dataset

    train_data = np.load(dir_str+'/training_dict.npy', allow_pickle=True).item()
    val_data = np.load(dir_str+'/validation_dict.npy', allow_pickle=True).item()
    val_warm_data = np.load(dir_str+'/validation_warm_dict.npy', allow_pickle=True).item()
    val_cold_data = np.load(dir_str+'/validation_cold_dict.npy', allow_pickle=True).item()
    test_data = np.load(dir_str+'/testing_dict.npy', allow_pickle=True).item()
    test_warm_data = np.load(dir_str+'/testing_warm_dict.npy', allow_pickle=True).item()
    test_cold_data = np.load(dir_str+'/testing_cold_dict.npy', allow_pickle=True).item()

    if dataset == "amazon":
        num_user = 21607
        num_item = 93755
        num_warm_item = 75069
        pca_feat = np.load(dir_str + '/img_pca_map.npy', allow_pickle=True).item()
        v_feat = np.zeros((num_item,len(pca_feat[0]))) # pca dim = 64
        for i_id in pca_feat:
            v_feat[i_id] = np.array(pca_feat[i_id])
        v_feat = torch.tensor(v_feat,dtype=torch.float).cuda()
        a_feat = None
        t_feat = None
    
    elif dataset == "micro-video":
        num_user = 21608
        num_item = 64437
        num_warm_item = 56722
        pca_feat = np.load(dir_str + '/visual_feature_64.npy', allow_pickle=True)
        v_feat = np.zeros((num_item,pca_feat.shape[1])) # pca dim = 64
        for i_id in range(num_item):
            v_feat[i_id] = pca_feat[i_id]
            i_id += 1
        v_feat = torch.tensor(v_feat,dtype=torch.float).cuda()
        a_feat = None
        text_feat = np.load(dir_str + '/text_name_feature.npy', allow_pickle=True)
        t_feat = np.zeros((num_item,text_feat.shape[1]))
        for i_id in range(num_item):
            t_feat[i_id] = text_feat[i_id]
            i_id += 1
        t_feat = torch.tensor(t_feat,dtype=torch.float).cuda()

    elif dataset == "kwai":
        num_user = 7010
        num_item = 86483
        num_warm_item = 74470
        pca_feat = np.load(dir_str + '/img_pca_map.npy', allow_pickle=True)
        v_feat = torch.tensor(pca_feat,dtype=torch.float).cuda()
        a_feat = None
        t_feat = None

    else:
        raise NotImplementedError

    # add item id to org_id + num_user
    for u_id in train_data:
        for i,i_id in enumerate(train_data[u_id]):
            train_data[u_id][i] = i_id + num_user
        for i,i_id in enumerate(val_data[u_id]):
            val_data[u_id][i] = i_id + num_user
        for i,i_id in enumerate(val_warm_data[u_id]):
            val_warm_data[u_id][i] = i_id + num_user
        for i,i_id in enumerate(val_cold_data[u_id]):
            val_cold_data[u_id][i] = i_id + num_user
        for i,i_id in enumerate(test_data[u_id]):
            test_data[u_id][i] = i_id + num_user
        for i,i_id in enumerate(test_warm_data[u_id]):
            test_warm_data[u_id][i] = i_id + num_user
        for i,i_id in enumerate(test_cold_data[u_id]):
            test_cold_data[u_id][i] = i_id + num_user

    return num_user, num_item, num_warm_item, train_data, val_data, val_warm_data, val_cold_data, test_data, test_warm_data, test_cold_data, v_feat, a_feat, t_feat

def get_item_group_data_mask(ui_dict, portion=[3,3,3]):
    """
    This function returns the item group list, where each group contains items of different popularity.
    The group is assigned according to the popularity in test set.
    """
    item_cnt = {}
    for u_id in ui_dict:
        for i_id in ui_dict[u_id]:
            item_cnt[i_id] = item_cnt.get(i_id,0) + 1
    item_cnt = dict(sorted(item_cnt.items()))

    n_item = math.ceil(len(item_cnt)/portion[0])
    item_group = {}
    i_cnt = 0
    for i_id in item_cnt:
        item_group[i_id] = i_cnt//n_item
        i_cnt+=1
    
    group_data = [{u_id:[] for u_id in ui_dict} for _ in range(len(portion))]
    group_mask = [set() for _ in range(len(portion))]
    for u_id in ui_dict:
        for i_id in ui_dict[u_id]:
            group_data[item_group[i_id]][u_id].append(i_id)
    for i_id, g_idx in item_group.items():
        for idx in range(len(portion)):
            if idx == g_idx:
                continue
            group_mask[idx].add(i_id)

    return group_data, group_mask

def get_test_group(ui_dict, warm_item, cold_item, portion=[0.3,0.6,1]):
    """
    This function returns the user group list, where each group contains different portions of cold item interactions.
    Portion = cold interactions / all interactions (per user)
    """
    if portion[0] > 1:
        group_list = [0 for u_id in range(len(ui_dict))]
        u_dict = {}
        for u_id in ui_dict:
            if len(ui_dict[u_id]) == 0:
                continue
            cold_cnt = 0
            for i_id in ui_dict[u_id]:
                if i_id in cold_item:
                    cold_cnt += 1
            u_coldPortion = cold_cnt / len(ui_dict[u_id]) if len(ui_dict[u_id]) else 0
            u_dict[u_id] = u_coldPortion
        u_dict = dict(sorted(u_dict.items(), key=lambda x: x[1]))

        n_user = math.ceil(len(u_dict) / portion[0])
        u_cnt = 0
        for u_id in u_dict:
            group_list[u_id] = u_cnt // n_user
            u_cnt += 1
        return group_list
    else:
        group_list = [0 for u_id in range(len(ui_dict))]
        for u_id in ui_dict:
            cold_cnt = 0
            for i_id in ui_dict[u_id]:
                if i_id in cold_item:
                    cold_cnt += 1
            u_coldPortion = cold_cnt / len(ui_dict[u_id]) if len(ui_dict[u_id]) else 0
            for i, p in enumerate(portion):
                if u_coldPortion <= p:
                    group_list[u_id] = i
                    break
        return group_list

class DRO_Dataset(Dataset):
    def __init__(self, num_user, num_item, user_item_all_dict, cold_set, train_data, num_neg, n_group, n_period, split_mode, pretrained_emb, v_feat, a_feat, t_feat):
        self.num_user = num_user
        self.num_item = num_item
        self.user_item_all_dict = user_item_all_dict
        self.cold_set = cold_set
        self.train_data = train_data
        self.num_neg = num_neg
        self.n_group = n_group
        self.n_period = n_period
        self.split_mode = split_mode
        self.pretrained_emb = pretrained_emb
        self.v_feat = v_feat
        self.a_feat = a_feat
        self.t_feat = t_feat
        self.dataset = 'amazon' if num_user == 21607 else 'micro-video' if num_user == 21608 else 'kwai'

        self.user_list = []
        self.item_list = []
        self.group_list = []
        self.env_list = []

        self.gen_group(pretrained_emb, n_group, train_data)
        if split_mode == 'global':
            self.gen_env_global(n_period, train_data)
        else:
            self.gen_env_relative(n_period, user_item_all_dict)

        for u_id in range(num_user):
            for i_id in train_data[u_id]:
                self.user_list.append(u_id)
                self.item_list.append([i_id])
                self.group_list.append(self.group[u_id])
                self.env_list.append(self.env[u_id])
                neg_list = []
                i_cnt = 0
                while i_cnt < num_neg:
                    neg_id = random.randint(num_user, num_user + num_item - 1)
                    if neg_id not in user_item_all_dict[u_id] and neg_id not in neg_list:
                        neg_list.append(neg_id)
                        i_cnt += 1
                self.item_list[-1] += neg_list

    def __len__(self):
        return len(self.user_list)

    def __getitem__(self, idx):
        return torch.tensor(self.user_list[idx]), torch.tensor(self.item_list[idx]), torch.tensor(self.group_list[idx]), torch.tensor(self.env_list[idx])

    def gen_group(self, pretrained_emb, n_group, train_data):
        self.group = [0 for _ in range(self.num_user)]
        item_cnt = {}
        for u_id in train_data:
            for i_id in train_data[u_id]:
                item_cnt[i_id] = item_cnt.get(i_id, 0) + 1
        warm_item = list(item_cnt.keys())
        all_item = warm_item + [i_id for i_id in range(self.num_user, self.num_user + self.num_item) if i_id not in warm_item]
        rep = torch.zeros((len(all_item), 64)).cuda()  # Default dim=64 for features
        if self.dataset in ['amazon', 'micro-video', 'kwai']:
            feat = torch.zeros_like(rep).cuda()
            if self.dataset == 'amazon':
                feat = F.normalize(self.v_feat, dim=1) if self.v_feat is not None else feat
            elif self.dataset == 'micro-video':
                feat = F.normalize(self.v_feat, dim=1) if self.v_feat is not None else feat
                if self.t_feat is not None:
                    feat += F.normalize(self.t_feat, dim=1)
            elif self.dataset == 'kwai':
                feat = F.normalize(self.v_feat, dim=1) if self.v_feat is not None else feat
            rep = feat
        cluster_ids, _ = kmeans(X=rep, num_clusters=n_group, distance='cosine', device=torch.device('cuda:0'))
        item_group = {all_item[i]: int(cluster_ids[i]) for i in range(len(all_item))}
        for u_id in train_data:
            if len(train_data[u_id]) == 0:
                continue
            u_group = []
            for i_id in train_data[u_id]:
                u_group.append(item_group[i_id])
            self.group[u_id] = max(set(u_group), key=u_group.count)

    def gen_env_global(self, n_period, train_data):
        dir_str = '../data/' + self.dataset
        item_time_dict = np.load(dir_str + '/item_time_dict.npy', allow_pickle=True).item()
        item_map_reverse = np.load(dir_str + '/item_map_reverse.npy', allow_pickle=True).item()
        time_list = []
        for i_id in item_time_dict:
            if item_map_reverse[i_id] + self.num_user in train_data:
                time_list.append(item_time_dict[i_id])
        time_list = sorted(time_list)
        time_split = [time_list[int(len(time_list) * i / n_period)] for i in range(n_period)]
        time_split.append(float('inf'))
        self.env = [0 for _ in range(self.num_user)]
        for u_id in train_data:
            if len(train_data[u_id]) == 0:
                continue
            u_time = []
            for i_id in train_data[u_id]:
                u_time.append(item_time_dict[item_map_reverse[i_id - self.num_user]])
            u_time = sorted(u_time)[0]
            for i, t in enumerate(time_split):
                if u_time <= t:
                    self.env[u_id] = i
                    break

    def gen_env_relative(self, n_period, user_item_all_dict):
        self.env = [0 for _ in range(self.num_user)]
        for u_id in user_item_all_dict:
            if len(user_item_all_dict[u_id]) == 0:
                continue
            n_inter = len(user_item_all_dict[u_id])
            if n_inter < n_period:
                self.env[u_id] = 0
            else:
                self.env[u_id] = min(int(n_inter / (n_period + 1)), n_period - 1)
