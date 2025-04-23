import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.nn.init import xavier_normal_

class Generator(nn.Module):
    def __init__(self, dim_E, v_feat, a_feat, t_feat):
        super(Generator, self).__init__()
        self.v_feat = v_feat
        self.a_feat = a_feat
        self.t_feat = t_feat
        self.dim_E = dim_E
        self.feat_dim = 64 if v_feat is not None else 0
        self.feat_dim += 64 if a_feat is not None else 0
        self.feat_dim += 64 if t_feat is not None else 0
        if self.feat_dim > 0:
            self.linear = nn.Linear(self.feat_dim, dim_E)
            xavier_normal_(self.linear.weight)
        else:
            self.linear = None

    def forward(self, content):
        if self.linear is None:
            return content
        return self.linear(content)

class Discriminator(nn.Module):
    def __init__(self, dim_E):
        super(Discriminator, self).__init__()
        self.dim_E = dim_E
        self.linear1 = nn.Linear(dim_E, dim_E)
        self.linear2 = nn.Linear(dim_E, 1)
        xavier_normal_(self.linear1.weight)
        xavier_normal_(self.linear2.weight)

    def forward(self, input_emb):
        hidden = F.relu(self.linear1(input_emb))
        return self.linear2(hidden).squeeze(-1)

class GAR(nn.Module):
    def __init__(self, warm_item, cold_item, num_user, num_item, reg_weight, dim_E, v_feat, a_feat, t_feat, temp_value, num_neg, contrastive, num_sample):
        super(GAR, self).__init__()
        self.warm_item = warm_item
        self.cold_item = cold_item
        self.num_user = num_user
        self.num_item = num_item
        self.reg_weight = reg_weight
        self.temp_value = temp_value
        self.num_neg = num_neg
        self.contrastive = contrastive
        self.num_sample = num_sample
        self.id_embedding = nn.Embedding(num_user + num_item, dim_E)
        xavier_normal_(self.id_embedding.weight)
        self.v_feat = v_feat
        self.a_feat = a_feat
        self.t_feat = t_feat
        self.generator = Generator(dim_E, v_feat, a_feat, t_feat)
        self.discriminator_user = Discriminator(dim_E)
        self.discriminator_item = Discriminator(dim_E)

    def feature_extractor(self):
        content = torch.zeros((self.num_item, 0)).cuda()
        if self.v_feat is not None:
            content = torch.cat([content, self.v_feat], dim=-1)
        if self.a_feat is not None:
            content = torch.cat([content, self.a_feat], dim=-1)
        if self.t_feat is not None:
            content = torch.cat([content, self.t_feat], dim=-1)
        return content

    def get_user_emb(self, user_emb):
        return user_emb

    def get_item_emb(self, content, item_emb):
        cold_mask = torch.tensor([i_id in self.cold_item for i_id in range(self.num_user, self.num_user + self.num_item)]).cuda()
        generated_emb = self.generator(content)
        item_emb = torch.where(cold_mask.unsqueeze(-1), generated_emb, item_emb)
        return item_emb

    def loss(self, user, item):
        print(f"loss: user.shape={user.shape}, item.shape={item.shape}")
        print(f"loss: item.min={item.min().item()}, item.max={item.max().item()}")
        if item.max() >= self.num_user + self.num_item or item.min() < self.num_user:
            raise ValueError(f"Item indices out of bounds: min={item.min().item()}, max={item.max().item()}, expected [21607, 115361]")
        content = self.feature_extractor()
        user_emb = self.id_embedding(user)  # Shape: (batch_size, dim_E)
        item_indices = item  # Shape: (batch_size, 257), already offset by num_user
        item_emb = self.id_embedding(item_indices)  # Shape: (batch_size, 257, dim_E)
        user_emb = self.get_user_emb(user_emb)
        all_item_indices = torch.arange(self.num_user, self.num_user + self.num_item, device=user.device)  # Shape: (num_item,)
        all_item_emb = self.id_embedding(all_item_indices)  # Shape: (num_item, dim_E)
        all_item_emb = self.get_item_emb(content, all_item_emb)  # Shape: (num_item, dim_E)
        item_emb = all_item_emb[item - self.num_user]  # Shape: (batch_size, 257, dim_E)
        pos_score = torch.sum(user_emb * item_emb[:, 0], dim=-1)  # Shape: (batch_size,)
        neg_score = torch.sum(user_emb.unsqueeze(1) * item_emb[:, 1:], dim=-1)  # Shape: (batch_size, 256)
        g_loss = -torch.mean(F.logsigmoid(pos_score - neg_score))
        d_user_score = self.discriminator_user(user_emb)
        d_item_score = self.discriminator_item(item_emb[:, 0])
        d_loss = torch.mean(F.binary_cross_entropy_with_logits(d_user_score, torch.ones_like(d_user_score)) +
                            F.binary_cross_entropy_with_logits(d_item_score, torch.zeros_like(d_item_score)))
        reg_loss = self.reg_weight * (torch.mean(user_emb**2) + torch.mean(item_emb**2))
        sim_loss = 0
        if self.contrastive > 0:
            pos_sim = torch.sum(item_emb[:, 0] * item_emb[:, 1:self.num_neg//2 + 1], dim=-1)
            neg_sim = torch.sum(item_emb[:, 0] * item_emb[:, self.num_neg//2 + 1:], dim=-1)
            sim_loss = -torch.mean(F.logsigmoid((pos_sim - neg_sim) / self.temp_value))
        total_loss = g_loss + d_loss + reg_loss + self.contrastive * sim_loss
        return total_loss, (g_loss, d_loss, reg_loss, sim_loss)
