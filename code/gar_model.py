import torch
import torch.nn as nn
import torch.nn.functional as F

class GAR(nn.Module):
    def __init__(self, warm_item, cold_item, num_user, num_item, reg_weight, dim_E, v_feat, a_feat, t_feat, temp_value, num_neg, contrastive, num_sample):
        super(GAR, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.reg_weight = reg_weight
        self.dim_E = dim_E
        self.temp_value = temp_value  # Unused but kept for compatibility
        self.num_neg = num_neg
        self.contrastive = contrastive  # Unused but kept for compatibility
        self.num_sample = num_sample

        # Convert sets to lists for tensor creation
        self.warm_item = torch.tensor(list(warm_item), dtype=torch.long).cuda()
        self.cold_item = torch.tensor(list(cold_item), dtype=torch.long).cuda()

        # Multimodal features
        self.v_feat = v_feat
        self.a_feat = a_feat
        self.t_feat = t_feat

        # Initialize embeddings
        self.id_embedding = nn.Parameter(torch.randn(num_user + num_item, dim_E))
        nn.init.xavier_uniform_(self.id_embedding)

        # Generator: MLP to transform multimodal features to embedding space
        self.generator = nn.Sequential(
            nn.Linear(self.get_feature_dim(), dim_E * 2),
            nn.ReLU(),
            nn.Linear(dim_E * 2, dim_E),
            nn.ReLU()
        )

        # Discriminator: Separate MLPs for users and items
        self.discriminator_user = nn.Sequential(
            nn.Linear(dim_E, dim_E * 2),
            nn.ReLU(),
            nn.Linear(dim_E * 2, 1)
        )
        self.discriminator_item = nn.Sequential(
            nn.Linear(dim_E, dim_E * 2),
            nn.ReLU(),
            nn.Linear(dim_E * 2, 1)
        )

        self.alpha = 0.1  # Weight for similarity loss
        self.beta = 0.1   # Weight for discriminator loss

    def get_feature_dim(self):
        dims = 0
        if self.v_feat is not None:
            dims += self.v_feat.shape[1]
        if self.a_feat is not None:
            dims += self.a_feat.shape[1]
        if self.t_feat is not None:
            dims += self.t_feat.shape[1]
        return dims if dims > 0 else self.dim_E

    def feature_extractor(self):
        features = []
        if self.v_feat is not None:
            features.append(self.v_feat)
        if self.a_feat is not None:
            features.append(self.a_feat)
        if self.t_feat is not None:
            features.append(self.t_feat)
        return torch.cat(features, dim=1) if features else None

    def get_user_emb(self, user_emb):
        return self.discriminator_user(user_emb).squeeze(-1)

    def get_item_emb(self, content, item_emb):
        if content is None:
            return item_emb
        cold_mask = torch.isin(torch.arange(self.num_user, self.num_user + self.num_item, device='cuda'), self.cold_item)
        generated_emb = self.generator(content)
        item_emb = torch.where(cold_mask.unsqueeze(-1), generated_emb, item_emb)
        return self.discriminator_item(item_emb).squeeze(-1)

    def get_user_rating(self, user_ids, item_ids, user_emb, item_emb):
        user_emb = user_emb[user_ids]
        item_emb = item_emb[item_ids - self.num_user]
        return torch.sum(user_emb * item_emb, dim=-1)

    def get_ranked_rating(self, user_ids, user_emb, item_emb, top_k):
        user_emb = user_emb[user_ids]
        scores = torch.matmul(user_emb, item_emb.t())
        _, indices = torch.topk(scores, top_k, dim=-1)
        return indices + self.num_user

    def loss(self, user_tensor, item_tensor):
        user_emb = self.id_embedding[user_tensor]
        item_emb = self.id_embedding[item_tensor]
        content = self.feature_extractor()

        # Get transformed embeddings
        user_emb = self.get_user_emb(user_emb)
        item_emb = self.get_item_emb(content, item_emb)

        pos_item_emb = item_emb[:, 0]
        neg_item_emb = item_emb[:, 1:]

        # Generator loss: BPR-like loss for positive vs negative items
        pos_scores = torch.sum(user_emb * pos_item_emb, dim=-1)
        neg_scores = torch.sum(user_emb.unsqueeze(1) * neg_item_emb, dim=-1)
        g_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores.unsqueeze(-1) - neg_scores)))

        # Discriminator loss: Binary classification for warm vs generated embeddings
        warm_mask = torch.isin(item_tensor[:, 0], self.warm_item)
        d_real = self.discriminator_item(pos_item_emb[warm_mask])
        d_fake = self.discriminator_item(self.generator(content[item_tensor[:, 0] - self.num_user][~warm_mask]))
        d_loss = -torch.mean(torch.log(torch.sigmoid(d_real)) + torch.log(1 - torch.sigmoid(d_fake)))

        # Regularization loss
        reg_loss = self.reg_weight * (torch.norm(self.id_embedding) ** 2 + 
                                      sum(torch.norm(p) ** 2 for p in self.generator.parameters()) +
                                      sum(torch.norm(p) ** 2 for p in self.discriminator_user.parameters()) +
                                      sum(torch.norm(p) ** 2 for p in self.discriminator_item.parameters()))

        # Similarity loss: Encourage generated embeddings to match pretrained embeddings
        sim_loss = 0
        if content is not None:
            generated_emb = self.generator(content[item_tensor[:, 0] - self.num_user])
            sim_loss = self.alpha * torch.mean((generated_emb - pos_item_emb) ** 2)

        total_loss = g_loss + self.beta * d_loss + reg_loss + sim_loss
        return total_loss, (g_loss, d_loss, reg_loss, sim_loss)
