import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, act='tanh', drop_rate=0.1):
        super(MLP, self).__init__()
        layers = []
        in_dim = input_dim
        
        for i, out_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.BatchNorm1d(out_dim))
            if act == 'relu':
                layers.append(nn.LeakyReLU(negative_slope=0.01))
            elif act == 'tanh':
                layers.append(nn.Tanh())
            layers.append(nn.Dropout(drop_rate))
            in_dim = out_dim
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

class GAR(nn.Module):
    def __init__(self, warm_item, cold_item, num_user, num_item, reg_weight, dim_E, v_feat, a_feat, t_feat, temp_value, num_neg, contrastive, num_sample):
        super(GAR, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.reg_weight = reg_weight
        self.dim_E = dim_E
        self.warm_item = torch.tensor(warm_item, dtype=torch.long).cuda()
        self.cold_item = torch.tensor(cold_item, dtype=torch.long).cuda()
        
        # Learning rates and dropout rates
        self.g_lr = 1e-3
        self.d_lr = 1e-3
        self.g_drop = 0.1
        self.d_drop = 0.5
        
        # MLP layer configurations
        self.g_layer = [200, 200]
        self.d_layer = [200, 200]
        self.g_act = 'tanh'
        self.d_act = 'tanh'
        
        # Loss weights
        self.alpha = 0.5  # Weight for similarity loss
        self.beta = 0.5   # Weight for negative samples in discriminator loss
        
        # Multimodal feature handling
        self.dim_feat = 0
        if v_feat is not None:
            self.v_feat = F.normalize(v_feat, dim=1)
            self.dim_feat += self.v_feat.size(1)
        else:
            self.v_feat = None
        
        if a_feat is not None:
            self.a_feat = F.normalize(a_feat, dim=1)
            self.dim_feat += self.a_feat.size(1)
        else:
            self.a_feat = None
        
        if t_feat is not None:
            self.t_feat = F.normalize(t_feat, dim=1)
            self.dim_feat += self.t_feat.size(1)
        else:
            self.t_feat = None
        
        # Generator: Maps multimodal features to embeddings
        self.generator = MLP(self.dim_feat, self.g_layer + [dim_E], self.g_act, self.g_drop)
        
        # Discriminator: Processes user and item embeddings
        self.discriminator_user = MLP(dim_E, self.d_layer, self.d_act, self.d_drop)
        self.discriminator_item = MLP(dim_E, self.d_layer, self.d_act, self.d_drop)
        
        # ID embeddings
        self.id_embedding = nn.Parameter(nn.init.xavier_normal_(torch.rand((num_user + num_item, dim_E))))
        
    def feature_extractor(self):
        feature = torch.tensor([]).cuda()
        
        if self.v_feat is not None:
            feature = torch.cat((feature, self.v_feat), dim=1)
        
        if self.a_feat is not None:
            feature = torch.cat((feature, self.a_feat), dim=1)
        
        if self.t_feat is not None:
            feature = torch.cat((feature, self.t_feat), dim=1)
        
        return feature

    def build_generator(self, content):
        return self.generator(content)

    def build_discriminator(self, uembs, iembs):
        out_uemb = self.discriminator_user(uembs)
        out_iemb = self.discriminator_item(iembs)
        out = torch.sum(out_uemb * out_iemb, dim=-1)  # Dot product
        return out

    def forward(self, user_tensor, item_tensor):
        pos_item_tensor = item_tensor[:, 0].unsqueeze(1)
        pos_item_tensor = pos_item_tensor.repeat(1, 1 + self.num_neg).view(-1, 1).squeeze()
        
        user_tensor = user_tensor.view(-1, 1).squeeze()
        item_tensor = item_tensor.view(-1, 1).squeeze()
        
        # Extract multimodal features
        content = self.feature_extractor()
        content = content[item_tensor - self.num_user]
        
        # Generate embeddings for cold items
        gen_emb = self.build_generator(content)
        
        # Get embeddings
        user_embedding = self.id_embedding[user_tensor]
        real_item_embedding = self.id_embedding[pos_item_tensor]
        all_item_embedding = self.id_embedding[item_tensor]
        
        # Discriminator outputs
        real_logit = self.build_discriminator(user_embedding, real_item_embedding)
        neg_logit = self.build_discriminator(user_embedding, all_item_embedding)
        fake_logit = self.build_discriminator(user_embedding, gen_emb)
        
        # Discriminator loss
        d_loss = F.binary_cross_entropy_with_logits(
            real_logit - (1 - self.beta) * fake_logit - self.beta * neg_logit,
            torch.ones_like(real_logit)
        )
        
        # Generator loss
        g_out = self.build_discriminator(user_embedding, gen_emb)
        d_out = self.build_discriminator(user_embedding, real_item_embedding)
        g_loss = (1.0 - self.alpha) * F.binary_cross_entropy_with_logits(
            g_out - d_out, torch.ones_like(g_out)
        )
        sim_loss = self.alpha * torch.mean(torch.abs(gen_emb - real_item_embedding))
        g_loss += sim_loss
        
        # Regularization loss
        reg_loss = ((torch.sqrt((user_embedding**2).sum(1))).mean() + 
                    (torch.sqrt((all_item_embedding**2).sum(1))).mean()) / 2
        
        return g_loss, d_loss, reg_loss * self.reg_weight, sim_loss

    def loss(self, user_tensor, item_tensor):
        g_loss, d_loss, reg_loss, sim_loss = self.forward(user_tensor, item_tensor)
        total_loss = g_loss + d_loss + reg_loss + sim_loss
        return total_loss, (g_loss, d_loss, reg_loss, sim_loss)

    def get_item_emb(self, content, item_emb):
        out_emb = item_emb.clone()
        cold_content = content[self.cold_item - self.num_user]
        out_emb[self.cold_item] = self.build_generator(cold_content)
        out_emb = self.discriminator_item(out_emb)
        return out_emb

    def get_user_emb(self, user_emb):
        return self.discriminator_user(user_emb)

    def get_user_rating(self, uids, iids, uemb, iemb):
        u_emb = uemb[uids]
        i_emb = iemb[iids]
        return torch.matmul(u_emb, i_emb.t())

    def get_ranked_rating(self, ratings, k):
        top_score, top_indices = torch.topk(ratings, k, dim=1)
        return top_score, top_indices
