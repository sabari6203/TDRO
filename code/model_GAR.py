import torch
import torch.nn as nn
import torch.nn.functional as F

class GAR(nn.Module):
    def __init__(
        self, warm_item, cold_item, num_user, num_item, reg_weight, dim_E,
        v_feat, a_feat, t_feat, temp_value, num_neg, contrastive, num_sample
    ):
        super(GAR, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.num_neg = num_neg
        self.contrastive = contrastive
        self.reg_weight = reg_weight
        self.temp_value = temp_value
        self.num_sample = num_sample

        self.warm_item = list(warm_item)
        self.cold_item = list(cold_item)
        self.emb_id = list(range(num_user)) + self.warm_item
        self.feat_id = torch.tensor([i_id - num_user for i_id in self.cold_item])

        # ID-based embeddings for users and warm items
        self.id_embedding = nn.Parameter(
            nn.init.xavier_normal_(torch.empty(num_user + num_item, dim_E))
        )

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

        # Generator: maps multimodal features to embedding space
        self.encoder_layer1 = nn.Linear(self.dim_feat, 256)
        self.encoder_layer2 = nn.Linear(256, dim_E)


        # Optional: MLP for user/item embedding transformation (used in scoring)
        self.mlp = nn.Linear(dim_E, dim_E)

        # Store the final embeddings for evaluation
        self.result = nn.init.kaiming_normal_(torch.empty(num_user + num_item, dim_E)).cuda()

    def feature_extractor(self):
        features = []
        if self.v_feat is not None:
            features.append(self.v_feat)
        if self.a_feat is not None:
            features.append(self.a_feat)
        if self.t_feat is not None:
            features.append(self.t_feat)
        feature = torch.cat(features, dim=1) if features else None
        if feature is not None:
            feature = F.leaky_relu(self.encoder_layer1(feature))
            feature = self.encoder_layer2(feature)
        return feature


    def get_item_embedding(self):
        # Warm items: use learned embeddings
        # Cold items: use generator output from features
        feature_emb = self.feature_extractor()
        result = self.id_embedding.clone()
        if feature_emb is not None and len(self.feat_id) > 0:
            result[self.feat_id + self.num_user] = feature_emb[self.feat_id].data
        return result

    def forward(self, user_tensor, item_tensor):
        # user_tensor: [batch]
        # item_tensor: [batch, 1 + num_neg]

        # Prepare positive and negative item indices
        pos_item_tensor = item_tensor[:, 0].unsqueeze(1)
        pos_item_tensor = pos_item_tensor.repeat(1, 1 + self.num_neg).view(-1, 1).squeeze()
        user_tensor = user_tensor.view(-1, 1).squeeze()
        item_tensor = item_tensor.view(-1, 1).squeeze()

        # Get item embeddings (with generator for cold items)
        item_embeddings = self.get_item_embedding()
        user_embedding = self.id_embedding[user_tensor]
        pos_item_embedding = item_embeddings[pos_item_tensor]
        all_item_embedding = item_embeddings[item_tensor]

        # Contrastive loss: positive user-item vs. negatives
        head_embed = F.normalize(pos_item_embedding, dim=1)
        head_user = F.normalize(user_embedding, dim=1)
        all_item_input = all_item_embedding.clone()
        rand_index = torch.randint(all_item_embedding.size(0), (int(all_item_embedding.size(0) * self.num_sample),)).cuda()
        # Optionally replace some item embeddings with feature-based ones for robustness
        feature_emb = self.feature_extractor()
        if feature_emb is not None and len(self.feat_id) > 0:
            all_item_input[rand_index] = feature_emb[rand_index % feature_emb.size(0)].clone()

        # Compute scores and losses
        contrastive_loss_1, sample_loss_1 = self.loss_contrastive(head_user, head_embed, self.temp_value)
        contrastive_loss_2, sample_loss_2 = self.loss_contrastive(user_embedding, all_item_input, self.temp_value)

        reg_loss = ((torch.sqrt((user_embedding ** 2).sum(1))).mean() +
                    (torch.sqrt((all_item_embedding ** 2).sum(1))).mean()) / 2

        return sample_loss_1 * self.contrastive, sample_loss_2 * (1 - self.contrastive), reg_loss

    def loss_contrastive(self, anchor, positives, temp_value):
        # Ensure anchor is repeated to match positives if needed
        if anchor.size(0) != positives.size(0):
            # If anchor is [batch_size, dim] and positives is [batch_size * (1+num_neg), dim]
            anchor = anchor.repeat_interleave(1 + self.num_neg, dim=0)
        
        # Now both tensors should have the same first dimension
        similarity = torch.sum(anchor * positives, dim=1) / temp_value
        all_score = torch.exp(similarity).view(-1, 1 + self.num_neg)
        pos_score = all_score[:, 0]
        all_score = torch.sum(all_score, dim=1)
        sample_loss = -torch.log(pos_score / all_score)
        contrastive_loss = sample_loss.mean()
        return contrastive_loss, sample_loss

    def loss(self, user_tensor, item_tensor):
        cf_loss, constraint_loss, reg_loss = self.forward(user_tensor, item_tensor)
        reg_loss = self.reg_weight * reg_loss
        return cf_loss + constraint_loss, reg_loss

    def eval_mode_embedding(self):
        # For evaluation: update self.result with the latest embeddings
        with torch.no_grad():
            self.result[self.emb_id] = self.id_embedding[self.emb_id].data
            feature_emb = self.feature_extractor()
            if feature_emb is not None and len(self.feat_id) > 0:
                self.result[self.feat_id + self.num_user] = feature_emb[self.feat_id].data

