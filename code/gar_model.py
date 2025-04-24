import torch
import torch.nn as nn
import torch.nn.functional as F

class GARModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(GARModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

    def forward(self, user_ids, item_ids):
        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(item_ids)
        return torch.sum(user_embeds * item_embeds, dim=1)

    def get_user_item_embeddings(self):
        return self.user_embedding.weight, self.item_embedding.weight

class Discriminator(nn.Module):
    def __init__(self, embedding_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(embedding_dim * 2, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, user_embed, item_embed):
        x = torch.cat([user_embed, item_embed], dim=1)
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x)).squeeze()
