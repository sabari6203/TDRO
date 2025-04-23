import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, user_dim, item_dim, content_dim, hidden_dim):
        super(Generator, self).__init__()
        self.fc_user = nn.Linear(user_dim, hidden_dim)
        self.fc_content = nn.Linear(content_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, item_dim)

    def forward(self, user_embedding, item_content):
        x_u = F.relu(self.fc_user(user_embedding))
        x_c = F.relu(self.fc_content(item_content))
        x = x_u + x_c
        return torch.sigmoid(self.fc_out(x))


class Discriminator(nn.Module):
    def __init__(self, user_dim, item_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(user_dim + item_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, user_embedding, item_embedding):
        x = torch.cat([user_embedding, item_embedding], dim=-1)
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))


class GARModel(nn.Module):
    def __init__(self, num_users, num_items, emb_dim, content_dim, hidden_dim):
        super(GARModel, self).__init__()
        self.user_embeddings = nn.Embedding(num_users, emb_dim)
        self.item_embeddings = nn.Embedding(num_items, emb_dim)
        self.generator = Generator(emb_dim, emb_dim, content_dim, hidden_dim)
        self.discriminator = Discriminator(emb_dim, emb_dim, hidden_dim)

    def generate(self, user_ids, item_content):
        user_emb = self.user_embeddings(user_ids)
        return self.generator(user_emb, item_content)

    def discriminate(self, user_ids, item_ids):
        user_emb = self.user_embeddings(user_ids)
        item_emb = self.item_embeddings(item_ids)
        return self.discriminator(user_emb, item_emb)
