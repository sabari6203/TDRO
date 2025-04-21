import torch
import torch.nn as nn
import torch.nn.functional as F

class GAR(nn.Module):
    def __init__(self, args, user_emb_size, item_emb_size):
        super(GAR, self).__init__()
        
        # Hyperparameters from arguments
        self.args = args
        self.user_emb_size = user_emb_size
        self.item_emb_size = item_emb_size
        self.alpha = args.alpha
        self.beta = args.beta

        # Define embedding layers for users and items
        self.user_embedding = nn.Embedding(user_emb_size, args.embed_size)
        self.item_embedding = nn.Embedding(item_emb_size, args.embed_size)

        # Define generator and discriminator networks
        self.generator = nn.Sequential(
            nn.Linear(args.embed_size, args.embed_size),
            nn.ReLU(),
            nn.Linear(args.embed_size, args.embed_size)
        )

        self.discriminator = nn.Sequential(
            nn.Linear(args.embed_size, args.embed_size),
            nn.ReLU(),
            nn.Linear(args.embed_size, 1)
        )
        
    def forward(self, user_emb, item_emb, content_emb):
        # Forward pass for generator and discriminator
        gen_item_emb = self.generator(item_emb)
        real_item_score = self.discriminator(item_emb)
        fake_item_score = self.discriminator(gen_item_emb)
        
        return real_item_score, fake_item_score

    def train_d(self, user_emb, item_emb, content_emb):
        real_item_score = self.discriminator(item_emb)
        fake_item_score = self.generator(item_emb)
        
        # Discriminator loss (Real items should have high score, fake items should have low score)
        real_loss = F.binary_cross_entropy_with_logits(real_item_score, torch.ones_like(real_item_score))
        fake_loss = F.binary_cross_entropy_with_logits(fake_item_score, torch.zeros_like(fake_item_score))
        
        loss = real_loss + fake_loss
        return loss

    def train_g(self, user_emb, item_emb, content_emb):
        # Generator loss: the generator tries to make the fake items look like real items
        fake_item_emb = self.generator(item_emb)
        fake_item_score = self.discriminator(fake_item_emb)
        
        # We want the fake items to be classified as real
        loss = F.binary_cross_entropy_with_logits(fake_item_score, torch.ones_like(fake_item_score))
        return loss

    def get_user_emb(self, user_emb):
        # Get user embedding by simply returning the embedding (could be extended)
        return self.user_embedding(user_emb)

    def get_item_emb(self, content_emb, item_emb):
        # Get item embedding by combining content and embedding (could be extended)
        return self.item_embedding(item_emb)

    def get_user_rating(self, user_emb, item_index, gen_user_emb, gen_item_emb):
        # Generate ratings (user-item interaction predictions)
        user_emb = self.user_embedding(user_emb)
        item_emb = self.item_embedding(item_index)
        scores = torch.sum(user_emb * item_emb, dim=-1)
        return scores

