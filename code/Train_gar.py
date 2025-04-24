import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def adversarial_train(generator, discriminator, train_data, embedding_dim,
                      epochs=10, batch_size=512, device='cpu', lr_gen=0.001, lr_dis=0.001):
    
    generator.to(device)
    discriminator.to(device)

    gen_optimizer = optim.Adam(generator.parameters(), lr=lr_gen)
    dis_optimizer = optim.Adam(discriminator.parameters(), lr=lr_dis)

    bce_loss = nn.BCELoss()

    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        total_gen_loss, total_dis_loss = 0, 0

        dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        for user_ids, pos_items, neg_items in dataloader:
            user_ids = user_ids.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)

            # === Train Discriminator ===
            user_embeds = generator.user_embedding(user_ids)
            pos_item_embeds = generator.item_embedding(pos_items)
            neg_item_embeds = generator.item_embedding(neg_items)

            real_scores = discriminator(user_embeds, pos_item_embeds)
            fake_scores = discriminator(user_embeds, neg_item_embeds)

            real_labels = torch.ones_like(real_scores)
            fake_labels = torch.zeros_like(fake_scores)

            dis_loss = bce_loss(real_scores, real_labels) + bce_loss(fake_scores, fake_labels)

            dis_optimizer.zero_grad()
            dis_loss.backward()
            dis_optimizer.step()
            total_dis_loss += dis_loss.item()

            # === Train Generator ===
            user_embeds = generator.user_embedding(user_ids)
            neg_item_embeds = generator.item_embedding(neg_items)
            gen_scores = discriminator(user_embeds, neg_item_embeds)

            gen_loss = bce_loss(gen_scores, torch.ones_like(gen_scores))  # Fool discriminator

            gen_optimizer.zero_grad()
            gen_loss.backward()
            gen_optimizer.step()
            total_gen_loss += gen_loss.item()

        print(f"[Epoch {epoch+1}] Gen Loss: {total_gen_loss:.4f} | Dis Loss: {total_dis_loss:.4f}")
