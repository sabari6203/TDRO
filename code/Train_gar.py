import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.amp import autocast, GradScaler

def train_TDRO(train_dataloader, model, optimizer, n_group, n_period, loss_list, w_list, mu, eta, lam, p):
    g_optimizer, d_optimizer = optimizer
    model.train()
    total_loss = 0
    loss_dict = {'g_loss': [], 'd_loss': [], 'reg_loss': [], 'sim_loss': []}
    scaler = GradScaler('cuda')
    for _ in range(n_group):
        loss_list.append([[] for _ in range(n_period)])
        w_list.append([1.0 / n_period for _ in range(n_period)])
    for batch_idx, (user_tensor, item_tensor, group_tensor, period_tensor) in enumerate(train_dataloader):
        print(f"Batch {batch_idx}: Starting")
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        g_optimizer.zero_grad()
        d_optimizer.zero_grad()
        with autocast('cuda'):
            total_loss, (g_loss, d_loss, reg_loss, sim_loss) = model.loss(user_tensor.cuda(), item_tensor.cuda())
        scaler.scale(total_loss).backward()
        for g in range(n_group):
            for t in range(n_period):
                loss_list[g][t].append(g_loss.item() + d_loss.item() + reg_loss.item() + model.contrastive * sim_loss.item())
        # Calculate total number of parameters
        gen_param_size = sum(param.numel() for param in model.generator.parameters() if param.requires_grad)
        dis_param_size = sum(param.numel() for param in model.discriminator_user.parameters() if param.requires_grad) + \
                         sum(param.numel() for param in model.discriminator_item.parameters() if param.requires_grad)
        grad_ge = torch.zeros((n_group, n_period, gen_param_size)).cuda()
        grad_dis = torch.zeros((n_group, n_period, dis_param_size)).cuda()
        print(f"Batch {batch_idx}: gen_param_size={gen_param_size}, dis_param_size={dis_param_size}")
        batch_size_inner = 64
        for idx in range(0, user_tensor.size(0), batch_size_inner):
            end_idx = min(idx + batch_size_inner, user_tensor.size(0))
            g_optimizer.zero_grad()
            d_optimizer.zero_grad()
            with autocast('cuda'):
                loss = model.loss(user_tensor[idx:end_idx].cuda(), item_tensor[idx:end_idx].cuda())[0]
            scaler.scale(loss).backward()
            # Debug gradients
            grad_list_ge = [param.grad.reshape(-1) for param in model.generator.parameters() if param.grad is not None]
            grad_list_dis = [param.grad.reshape(-1) for param in model.discriminator_user.parameters() if param.grad is not None] + \
                            [param.grad.reshape(-1) for param in model.discriminator_item.parameters() if param.grad is not None]
            print(f"Batch {batch_idx}, idx {idx}: len(grad_list_ge)={len(grad_list_ge)}, len(grad_list_dis)={len(grad_list_dis)}")
            if len(grad_list_ge) == 0:
                print(f"Warning: No gradients for generator parameters in batch {batch_idx}, idx {idx}")
                continue
            grad_cat_ge = torch.cat(grad_list_ge)
            grad_cat_dis = torch.cat(grad_list_dis)
            print(f"Batch {batch_idx}, idx {idx}: grad_cat_ge.shape={grad_cat_ge.shape}, grad_cat_dis.shape={grad_cat_dis.shape}")
            g = group_tensor[idx:end_idx].cuda()
            t = period_tensor[idx:end_idx].cuda()
            for i in range(end_idx - idx):
                g_idx = g[i].item()
                t_idx = t[i].item()
                print(f"Batch {batch_idx}, idx {idx}, i {i}: g_idx={g_idx}, t_idx={t_idx}")
                if g_idx >= n_group or t_idx >= n_period:
                    print(f"Warning: Invalid indices g_idx={g_idx}, t_idx={t_idx}, skipping")
                    continue
                grad_ge[g_idx][t_idx] += grad_cat_ge[i * gen_param_size:(i + 1) * gen_param_size]
                grad_dis[g_idx][t_idx] += grad_cat_dis[i * dis_param_size:(i + 1) * dis_param_size]
            if idx % 100 == 0:
                print(f"Batch {batch_idx}: Processed samples {idx}/{user_tensor.size(0)}")
        print(f"Batch {batch_idx}: Completed gradient accumulation")
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        for g in range(n_group):
            for t in range(n_period):
                grad_ge[g][t] /= user_tensor.size(0)
                grad_dis[g][t] /= user_tensor.size(0)
        for g in range(n_group):
            for t in range(n_period):
                if len(loss_list[g][t]) == 0:
                    continue
                loss_g_t = np.mean(loss_list[g][t][-1:])
                w_list[g][t] = w_list[g][t] * np.exp(eta * loss_g_t)
        for g in range(n_group):
            w_sum = sum(w_list[g])
            for t in range(n_period):
                w_list[g][t] /= w_sum
        g_optimizer.zero_grad()
        d_optimizer.zero_grad()
        print(f"Batch {batch_idx}: Starting weighted loss computation")
        with autocast('cuda'):
            losses = model.loss(user_tensor.cuda(), item_tensor.cuda())[0]
            weights = torch.zeros(user_tensor.size(0)).cuda()
            for idx in range(user_tensor.size(0)):
                g_idx = group_tensor[idx].item()
                t_idx = period_tensor[idx].item()
                if g_idx >= n_group or t_idx >= n_period:
                    print(f"Warning: Invalid indices g_idx={g_idx}, t_idx={t_idx} in weighted loss, skipping")
                    continue
                weights[idx] = w_list[g_idx][t_idx]
            weighted_loss = (losses * weights).mean()
        print(f"Batch {batch_idx}: Weighted loss: {weighted_loss.item()}")
        scaler.scale(weighted_loss).backward()
        scaler.unscale_(g_optimizer)
        scaler.unscale_(d_optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(g_optimizer)
        scaler.step(d_optimizer)
        scaler.update()
        total_loss += weighted_loss.item()
        loss_dict['g_loss'].append(g_loss.item())
        loss_dict['d_loss'].append(d_loss.item())
        loss_dict['reg_loss'].append(reg_loss.item())
        loss_dict['sim_loss'].append(sim_loss.item())
        torch.cuda.empty_cache()
        print(f"Batch {batch_idx}: Completed, total_loss: {total_loss}")
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Epoch completed, average loss: {total_loss / len(train_dataloader)}")
    return total_loss / len(train_dataloader), loss_dict
