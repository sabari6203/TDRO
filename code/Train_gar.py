import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

def train_TDRO(train_dataloader, model, optimizer, n_group, n_period, loss_list, w_list, mu, eta, lam, p):
    g_optimizer, d_optimizer = optimizer
    model.train()
    total_loss = 0
    loss_dict = {'g_loss': [], 'd_loss': [], 'reg_loss': [], 'sim_loss': []}
    for _ in range(n_group):
        loss_list.append([[] for _ in range(n_period)])
        w_list.append([1.0 / n_period for _ in range(n_period)])
    for batch_idx, (user_tensor, item_tensor, group_tensor, period_tensor) in enumerate(train_dataloader):
        g_optimizer.zero_grad()
        d_optimizer.zero_grad()
        total_loss, (g_loss, d_loss, reg_loss, sim_loss) = model.loss(user_tensor.cuda(), item_tensor.cuda())
        total_loss.backward()
        for g in range(n_group):
            for t in range(n_period):
                loss_list[g][t].append(g_loss.item() + d_loss.item() + reg_loss.item() + model.contrastive * sim_loss.item())
        # Calculate total number of generator parameters
        gen_param_size = sum(param.numel() for param in model.generator.parameters() if param.requires_grad)
        grad_ge = torch.zeros((n_group, n_period, gen_param_size)).cuda()
        grad_dis = torch.zeros((n_group, n_period, model.discriminator_user.linear1.weight.reshape(-1).size(0) + model.discriminator_item.linear1.weight.reshape(-1).size(0))).cuda()
        for idx in range(user_tensor.size(0)):
            g_optimizer.zero_grad()
            d_optimizer.zero_grad()
            loss = model.loss(user_tensor[idx:idx+1].cuda(), item_tensor[idx:idx+1].cuda())[0]
            loss.backward()
            g = group_tensor[idx].item()
            t = period_tensor[idx].item()
            grad_ge[g][t] += torch.cat([param.grad.reshape(-1) for param in model.generator.parameters() if param.grad is not None])
            grad_dis[g][t] += torch.cat([param.grad.reshape(-1) for param in model.discriminator_user.parameters() if param.grad is not None] +
                                        [param.grad.reshape(-1) for param in model.discriminator_item.parameters() if param.grad is not None])
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
        weighted_loss = 0
        for idx in range(user_tensor.size(0)):
            g = group_tensor[idx].item()
            t = period_tensor[idx].item()
            loss = model.loss(user_tensor[idx:idx+1].cuda(), item_tensor[idx:idx+1].cuda())[0]
            weighted_loss += w_list[g][t] * loss
        weighted_loss /= user_tensor.size(0)
        weighted_loss.backward()
        g_optimizer.step()
        d_optimizer.step()
        total_loss += weighted_loss.item()
        loss_dict['g_loss'].append(g_loss.item())
        loss_dict['d_loss'].append(d_loss.item())
        loss_dict['reg_loss'].append(reg_loss.item())
        loss_dict['sim_loss'].append(sim_loss.item())
    return total_loss / len(train_dataloader), loss_dict
