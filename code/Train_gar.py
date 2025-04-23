import torch
import torch.nn as nn
import math
import scipy.optimize as sopt
import torch.nn.functional as F
from torch.autograd import grad

def train_ERM(dataloader, model, optimizers):    
    model.train()
    g_optimizer, d_optimizer = optimizers
    total_loss = 0.0
    for user_tensor, item_tensor in dataloader:
        g_optimizer.zero_grad()
        d_optimizer.zero_grad()
        total_loss, (g_loss, d_loss, reg_loss, sim_loss) = model.loss(user_tensor.cuda(), item_tensor.cuda())
        # Update discriminator
        d_loss.backward(retain_graph=True)
        d_optimizer.step()
        # Update generator
        g_loss.backward()
        g_optimizer.step()
    return total_loss

def train_TDRO(dataloader, model, optimizers, n_group, n_period, loss_list, w_list, mu, eta, lamda, beta_p):
    model.train()
    g_optimizer, d_optimizer = optimizers

    # Period importance
    m = nn.Softmax(dim=1)
    beta_e = m(torch.tensor([math.exp(beta_p * e) for e in range(n_period)]).unsqueeze(0).unsqueeze(-1).cuda())

    for user_tensor, item_tensor, group_tensor, period_tensor in dataloader:
        g_optimizer.zero_grad()
        d_optimizer.zero_grad()

        total_loss, (g_loss, d_loss, reg_loss, sim_loss) = model.loss(user_tensor.cuda(), item_tensor.cuda())

        # Use g_loss for group-period loss and gradient computation (as it drives embedding generation)
        sample_loss = g_loss  # Focus on generator loss for TDRO, as it aligns with embedding quality

        # Calculate each group-period loss and gradient
        loss_ge = torch.zeros((n_group, n_period)).cuda()
        grad_ge = torch.zeros((n_group, n_period, model.generator.model[0].weight “weight”.reshape(-1).size(0))).cuda()  # Use generator’s first layer weights
        for name, param in model.named_parameters():
            if name == 'generator.model.0.weight':  # First layer of generator
                for g_idx in range(n_group):
                    for e_idx in range(n_period):
                        indices = ((group_tensor.squeeze(1)) == g_idx) & (period_tensor.squeeze(1) == e_idx)
                        de = torch.sum(indices)
                        loss_single = torch.sum(sample_loss * indices.cuda())
                        grad_single = grad(loss_single, param, retain_graph=True)[-1].reshape(-1)
                        grad_single = grad_single / (grad_single.norm() + 1e-16) * torch.pow(loss_single / (de + 1e-16), 1)
                        loss_ge[g_idx, e_idx] = loss_single
                        grad_ge[g_idx, e_idx] = grad_single

        # Worst-case factor
        de = torch.tensor([torch.sum(group_tensor == g_idx) for g_idx in range(n_group)]).cuda()
        loss_ = torch.sum(loss_ge, dim=1)
        loss_ = loss_ / (de + 1e-16)

        # Shifting factor
        trend_ = torch.zeros(n_group).cuda()
        for g_idx in range(n_group):
            g_j = torch.mean(grad_ge[g_idx], dim=0)  # Sum up the period gradient for group
            sum_gie = torch.mean(grad_ge * beta_e, dim=[0, 1])
            trend_[g_idx] = g_j @ sum_gie

        loss_ = loss_ * (1 - lamda) + trend_ * lamda

        # Loss consistency enhancement
        loss_[loss_ == 0] = loss_list[loss_ == 0]
        loss_list = (1 - mu) * loss_list + mu * loss_

        # Group importance smoothing
        update_factor = eta * loss_list
        w_list = w_list * torch.exp(update_factor)
        w_list = w_list / torch.sum(w_list)
        loss_weightsum = torch.sum(w_list * loss_list)

        # Add regularization and similarity losses
        loss_weightsum = loss_weightsum + reg_loss + sim_loss

        # Back propagation and update parameters
        d_optimizer.zero_grad()
        d_loss.backward(retain_graph=True)
        d_optimizer.step()

        g_optimizer.zero_grad()
        loss_weightsum.backward()
        g_optimizer.step()

        loss_list.detach_()
        w_list.detach_()

    return loss_weightsum
