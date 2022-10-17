import itertools
import os
import numpy as np
import torch
import torchvision
import random
import argparse
import copy




def pretrain(Pretrain_p, data_loader, view, optimizer, device):
    tot_loss = 0.
    criterion = torch.nn.MSELoss()
    for batch_idx, (xs, y, idx) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        _, _, xrs, zs = model(xs)
        loss_list = []
        for v in range(view):
            for w in range(v+1, view):
                loss_list.append(
                    0.0001 * crossview_contrastive_Loss(torch.softmax(zs[v], dim=0), torch.softmax(zs[w], dim=0)))
            loss_list.append(criterion(xs[v], xrs[v]))
        for v in range(view):
            loss_list.append(criterion(xs[v], xrs[v]))

        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))
