import torch

from core.evaulate import valid


def pretrain(model, Pretrain_p, data_loader, criterion, view, optimizer, device):
    tot_loss = 0.
    for batch_idx, (xs, y, idx) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        zs, rs = model(xs)
        loss_list = []
        for v in range(view):
            loss_list.append(criterion.mse(xs[v], rs[v]))
            for w in range(v+1, view):
                loss_list.append(0.001*criterion.forward_iic(zs[v], zs[w]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(Pretrain_p['p_epoch']), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))
    acc, nmi, pur = valid(model, device, dataset, view, data_size, class_num, eval_h=False)