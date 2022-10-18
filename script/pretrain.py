import torch

from core.evaulate import evaluation


def pretrain(model, Pretrain_p, dataset, data_loader, criterion, view, optimizer, data_size, class_num, device):
    for epoch in range(Pretrain_p['p_epoch']):
        loss_tot, loss_rec, loss_iic = 0, 0, 0
        for batch_idx, (xs, y, idx) in enumerate(data_loader):
            for v in range(view):
                xs[v] = xs[v].to(device)
            zs, rs = model(xs)
            tot_list , rec_list, iic_list= [], [], []
            for v in range(view):
                rec_list.append(criterion.mse(xs[v], rs[v]))
                for w in range(v+1, view):
                    iic_list.append(0.001*criterion.forward_iic(zs[v], zs[w]))
            rec = sum(rec_list)
            iic = sum(iic_list)
            tot = rec + iic
            optimizer.zero_grad()
            tot.backward()
            optimizer.step()
            loss_rec += rec.item()
            loss_iic += iic.item()
            loss_tot += tot.item()
        if (Pretrain_p['p_epoch'] + 1) % Pretrain_p['p_interval'] == 0:
            output = "Epoch : {:.0f}/{:.0f} ===> Reconstruction loss = {:.4f}" \
                     "===> IIC loss = {:.4e} ===> Total Loss = {:.4e}" \
                .format((epoch + 1), Pretrain_p['p_interval'], loss_rec, loss_iic, loss_tot)
            print(output)
            scores = evaluation(Pretrain_p, model, dataset, view, data_size, class_num, device)









