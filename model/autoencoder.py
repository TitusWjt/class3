from model.encoder import Encoder
from model.decoder import Decoder
import torch
from core.evaulate import valid
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self,
                 config,
                 view, dims, class_num):
        super(Autoencoder, self).__init__()
        self._config = config
        self.view = view
        self.dims = dims
        self.class_num = class_num


        if self._config['Encoder']['arch'][-1] != self._config['Decoder']['arch'][0]:
            raise ValueError('Inconsistent latent dim!')

        self.encoders_list = []
        self.decoders_list = []
        for v in range(self.view):
            self.encoders_list.append(Encoder([self.dims[v]] + self._config['Encoder']['arch'], self._config['Encoder']['function'], self._config['Encoder']['batchnorm']))
            self.decoders_list.append(Decoder(self._config['Decoder']['arch'] + [self.dims[v]],  self._config['Decoder']['function'], self._config['Decoder']['batchnorm']))
        self.encoders = nn.ModuleList(self.encoders_list)
        self.decoders = nn.ModuleList(self.decoders_list)

    def forward(self, xs):
        zs = []
        rs = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            r = self.decoders[v](z)
            zs.append(z)
            rs.append(r)
        return  zs, rs

    def to_device(self, device):
        """ to cuda if gpu is used """
        for v in range(self.view):
            self.encoders[v].to(device)
            self.decoders[v].to(device)

    def pretrain(self, model, Pretrain_p, data_loader, view, optimizer, device):
        tot_loss = 0.
        criterion = torch.nn.MSELoss()
        for batch_idx, (xs, y, idx) in enumerate(data_loader):
            for v in range(view):
                xs[v] = xs[v].to(device)
            optimizer.zero_grad()
            zs, rs = model(xs)
            loss_list = []
            for v in range(view):
                loss_list.append(criterion(xs[v], rs[v]))
                for w in range(v + 1, view):
                    loss_list.append(criterion.forward_iic(zs[v], zs[w]))
            loss = sum(loss_list)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
        print('Epoch {}'.format(Pretrain_p['p_epoch']), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))
        acc, nmi, pur = valid(model, device, dataset, view, data_size, class_num, eval_h=False)
