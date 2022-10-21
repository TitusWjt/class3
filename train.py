import itertools
import os
import numpy as np
import torch
import torchvision
import random
import argparse
import copy
from torch.utils.data import Dataset

from core.loss import Loss
from data.dataloader.dataloader import load_data
from model.autoencoder import Autoencoder
from script.pretrain import pretrain
from utils import yaml_config_hook
from torch.utils.tensorboard import SummaryWriter

def main():
    #Load hyperparameters
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("./configs/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    tb_writer = SummaryWriter(log_dir="runs/tensorboard")

    # Environments
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    # Set random seeds
    np.random.seed(args.seed)
    random.seed(args.seed + 1)
    torch.manual_seed(args.seed + 2)
    torch.cuda.manual_seed(args.seed + 3)
    torch.backends.cudnn.deterministic = True

    #Load dataset
    dataset, dims, view, data_size, class_num = load_data(args.dataset_name)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )


    autoencoder = Autoencoder(args.model_kwargs, view, dims, class_num)
    optimizer = torch.optim.Adam(
        itertools.chain(autoencoder.encoders.parameters(), autoencoder.decoders.parameters(),),
        lr=args.learning_rate)
    autoencoder.to_device(device)
    criterion = Loss(args.batch_size, device).to(device)

    if args.isPretrain:
        pretrain(autoencoder, args.Pretrain_p, dataset, data_loader, criterion, view, optimizer, data_size, class_num, device, tb_writer)

    #load weight
    #autoencoder.load_state_dict(torch.load(args.Pretrain_p['pretrain_path']))























































if __name__ == '__main__':
    main()