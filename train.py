import itertools
import os
import numpy as np
import torch
import torchvision
import random
import argparse
import copy
from torch.utils.data import Dataset
from data.dataloader.dataloader import load_data
from model.autoencoder import Autoencoder
from utils.yaml_config_hook import yaml_config_hook




def main():
    #Load hyperparameters
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("./configs/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()

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
    autoencoder = autoencoder.to_device(device)



    pass

    X_list, Y_list = load_data(args.dataset_name)
    x1_train_raw = X_list[0]
    x2_train_raw = X_list[1]
    np.random.seed(args.seed)
    x1_train = torch.from_numpy(x1_train_raw).float().to(device)
    x2_train = torch.from_numpy(x2_train_raw).float().to(device)



    #Set random seeds
    random.seed(args.seed + 1)
    torch.manual_seed(args.seed + 2)
    torch.cuda.manual_seed(args.seed + 3)
    torch.backends.cudnn.deterministic = True


    isPretarin = False
    # Build the model
    wjt = WJT(args.model_kwargs, args.first_stage)
    optimizer = torch.optim.Adam(
        itertools.chain(wjt.autoencoder1.parameters(), wjt.autoencoder2.parameters()),
        lr=args.first_stage['p_lr'])
    wjt.to_device(device)
    # Pretrain the model
    if isPretarin:
        wjt.pretrain(args.first_stage, x1_train, x2_train, optimizer)
    # Exact weights
    encoder1_weight = torch.load(args.first_stage['a1_pretrain_path'])
    e1_del = []
    for key,_ in encoder1_weight.items():
        if "_decoder" in key:
            e1_del.append(key)
    for key in e1_del:
        del encoder1_weight[key]
    encoder2_weight = torch.load(args.first_stage['a2_pretrain_path'])
    e2_del = []
    for key, _ in encoder2_weight.items():
        if "_decoder" in key:
            e2_del.append(key)
    for key in e2_del:
        del encoder2_weight[key]
    # Load weights
    wjt.encoder1.load_state_dict(encoder1_weight)
    wjt.encoder2.load_state_dict(encoder2_weight)
    # Get the mapped data distribution
    X1 = wjt.encoder1.project(x1_train)
    X2 = wjt.encoder2.project(x2_train)
    Y = torch.tensor(Y_list)[0].to(device)  # TODO有点问题
    index = torch.arange(0,2386)



    # Robust Learning
    print('Start separating samples')
    net1_aux = copy.deepcopy(wjt.encoder1._d_c_encoder)   #TODO尝试用K均值来初始化聚类头的权重
    net1_confi1 = copy.deepcopy(wjt.encoder1._d_c_encoder)
    net1_aux.load_state_dict(dict_slice(encoder1_weight,14,44),strict=False)
    net1_confi1.load_state_dict(dict_slice(encoder1_weight,14,44),strict=False)
    net1_aux.to(device)
    net1_confi1.to(device)
    optimizer1 = torch.optim.SGD(wjt.encoder1._d_c_encoder.parameters(), args.second_stage['r_lr'], momentum=args.second_stage['r_momentum'], weight_decay=args.second_stage['r_weight_decay'],
                                 nesterov=True)
    optimizer2 = torch.optim.SGD(net1_aux.parameters(), args.second_stage['r_lr'], momentum=args.second_stage['r_momentum'], weight_decay=args.second_stage['r_weight_decay'],
                                 nesterov=True)
    criterion = criterion_rb()
    #Get Pseudo Label
    acc_Robust_before, p_label = get_P_label(net1_confi1, X1, Y, device, args.class_num)
    print('acc_Robust_before is {}'.format(acc_Robust_before))
    devide = extract_confidence(net1_confi1, p_label, X1, Y,args.second_stage['r_threshold'])
    print('Start refurbishing the encoder')
    conf1 = torch.zeros(50000)
    conf2 = torch.zeros(50000)
    for epoch in range(args.second_stage['r_epoch']):
        X1, X2, Y, p_label, index = shuffle(X1, X2, Y, p_label, index)
        loss, devide, p_label, conf1 = Robust_train(epoch, wjt.encoder1._d_c_encoder, net1_aux, X1, X2, Y, index, optimizer1, criterion, devide, p_label,
                                             conf2,args.second_stage)
        loss, devide, p_label, conf2 = Robust_train(epoch, net1_aux, wjt.encoder1._d_c_encoder, X1, X2, Y, index,
                                                    optimizer2, criterion, devide, p_label,
                                                    conf2, args.second_stage)
        print(loss)
        acc, p_list = test_ruc(wjt.encoder1._d_c_encoder, net1_aux, X1, Y, device, args.class_num)



    pass

























































if __name__ == '__main__':
    main()