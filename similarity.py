import argparse

import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

import utils
from model import Model

def sim(net, memory_data_loader, test_data_loader):
    net.eval()
    similarity = torch.nn.CosineSimilarity(dim=1)
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    test_feature_bank, train_feature_bank = [], []
    with torch.no_grad():
        # generate feature bank
        for x_k, x_q, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature_k, _ = net(x_k.cuda(non_blocking=True))
            feature_q, _ = net(x_q.cuda(non_blocking=True))
            sim = similarity(feature_k, feature_q)
            train_feature_bank.append(sim)
        # [D, N]
        # train_feature_bank = torch.cat(train_feature_bank, dim=0).t().contiguous()
        train_feature_bank = torch.cat(train_feature_bank, dim=0).contiguous()

        test_bar = tqdm(test_data_loader)
        for x_k, x_q, target in test_bar:
            x_k, x_q = x_k.cuda(non_blocking=True), x_q.cuda(non_blocking=True)
            feature_k, _ = net(x_k.cuda(non_blocking=True))
            feature_q, _ = net(x_q.cuda(non_blocking=True))
            sim = similarity(feature_k, feature_q)
            test_feature_bank.append(sim)
        # [D, N]
        # test_feature_bank = torch.cat(test_feature_bank, dim=0).t().contiguous()
        test_feature_bank = torch.cat(test_feature_bank, dim=0).contiguous()
        if len(train_feature_bank) <= len(test_feature_bank):
            test_feature_bank = test_feature_bank[:len(train_feature_bank)-1]
        else:
            train_feature_bank = train_feature_bank[:len(test_feature_bank)-1]

    test_data = [[s] for s in test_feature_bank]
    test_table = wandb.Table(data=test_data)
    wandb.log({'Test histogram': wandb.plot.histogram(test_table, "Similarity",
           title="Test Similarity")})
    train_data = [[s] for s in train_feature_bank]
    train_table = wandb.Table(data=train_data)
    wandb.log({'Train histogram': wandb.plot.histogram(train_table, "Similarity",
           title="Train Similarity")})
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MoCo')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for each image')
    parser.add_argument('--m', default=4096, type=int, help='Negative sample number')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--momentum', default=0.999, type=float, help='Momentum used for the update of memory bank')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=256, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=500, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning Rate at the training start')
    parser.add_argument('--weight_decay', default=1e-6, type=float, help='Weight Decay')
    parser.add_argument('--dataset', default='stl10', type=str, help='Training Dataset (e.g. CIFAR10, STL10)')
    parser.add_argument('--wandb_project', default='default_project', type=str, help='WandB Project name')
    parser.add_argument('--wandb_run', default='default_run', type=str, help='WandB run name')

    # args parse
    args = parser.parse_args()
    feature_dim, m, temperature, momentum = args.feature_dim, args.m, args.temperature, args.momentum
    k, batch_size, epochs = args.k, args.batch_size, args.epochs

    # wandb init
    config = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "arch": "resnet50",
        "dataset": args.dataset,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "momentum": args.momentum,
        "temperature": args.temperature,
        "feature_dim": args.feature_dim,
        "queue_size": args.m
    }
    wandb.init(project=args.wandb_project, name=args.wandb_run, config=config)

    # data prepare
    train_data = utils.available_dataset[args.dataset](root='data', split='train+unlabeled', transform=utils.train_transform, download=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True,
                              drop_last=True)
    memory_data = utils.available_dataset[args.dataset](root='data', split='train', transform=utils.test_transform, download=True)
    memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    test_data = utils.available_dataset[args.dataset](root='data', split='test', transform=utils.test_transform, download=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    # model setup and optimizer config
    model_q = Model(feature_dim).cuda()
    model_k = Model(feature_dim).cuda()
    # initialize
    for param_q, param_k in zip(model_q.parameters(), model_k.parameters()):
        param_k.data.copy_(param_q.data)
        # not update by gradient
        param_k.requires_grad = False


    sim(model_q, memory_loader, test_loader)

    wandb.finish()
