import argparse

import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import copy

import utils
from model import Model

class Net(nn.Module):
    def __init__(self, num_class, pretrained_path):
        super(Net, self).__init__()

        # encoder
        self.f = Model().f
        # classifier
        self.fc = nn.Linear(512, num_class, bias=True)
        self.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=False)

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return out

def sim(enc, cls, memory_data_loader, test_data_loader):
    enc.eval()
    cls.eval()
    similarity = torch.nn.CosineSimilarity(dim=1)
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    test_feature_bank, train_feature_bank = [], []
    test_feature_bank_k, train_feature_bank_k = [], []
    test_var, train_var = [], []
    idx_09 = []
    counter = 0
    with torch.no_grad():
        # generate feature bank
        for x, target in tqdm(memory_data_loader, desc='Feature extracting'):
            if counter > 10000:
                break
            feature_list = []
            feature_list_k = []
            for data in x:
                f, _ = enc(data.cuda(non_blocking=True))
                f_k, _ = cls(data.cuda(non_blocking=True))
                feature_list.append(f)
                feature_list_k.append(f_k)

            cos_list = []
            cos_list_k = []
            for i in range(len(x)-1):
                for j in range(len(x)-1):
                    if i >= j:
                        continue
                    cos_list.append(similarity(feature_list[i], feature_list[j]))
                    cos_list_k.append(similarity(feature_list_k[i], feature_list_k[j]))

            var = torch.var(torch.stack(cos_list, dim=1), dim=1)
            train_var.append(var)
            result = cos_list[0]
            for i in range(1,len(cos_list)):
                result += cos_list[i]
            result /= len(cos_list)

            
            result_k = cos_list_k[0]
            for i in range(1,len(cos_list_k)):
                result_k += cos_list_k[i]
            result_k /= len(cos_list_k)
                    
            if result[i] >= 0.94:
                idx_09.append(i+counter)
            # if diff[i] > 0.2 and result[i] >= 0.94:
                # idx.append(i+counter)

            train_feature_bank.append(result)
            train_feature_bank_k.append(result_k)
            counter += len(result)

        # [D, N]
        train_feature_bank = torch.cat(train_feature_bank, dim=0).contiguous()
        train_feature_bank_k = torch.cat(train_feature_bank_k, dim=0).contiguous()
        train_var = torch.cat(train_var, dim=0)
        print(train_feature_bank)

    bank_enc = []
    bank_cls = []
    for i in idx_09:
        if i >= len(train_feature_bank):
            break
        bank_enc.append(train_feature_bank[i])
        bank_cls.append(train_feature_bank_k[i])

    bank_enc_09 = []
    bank_enc_wo_09 = []
    bank_cls_09 = []
    bank_cls_wo_09 = []
    for i, data in enumerate(train_feature_bank_k):
        if i in idx_09:
            bank_cls_09.append(data)
            bank_enc_09.append(data)
        else:
            bank_cls_wo_09.append(data)
            bank_enc_wo_09.append(data)
    for i, data in enumerate(train_feature_bank):
        if i in idx_09:
            bank_enc_09.append(data)
        else:
            bank_enc_wo_09.append(data)

    bank_cls_09 = torch.stack(bank_cls_09)
    bank_cls_wo_09 = torch.stack(bank_cls_wo_09)

    plt.title('Cosine Similarity')
    labels = ['>=0.9', '<0.9']
    data = [bank_cls_09.to('cpu').detach().numpy().copy(), bank_cls_wo_09.to('cpu').detach().numpy().copy()]
    plt.hist(data, 50, label=labels, stacked=True)
    plt.savefig("results/sim_dt.png")
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
    enc = Model(feature_dim).cuda()
    cls = Net(feature_dim, 'moco_cifar_1000.pth').cuda()
    cls.load_state_dict(torch.load('moco_cifar_1000.pth'))

    sim(enc, cls, memory_loader, test_loader)
    # sim(model_q, memory_loader, memory_loader)

    wandb.finish()

