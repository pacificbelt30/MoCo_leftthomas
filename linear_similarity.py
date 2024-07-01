import argparse
import os

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import copy
import matplotlib.pyplot as plt

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

def sim(enc, cls, memory_data_loader, test_data_loader, topk=500):
    enc.eval()
    cls.eval()
    similarity = torch.nn.CosineSimilarity(dim=1)
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    test_feature_bank, train_feature_bank = [], []
    test_feature_bank_k, train_feature_bank_k = [], []
    test_feature_bank_g, train_feature_bank_g = [], []
    test_var, train_var = [], []
    idx_09 = []
    counter = 0
    test_counter = 0
    total_correct_1, test_total_correct_1 = 0, 0
    with torch.no_grad():
        # generate feature bank
        for x, target in tqdm(memory_data_loader, desc='Feature extracting'):
            target = target.cuda()
            if counter > 10000:
                break
            feature_list = []
            feature_list_k = []
            feature_list_g = []
            for data in x:
                f, g = enc(data.cuda(non_blocking=True))
                f_k = cls(data.cuda(non_blocking=True))
                feature_list.append(f)
                feature_list_k.append(f_k)

            prediction = torch.argsort(cls(x[0].cuda(non_blocking=True)), dim=-1, descending=True)
            total_correct_1 += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

            cos_list = []
            cos_list_k = []
            cos_list_g = []
            for i in range(len(x)-1):
                for j in range(len(x)-1):
                    if i >= j:
                        continue
                    cos_list.append(similarity(feature_list[i], feature_list[j]))
                    cos_list_k.append(similarity(feature_list_k[i], feature_list_k[j]))
                    cos_list_g.append(similarity(feature_list_g[i], feature_list_g[j]))

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

            result_g = cos_list_g[0]
            for i in range(1,len(cos_list_g)):
                result_g += cos_list_g[i]
            result_g /= len(cos_list_g)

            for i in range(len(result)):
                if result[i] >= 0.94:
                    idx_09.append(i+counter)
                # if diff[i] > 0.2 and result[i] >= 0.94:
                    # idx.append(i+counter)

            train_feature_bank.append(result)
            train_feature_bank_k.append(result_k)
            train_feature_bank_g.append(result_g)
            counter += len(result)

        for x, target in tqdm(test_data_loader, desc='Feature extracting'):
                target = target.cuda()
                if test_counter > 10000:
                    break
                feature_list_k = []
                for data in x:
                    f_k = cls(data.cuda(non_blocking=True))
                    feature_list_k.append(f_k)

                prediction = torch.argsort(cls(x[0].cuda(non_blocking=True)), dim=-1, descending=True)
                test_total_correct_1 += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

                cos_list_k = []
                for i in range(len(x)-1):
                    for j in range(len(x)-1):
                        if i >= j:
                            continue
                        cos_list_k.append(similarity(feature_list_k[i], feature_list_k[j]))

                result_k = cos_list_k[0]
                for i in range(1,len(cos_list_k)):
                    result_k += cos_list_k[i]
                result_k /= len(cos_list_k)

                test_feature_bank_k.append(result_k)
                test_counter += len(result_k)

        # [D, N]
        test_feature_bank_k = torch.cat(test_feature_bank_k, dim=0).contiguous()
        train_feature_bank = torch.cat(train_feature_bank, dim=0).contiguous()
        train_feature_bank_k = torch.cat(train_feature_bank_k, dim=0).contiguous()
        train_feature_bank_g = torch.cat(train_feature_bank_g, dim=0).contiguous()
        train_var = torch.cat(train_var, dim=0)
        print(train_feature_bank)
        print('Accuracy enc dataset:', total_correct_1/counter, 'Accuracy dt dataset:', test_total_correct_1/test_counter)
        sorted_idx = torch.argsort(train_feature_bank, descending=True)

    bank_enc = []
    bank_cls = []
    for i in idx_09:
        if i >= len(train_feature_bank):
            break
        bank_enc.append(train_feature_bank[i])
        bank_cls.append(train_feature_bank_k[i])

    bank_enc_09 = []
    bank_enc_wo_09 = []
    bank_enc_g_09 = []
    bank_enc_g_wo_09 = []
    bank_cls_09 = []
    bank_cls_wo_09 = []
    for i, data in enumerate(train_feature_bank_k):
        if i in idx_09:
            bank_cls_09.append(data)
        else:
            bank_cls_wo_09.append(data)
    for i, data in enumerate(train_feature_bank):
        if i in idx_09:
            bank_enc_09.append(data)
        else:
            bank_enc_wo_09.append(data)
    for i, data in enumerate(train_feature_bank_g):
        if i in idx_09:
            bank_enc_g_09.append(data)
        else:
            bank_enc_g_wo_09.append(data)

    bank_cls_09 = torch.stack(bank_cls_09)
    bank_cls_wo_09 = torch.stack(bank_cls_wo_09)
    bank_enc_09 = torch.stack(bank_enc_09)
    bank_enc_wo_09 = torch.stack(bank_enc_wo_09)
    bank_enc_g_09 = torch.stack(bank_enc_g_09)
    bank_enc_g_wo_09 = torch.stack(bank_enc_g_wo_09)

    # Top-k
    bank_enc_09 = train_feature_bank[sorted_idx[:topk]]
    bank_enc_wo_09 = train_feature_bank[sorted_idx[topk:]]
    bank_enc_g_09 = train_feature_bank_g[sorted_idx[:topk]]
    bank_enc_g_wo_09 = train_feature_bank_g[sorted_idx[topk:]]
    bank_cls_09 = train_feature_bank_k[sorted_idx[:topk]]
    bank_cls_wo_09 = train_feature_bank_k[sorted_idx[topk:]]

    plt.title('Cosine Similarity')
    labels = ['>=0.9', '<0.9', 'Train CLS']
    # data = [bank_cls_09.to('cpu').detach().numpy().copy(), bank_cls_wo_09.to('cpu').detach().numpy().copy(), test_feature_bank_k.to('cpu').detach().numpy().copy()]
    data = [bank_cls_09.to('cpu').detach().numpy().copy(), bank_cls_wo_09.to('cpu').detach().numpy().copy()]
    plt.hist(data, 30, label=labels, stacked=True, range=(0.7, 1.0))
    plt.legend()
    plt.savefig("results/sim_dt.png")
    plt.close()
    data = [bank_enc_09.to('cpu').detach().numpy().copy(), bank_enc_wo_09.to('cpu').detach().numpy().copy()]
    plt.hist(data, 50, label=labels, stacked=True, range=(0.5, 1.0))
    plt.savefig("results/sim_orig.png")
    plt.close()
    data = [bank_enc_g_09.to('cpu').detach().numpy().copy(), bank_enc_g_wo_09.to('cpu').detach().numpy().copy()]
    plt.hist(data, 50, label=labels, stacked=True, range=(0.5, 1.0))
    plt.savefig("results/sim_orig_projection.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MoCo')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for each image')
    parser.add_argument('--batch_size', default=256, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--classes', default=10, type=int, help='the number of classes')
    parser.add_argument('--topk', default=500, type=int, help='top-k')
    parser.add_argument('--dataset', default='stl10', type=str, help='Training Dataset (e.g. CIFAR10, STL10)')
    parser.add_argument('--enc_path', type=str, default='results/128_4096_0.5_0.999_200_256_500_model.pth',
                        help='The pretrained model path')
    parser.add_argument('--linear_path', type=str, default='results/linear_model.pth',
                        help='The pretrained model path')
    parser.add_argument('--wandb_enc_runpath', default='', type=str, help='the runpath if using a model stored in WandB')
    parser.add_argument('--wandb_downstream_runpath', default='', type=str, help='the runpath if using a model stored in WandB')
    parser.add_argument('--wandb_project', default='default_project', type=str, help='WandB Project name')
    parser.add_argument('--wandb_run', default='default_run', type=str, help='WandB run name')

    # args parse
    args = parser.parse_args()
    feature_dim = args.feature_dim
    batch_size = args.batch_size

    # wandb init
    config = {
        "arch": "resnet34",
        "dataset": args.dataset,
        "batch_size": args.batch_size,
        "feature_dim": args.feature_dim,
    }
    wandb.init(project=args.wandb_project, name=args.wandb_run, config=config)

    # data prepare
    if args.dataset == 'stl10':
        memory_data = utils.STL10NAug(root='data', split='unlabeled', transform=utils.train_transform, download=True)
        test_data = utils.STL10NAug(root='data', split='test', transform=utils.train_transform, download=True)
    elif args.dataset == 'cifar10':
        memory_data = utils.CIFAR10NAug(root='data', train=True, transform=utils.train_transform, download=True, n=10)
        test_data = utils.CIFAR10NAug(root='data', train=False, transform=utils.train_transform, download=True, n=10)
    else:
        memory_data = utils.CIFAR100NAug(root='data', train=True, transform=utils.train_transform, download=True, n=10)
        test_data = utils.CIFAR100NAug(root='data', train=False, transform=utils.train_transform, download=True, n=10)
    # memory_data = utils.CIFAR100NAug(root='data', train=True, transform=utils.train_transform, download=True, n=10)
    # test_data = utils.STL10NAug(root='data', split='train', transform=utils.train_transform, download=True)
    memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    # model setup and optimizer config
    if args.wandb_enc_runpath != '':
        import os
        if os.path.exists(args.enc_path):
            os.remove(args.enc_path)
        base_enc = wandb.restore(args.enc_path, run_path=args.wandb_enc_runpath)
        model_path = base_enc.name
    if args.wandb_downstream_runpath != '':
        import os
        if os.path.exists(args.linear_path):
            os.remove(args.linear_path)
        cls_model = wandb.restore(args.linear_path, run_path=args.wandb_downstream_runpath)
        cls_model = cls_model.name

    # model setup and optimizer config
    enc = Model(feature_dim).cuda()
    enc.load_state_dict(torch.load(model_path))
    cls = Net(args.classes, model_path).cuda()
    cls.load_state_dict(torch.load(cls_model))

    sim(enc, cls, memory_loader, test_loader, topk=args.topk)
    # sim(model_q, memory_loader, memory_loader)

    # wandb finish
    os.remove(os.path.join(wandb.run.dir, args.enc_path))
    os.remove(os.path.join(wandb.run.dir, args.linear_path))
    wandb.save('results/sim_dt.png')
    wandb.save('results/sim_orig.png')
    wandb.save("results/sim_orig_projection.png")
    wandb.finish()

