import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, STL10
from tqdm import tqdm
import wandb

torch.backends.quantized.engine = 'x86'

from torch.ao.quantization import (
  default_dynamic_qconfig,
  get_default_qconfig_mapping,
  get_default_qat_qconfig_mapping,
  QConfigMapping,
)
import torch.ao.quantization.quantize_fx as quantize_fx
import copy

import utils
from model import Model, Classifier, TwoLayerClassifier


# train or test for one epoch
def train_val(net, data_loader, train_optimizer, device):
    is_train = train_optimizer is not None
    net.train() if is_train else net.eval()

    total_loss, total_correct_1, total_correct_5, total_num, data_bar = 0.0, 0.0, 0.0, 0, tqdm(data_loader)
    with (torch.enable_grad() if is_train else torch.no_grad()):
        for data, target in data_bar:
            if device == 'cuda':
                data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            # data, target = data.half(), target.half()
            out = net(data)
            # loss = loss_criterion(out, target)

            if is_train:
                train_optimizer.zero_grad()
                loss.backward()
                train_optimizer.step()

            total_num += data.size(0)
            # total_loss += loss.item() * data.size(0)
            train_loss = 0.1
            prediction = torch.argsort(out, dim=-1, descending=True)
            total_correct_1 += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_correct_5 += torch.sum((prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

            data_bar.set_description('{} Epoch: [{}/{}] Loss: {:.4f} ACC@1: {:.2f}% ACC@5: {:.2f}%'
                                     .format('Train' if is_train else 'Test', epoch, epochs, total_loss / total_num,
                                             total_correct_1 / total_num * 100, total_correct_5 / total_num * 100))

    return total_loss / total_num, total_correct_1 / total_num * 100, total_correct_5 / total_num * 100

def calibrate(model, loader):
    model.eval()
    bar = tqdm(loader)
    with torch.no_grad():
        for image, target in bar:
            image = image.cpu()
            image = image.cuda()
            model(image)

def quantize(net: nn.Module, test_loader):
    model_to_quantize = copy.deepcopy(net)
    # qconfig_mapping = get_default_qconfig_mapping("qnnpack")
    qconfig_mapping = get_default_qconfig_mapping("x86")
    # qconfig_mapping = QConfigMapping().set_global(default_dynamic_qconfig)
    model_to_quantize = model_to_quantize.cpu()
    model_to_quantize.eval()
    # prepare
    model_prepared = quantize_fx.prepare_fx(model_to_quantize, qconfig_mapping, next(iter(test_loader))[0])

    # calibrate (not shown)
    model_prepared = model_prepared.cuda()
    calibrate(model_prepared, test_loader)
    model_prepared = model_prepared.cpu()

    # quantize
    model_quantized = quantize_fx.convert_fx(model_prepared)
    # print(model_quantized.conv1.weight().dtype)
    save_quantize_model(model_quantized)
    return model_quantized

def save_quantize_model(model_quantized, path: str='static_quantize.pth'):
    torch.save(model_quantized.state_dict(), path)

def load_quantize_model(net, path: str, example_inputs):
    qconfig_mapping = get_default_qconfig_mapping("x86")
    net.eval()
    # prepare
    model_prepared = quantize_fx.prepare_fx(net, qconfig_mapping, example_inputs)
    # quantize
    model_quantized = quantize_fx.convert_fx(model_prepared)
    model_prepared.load_state_dict(torch.load(path, weights_only=True))

    return model_prepared


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Linear Evaluation')
    parser.add_argument('--batch_size', type=int, default=256, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', type=int, default=100, help='Number of sweeps over the dataset to train')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning Rate at the training start')
    parser.add_argument('--arch', default='one', type=str, help='Specify CLS Architecture one or two')
    parser.add_argument('--seed', default=42, type=int, help='specify static random seed')
    parser.add_argument('--weight_decay', default=1e-6, type=float, help='Weight Decay')
    parser.add_argument('--dataset', default='stl10', type=str, help='Training Dataset (e.g. CIFAR10, STL10)')
    parser.add_argument('--quantization_method', default='half', type=str, help='quantization method. (e.g. int8, half)')
    parser.add_argument('--wandb_model_runpath', default='', type=str, help='the runpath if using a model stored in WandB')
    parser.add_argument('--wandb_project', default='default_project', type=str, help='WandB Project name')
    parser.add_argument('--model_path', type=str, default='results/128_4096_0.5_0.999_200_256_500_model.pth',
                        help='The pretrained model path')
    parser.add_argument('--wandb_run', default='default_run', type=str, help='WandB run name')

    args = parser.parse_args()
    model_path, batch_size, epochs = args.model_path, args.batch_size, args.epochs
    lr, weight_decay = args.lr, args.weight_decay

    # initialize random seed
    utils.set_random_seed(args.seed)

    if args.wandb_model_runpath != '':
        import os
        if os.path.exists(args.model_path):
            os.remove(args.model_path)
        base_model = wandb.restore(args.model_path, run_path=args.wandb_model_runpath)
        model_path = base_model.name

    # wandb init
    config = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "arch": "resnet34",
        "dataset": args.dataset,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "model": model_path,
        "arch": args.arch,
        "seed": args.seed,
        "method": args.quantization_method,
        "wandb_model_runpath": args.wandb_model_runpath
    }
    wandb.init(project=args.wandb_project, name=args.wandb_run, config=config)

    if args.dataset == 'stl10':
        train_data = STL10(root='data', split='train', transform=utils.stl_train_ds_transform, download=True)
        test_data = STL10(root='data', split='test', transform=utils.stl_test_ds_transform, download=True)
    elif args.dataset == 'cifar10':
        train_data = CIFAR10(root='data', train=True, transform=utils.train_ds_transform, download=True)
        test_data = CIFAR10(root='data', train=False, transform=utils.test_transform, download=True)
    else:
        train_data = CIFAR100(root='data', train=True, transform=utils.train_ds_transform, download=True)
        test_data = CIFAR100(root='data', train=False, transform=utils.test_transform, download=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    if args.arch == 'one':
        print('CLS Architecture is specified a One Layer')
        model = Classifier(num_class=len(train_data.classes), pretrained_path=model_path).cuda()
    else:
        print('CLS Architecture is specified Two Layers')
        model = TwoLayerClassifier(num_class=len(train_data.classes), pretrained_path=model_path).cuda()
    for param in model.f.parameters():
        param.requires_grad = False

    # model setup and optimizer config
    if args.wandb_model_runpath != '':
        import os
        if os.path.exists(args.model_path):
            os.remove(args.model_path)
        base_model = wandb.restore(args.model_path, run_path=args.wandb_model_runpath)
        model_path = base_model.name
    model.load_state_dict(torch.load(model_path))

    optimizer = optim.Adam(model.fc.parameters(), lr=lr, weight_decay=weight_decay)
    loss_criterion = nn.CrossEntropyLoss()
    results = {'train_loss': [], 'train_acc@1': [], 'train_acc@5': [],
               'test_loss': [], 'test_acc@1': [], 'test_acc@5': []}

    epoch=1
    example_inputs = next(iter(test_loader))[0].cuda()
    print(model(example_inputs))
    btest_loss, btest_acc_1, btest_acc_5 = train_val(model, test_loader, None, 'cuda')
    # model = model.half()
    model = quantize(model, test_loader)
    best_acc = 0.0
    print(model(example_inputs.cpu()))
    atest_loss, atest_acc_1, atest_acc_5 = train_val(model, test_loader, None, 'cpu')
    wandb.log({'after_test_loss': atest_loss, 'after_test_acc@1': atest_acc_1, 'after_test_acc@5': atest_acc_5, 'before_test_loss': btest_loss, 'before_test_acc@1': btest_acc_1, 'before_test_acc@5': btest_acc_5})

    torch.save(model.state_dict(), f"results/quant_{args.quantization_method}_model.pth")

    wandb.save(f"results/quant_{args.quantization_method}_model.pth")
    wandb.finish()

