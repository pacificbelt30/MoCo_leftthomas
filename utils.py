import torch
import numpy as np
import random
import torchvision.datasets as datasets
from PIL import Image
from torchvision import transforms
import numpy as np
from typing import Optional, Callable

# This is a quote from https://github.com/THUYimingLi/Untargeted_Backdoor_Watermark/blob/main/UBW-C/UBW_C.py.
def set_random_seed(seed=42):
    print('SET RANDOM SEED to', seed)
    torch.manual_seed(seed + 1)
    torch.cuda.manual_seed(seed + 2)
    torch.cuda.manual_seed_all(seed + 3)
    np.random.seed(seed + 4)
    torch.cuda.manual_seed_all(seed + 5)
    random.seed(seed + 6)

class CIFAR10Pair(datasets.CIFAR10):
    """CIFAR10 Dataset.
    """

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target

class CIFAR100Pair(datasets.CIFAR100):
    """CIFAR100 Dataset.
    """

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target

class STL10Pair(datasets.STL10):
    """STL10 Dataset.
    """

    def __getitem__(self, index):
        img, target = self.data[index], 0
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target

    def __len__(self):
        if self.split == 'train+unlabeled' or self.split == 'unlabeled':
            return 50000
        else:
            return self.data.shape[0]
        

class CIFAR10NAug(datasets.CIFAR10):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        n: int = 10,
    ) -> None:
        super().__init__(root, train, transform, target_transform, download)
        self.n = n

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        pos = []
        if self.transform is not None:
            for i in range(self.n):
                pos.append(self.transform(img))

        return pos, target

class CIFAR100NAug(datasets.CIFAR100):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        n: int = 10,
    ) -> None:
        super().__init__(root, train, transform, target_transform, download)
        self.n = n

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        pos = []
        if self.transform is not None:
            for i in range(self.n):
                pos.append(self.transform(img))

        return pos, target

class STL10NAug(datasets.STL10):
    """STL10 Dataset.
    """

    def __getitem__(self, index):
        if not self.is_mia_train_dataset:
            index = index + 50000
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        pos = []
        if self.transform is not None:
            for i in range(10):
                pos.append(self.transform(img))

        return pos, target

    def set_mia_train_dataset_flag(self, flag):
        self.is_mia_train_dataset = flag

    def __len__(self):
        if self.split == 'train+unlabeled' or self.split == 'unlabeled':
            return 50000
        else:
            return self.data.shape[0]

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

stl_train_transform = transforms.Compose([
    transforms.Resize(32),
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

stl_test_transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

train_ds_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

stl_train_ds_transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

stl_test_ds_transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])


available_dataset = {
    'cifar10': CIFAR10Pair,
    'cifar100': CIFAR100Pair,
    'stl10': STL10Pair
}
