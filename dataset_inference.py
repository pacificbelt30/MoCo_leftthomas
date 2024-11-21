import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.mixture import GaussianMixture
from scipy.stats import ttest_ind


def verify_model(model, trainloader, valloader, testloader, n_components: int=50, covariance: str='diag'):
    """
    PyTorch モデルの検証を行う関数

    Args:
        model: nn.Module クラスの重みがロードされたモデル

    Returns:
        p_value: t検定の p値
        effect_size: t検定の効果量
    """

    # モデルを評価モードに設定
    model.eval()

    # 表現の抽出
    train_representations = []
    val_representations = []
    test_representations = []

    with torch.no_grad():
        for data in trainloader:
            images, _ = data
            representations = model(images)
            train_representations.append(representations)

        for data in valloader:
            images, _ = data
            representations = model(images)
            val_representations.append(representations)

        for data in testloader:
            images, _ = data
            representations = model(images)
            test_representations.append(representations)

    train_representations = torch.cat(train_representations).numpy()
    val_representations = torch.cat(val_representations).numpy()
    test_representations = torch.cat(test_representations).numpy()

    # 密度推定器の作成 (ここではガウス混合モデルを使用)
    gmm = GaussianMixture(n_components=n_components, covariance_type=covariance)
    gmm.fit(train_representations)

    # 対数尤度の計算
    val_log_likelihood = gmm.score_samples(val_representations)
    test_log_likelihood = gmm.score_samples(test_representations)

    # t検定の実行
    t_statistic, p_value = ttest_ind(val_log_likelihood, test_log_likelihood)
    effect_size = (np.mean(val_log_likelihood) - np.mean(test_log_likelihood)) / np.std(test_log_likelihood)

    return p_value, effect_size

if __name__ == '__main__':
    # CIFAR-10 データセットの読み込み
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)

    # 学習データを訓練用と検証用に分割
    train_size = int(0.5 * len(trainset))
    val_size = len(trainset) - train_size
    generator = torch.Generator().manual_seed(42)
    trainset, valset = torch.utils.data.random_split(trainset, [train_size, val_size], generator)

    # DataLoader の作成
    trainloader = DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2)
    valloader = DataLoader(valset, batch_size=256, shuffle=False, num_workers=2)
    testloader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)

    # モデルの読み込み (ここでは ResNet18 を使用)
    model = torchvision.models.resnet18(pretrained=True)

    # 検証の実行
    """
    Classifier:
        n_components: 10
        covariance: full
    Encoder:
        n_components: 50
        covariance: diag
    """
    p_value, effect_size = verify_model(model, trainloader, valloader, testloader, 50, 'diag')

    # 結果の出力
    print(f'p値: {p_value}')
    print(f'効果量: {effect_size}')

