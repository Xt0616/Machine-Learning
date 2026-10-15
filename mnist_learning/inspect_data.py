import sys
print("当前使用的 Python:", sys.executable)

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image


def main():
    # 1. 选择计算设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("计算设备:", device)

    # 2. 下载训练集，并将图片转换成张量
    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )

    print("训练集图片数量:", len(train_dataset))

    # 3. 查看一张图片及其标签
    image, label = train_dataset[0]

    print("第一张图片的形状:", image.shape)
    print("第一张图片的标签:", label)
    print("像素最小值:", image.min().item())
    print("像素最大值:", image.max().item())

    # 4. 将数据组织成批次，每批 64 张图片
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0,
    )

    images, labels = next(iter(train_loader))

    print("一批图片的形状:", images.shape)
    print("一批标签的形状:", labels.shape)
    print("前 16 张图片的标签:", labels[:16].tolist())

    # 5. 保存前 16 张图片，每行 8 张
    save_image(images[:16], "mnist_samples.png", nrow=8)
    print("图片已保存到 mnist_samples.png")

    # 6. 将这一批数据放到 GPU
    images = images.to(device)
    labels = labels.to(device)

    print("图片所在设备:", images.device)
    print("标签所在设备:", labels.device)


if __name__ == "__main__":
    main()