import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import SimpleMLP


def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=False,
        transform=transforms.ToTensor(),
    )

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )

    images, labels = next(iter(loader))
    images = images.to(device)
    labels = labels.to(device)

    # 创建模型，并将模型参数放到计算设备上
    model = SimpleMLP().to(device)

    print("模型结构:")
    print(model)

    # 前向传播：从图片计算预测分数
    outputs = model(images)

    print("\n输入形状:", images.shape)
    print("输出形状:", outputs.shape)
    print("第一张图片的十个分数:")
    print(outputs[0].detach().cpu())

    # 每行最大分数所在的位置，就是预测类别
    predictions = outputs.argmax(dim=1)

    print("\n前 10 张图片的预测:", predictions[:10].tolist())
    print("前 10 张图片的标签:", labels[:10].tolist())

    # 根据预测分数和正确标签计算损失
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(outputs, labels)

    print("\n这一批数据的损失:", loss.item())

    parameter_count = sum(p.numel() for p in model.parameters())
    print("模型参数数量:", parameter_count)


if __name__ == "__main__":
    main()