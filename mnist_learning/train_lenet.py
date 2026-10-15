from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from lenet import LeNet
from train_mnist import train_one_epoch, evaluate


def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("计算设备:", device)

    project_dir = Path(__file__).resolve().parent
    checkpoint_dir = project_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # MNIST 原图为 28×28，四周各补两圈零，得到 32×32
    transform = transforms.Compose([
        transforms.Pad(2),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.MNIST(
        root=str(project_dir / "data"),
        train=True,
        download=False,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        root=str(project_dir / "data"),
        train=False,
        download=False,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0,
    )

    model = LeNet().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("参数数量:", sum(p.numel() for p in model.parameters()))

    epochs = 5

    for epoch in range(epochs):
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device
        )

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"训练损失: {train_loss:.4f} | "
            f"训练准确率: {train_accuracy:.2%}"
        )

    test_loss, test_accuracy = evaluate(
        model, test_loader, loss_fn, device
    )

    print(
        f"\n测试损失: {test_loss:.4f} | "
        f"测试准确率: {test_accuracy:.2%}"
    )

    checkpoint_path = checkpoint_dir / "lenet.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print("权重已保存到:", checkpoint_path)


if __name__ == "__main__":
    main()