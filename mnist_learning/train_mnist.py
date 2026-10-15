from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import SimpleMLP


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        # 1. 清除上一次计算留下的梯度
        optimizer.zero_grad()

        # 2. 前向传播，计算预测分数
        outputs = model(images)

        # 3. 计算损失
        loss = loss_fn(outputs, labels)

        # 4. 反向传播，计算每个参数的梯度
        loss.backward()

        # 5. 根据梯度更新参数
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


def evaluate(model, loader, loss_fn, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # 评估时只计算预测结果，不计算梯度
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("计算设备:", device)

    # 使用脚本所在目录，避免运行位置不同导致数据路径变化
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"
    checkpoint_dir = project_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = datasets.MNIST(
        root=str(data_dir),
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )

    test_dataset = datasets.MNIST(
        root=str(data_dir),
        train=False,
        download=True,
        transform=transforms.ToTensor(),
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

    model = SimpleMLP().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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

    checkpoint_path = checkpoint_dir / "simple_mlp.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print("权重已保存到:", checkpoint_path)


if __name__ == "__main__":
    main()