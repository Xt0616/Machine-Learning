import torch
from torch import nn


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def main():
    # 先用模拟数据检查结构，无需下载数据或使用 GPU
    model = LeNet()
    x = torch.zeros(64, 1, 32, 32)

    print("输入:", x.shape)

    with torch.no_grad():
        for layer in model.features:
            x = layer(x)
            print(f"{layer.__class__.__name__:12s} -> {list(x.shape)}")

        for layer in model.classifier:
            x = layer(x)
            print(f"{layer.__class__.__name__:12s} -> {list(x.shape)}")

    parameter_count = sum(p.numel() for p in model.parameters())
    print("\n参数数量:", parameter_count)


if __name__ == "__main__":
    main()