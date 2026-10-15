from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from lenet import LeNet


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_dir = Path(__file__).resolve().parent

    # 1. 创建相同结构的模型
    model = LeNet().to(device)

    # 2. 加载训练好的参数
    checkpoint_path = project_dir / "checkpoints" / "lenet.pth"
    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.eval()

    print("已加载权重:", checkpoint_path)

    # 3. 加载测试集
    dataset = datasets.MNIST(
        root=str(project_dir / "data"),
        train=False,
        download=False,
        transform=transforms.Compose([
            transforms.Pad(2),
            transforms.ToTensor(),
        ]),
    )

    loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0,
    )

    total_correct = 0
    total_samples = 0
    wrong_images = []
    wrong_labels = []
    wrong_predictions = []

    # 4. 预测所有测试图片，并收集前 16 个错误案例
    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(loader):
            outputs = model(images.to(device))
            predictions = outputs.argmax(dim=1).cpu()

            if batch_index == 0:
                save_image(
                    images[:16],
                    project_dir / "lenet_test_samples.png",
                    nrow=8,
                )
                print("\n前 16 张图片的预测:", predictions[:16].tolist())
                print("前 16 张图片的标签:", labels[:16].tolist())

            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

            wrong_indices = (
                predictions != labels
            ).nonzero(as_tuple=True)[0]

            for index in wrong_indices.tolist():
                if len(wrong_images) >= 16:
                    break

                wrong_images.append(images[index])
                wrong_labels.append(labels[index].item())
                wrong_predictions.append(predictions[index].item())

    print(f"\n加载后的测试准确率: {total_correct / total_samples:.2%}")
    print("预测错误数量:", total_samples - total_correct)

    # 5. 将错误图片保存成网格
    if wrong_images:
        save_image(
            torch.stack(wrong_images),
            project_dir / "lenet_wrong_samples.png",
            nrow=8,
        )

        print("\n错误图片的真实标签:", wrong_labels)
        print("错误图片的预测结果:", wrong_predictions)
        print("错误图片已保存到 lenet_wrong_samples.png")


if __name__ == "__main__":
    main()