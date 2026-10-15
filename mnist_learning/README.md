# MNIST：MLP 与 LeNet 风格网络实验

## 1. 实验目标

掌握图像分类的完整流程：
数据加载、模型实现、训练、评估、权重保存与加载、错误案例分析。

## 2. 数据集

- 数据集：MNIST 手写数字
- 训练集：60,000 张
- 测试集：10,000 张
- 类别：数字 0–9
- 原始图片：单通道，28×28
- MLP 输入：像素缩放至 0–1
- LeNet 输入：四周各补 2 个像素，得到 32×32，再缩放至 0–1

## 3. 训练设置
Python: 3.11.4
PyTorch: 2.5.1+cu121
torchvision: 0.20.1+cu121
PyTorch CUDA: 12.1
GPU: NVIDIA GeForce RTX 4090
- 损失函数：交叉熵
- 优化器：Adam
- 学习率：0.001
- 训练批次大小：64
- 训练轮数：5
- 随机种子：42
- 从随机初始化开始训练
- 固定训练设置，训练结束后评估测试集

## 4. 实验结果

| 模型 | 参数数量 | 最后一轮训练准确率 | 测试准确率 | 测试错误数量 |
|---|---:|---:|---:|---:|
| MLP | 101,770 | 98.08% | 97.43% | 257 |
| LeNet 风格网络 | 61,706 | 98.63% | 98.85% | 115 |

LeNet 的测试准确率提高了 1.42 个百分点，
参数数量减少约 39.4%，测试错误减少 142 张。

上述结果来自各一次训练，尚未进行多个随机种子的实验。

## 5. 与原始 LeNet-5 的区别

本实验属于学习性实现，不是原论文指标的严格复现。

主要差异包括：
- 使用 ReLU 激活函数。
- 使用普通平均池化。
- 第二个卷积层使用完整通道连接。
- 使用交叉熵损失和 Adam 优化器。
- 输出层使用线性分类分数。

## 6. 文件说明

- inspect_data.py：查看数据和设备
- model.py：MLP 结构
- check_model.py：检查 MLP 前向传播
- train_mnist.py：训练 MLP，提供训练和评估函数
- predict_mnist.py：加载 MLP 权重并分析错误
- lenet.py：LeNet 风格结构
- train_lenet.py：训练 LeNet
- predict_lenet.py：加载 LeNet 权重并分析错误
- checkpoints/：保存模型参数

## 7. 运行命令

先激活环境并进入项目目录：

conda activate /data161/baoxt/envs/model
cd /data161/baoxt/models/mnist_learning

训练与预测：

python train_mnist.py
python predict_mnist.py
python train_lenet.py
python predict_lenet.py

## 8. 错误分析

查看 lenet_wrong_samples.png，并结合终端打印的标签，
记录 3 个具体错误案例：

1. 真实数字：___；预测数字：___；观察到的图片特点：___
2. 真实数字：___；预测数字：___；观察到的图片特点：___
3. 真实数字：___；预测数字：___；观察到的图片特点：___

## 9. 保存与加载验证

MLP 加载权重后的准确率：97.43%，与训练结束时一致。

LeNet 加载权重后的准确率：___。
是否与训练结束时的 98.85% 一致：___。