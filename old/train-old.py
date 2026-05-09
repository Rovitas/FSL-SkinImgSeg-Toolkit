# -*- coding: utf-8 -*-

import os
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from _MSDT.code.myDataset import MYDataset
from _MSDT.code.myModel import UNet
import matplotlib.pyplot as plt
from datetime import datetime

import time
from functools import wraps

def timer(
    enabled: bool = True,
    prefix: str = "⏱️",
    unit: str = "ms",          # 's', 'ms', 'us'
    precision: int = 6,
    output_func: callable = print
):
    # 单位换算系数
    unit_factors = {
        's': 1,
        'ms': 1000,
        'us': 1_000_000
    }
    if unit not in unit_factors:
        raise ValueError("unit 必须是 's', 'ms' 或 'us'")

    factor = unit_factors[unit]
    unit_label = {'s': '秒', 'ms': '毫秒', 'us': '微秒'}[unit]

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not enabled:
                return func(*args, **kwargs)

            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
            finally:
                end = time.perf_counter()
                elapsed = (end - start) * factor
                message = f"{prefix} 函数 {func.__name__} 耗时: {elapsed:.{precision}f} {unit_label}"
                output_func(message)
            return result
        return wrapper
    return decorator

@timer()
def train_model(model, criterion, optimizer, train_loader, num_epochs, device, base_dir=r'.\\' ):
    model = model.to(device)
    
    # 存储每个epoch的损失值
    train_losses = []
    # 设置一个存放模型的文件夹路径
    save_dir = os.path.join(base_dir, 'saved_results')
    loss_plot_path_dir = os.path.join(save_dir,  'saved_plots')
    model_path_dir = os.path.join(save_dir, 'saved_models')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  # 如果文件夹不存在，则创建
    if not os.path.exists(loss_plot_path_dir):
        os.makedirs(loss_plot_path_dir)  # 如果文件夹不存在，则创建
    if not os.path.exists(model_path_dir):
        os.makedirs(model_path_dir)  # 如果文件夹不存在，则创建

    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # 训练模式
        model.train()

        running_loss = 0.0
        total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)     # 对齐张量形状
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 统计损失和准确率
            running_loss += loss.item() * inputs.size(0)
            total += labels.size(0)

        # 计算并记录训练损失
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Train Loss: {epoch_loss:.4f} ')
        train_losses.append(epoch_loss)

        # 每 10 个 epoch 保存一次模型
        if (epoch + 1) % 10 == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # loss_str = f"{epoch_loss:.4f}"
            model_name = f"model_ep_{epoch+1}_in_{num_epochs}_{timestamp}.pth"
            model_path = os.path.join(model_path_dir ,model_name) # 将模型保存到指定文件夹
            torch.save(model.state_dict(), model_path)
            print(f'Saved Model at Epoch {epoch+1} as {model_name}')

        # 实时更新loss曲线
        
    plot_loss(train_losses)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_name = f"loss_curve_{num_epochs}_{timestamp}.png"
    loss_plot_path = os.path.join(loss_plot_path_dir ,img_name)
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    plt.close()  # 显示图像（可选）
    print(f"Loss curve saved to: {loss_plot_path}")


# 绘制实时loss曲线
def plot_loss(train_losses):
    # plt.figure(figsize=(10, 5))
    # plt.clf() 
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.pause(0.001)  # 暂停0.001秒以更新图形


# 开始训练模型
if __name__ == '__main__':
    # ===== CPU 优化 =====
    torch.set_num_threads(min(8, os.cpu_count() or 4))  # 控制最优的核心使用数

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_dir = r"d:\Work\Python\_MSDT" # os.path.abspath(__file__).split('code')[0]
    print(f"Using device: {device}")
    # 定义训练集的数据变换
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 调整图像大小
        transforms.ToTensor()           # 转换为Tensor
    ])
    # print(image_dir)
    # 加载训练集
    train_dataset = MYDataset(base_dir=image_dir, transform=transform)
    # 定义训练数据加载器
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=False)
    # 使用Unet
    model = UNet()
    # 定义损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()  # 二元交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_model(model, criterion, optimizer, train_loader, num_epochs=10, device=device, base_dir=image_dir)
    