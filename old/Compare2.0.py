# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from myDataset import MYDataset
from myModel import UNet
import matplotlib.pyplot as plt

def test_multiple_loss_functions(base_dir, loss_functions):
    """
    对比多个损失函数的定性分析结果
    
    Args:
        base_dir: 测试数据集路径
        loss_functions: 损失函数列表，例如 ['BCE', 'Dice', 'Focal', 'Tversky', 'Combo']
    """
    # 定义训练和测试集的数据变换
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

    # 加载测试集
    test_dataset = MYDataset(base_dir=base_dir, transform=transform)
    # 定义数据加载器
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    LOSS_FUNCTIONS = {
        'Baseline': "Adam_wd_1e-05",
        'AdamW': "AdamW_wd_1e-05",
        'BCE': "AdamW_wd_1e-05",
        'Dice': "DiceLoss",
        'Focal': "FocalLoss(a0.25_g2)",
        'Tversky': "TverskyLoss(a0.3_b0.7)",
        'Combo': "TverskyHD(a0.3_b0.7w0.8_0.2)",
        'TverskyHD': "TverskyHD(a0.3_b0.7w0.8_0.2)single"
    }
    
    # 初始化模型字典
    models = {}
    
    # 加载每个损失函数对应的模型
    for loss_func in loss_functions:
        model = UNet()
        model_path = os.path.join(r"D:\Work\Python\_MSDT\saved_results", f"{LOSS_FUNCTIONS[loss_func]}[Seed_48]", "models\\best_model.pth")
        
        # 打印出它正在寻找的路径，你可以直接复制这个路径去资源管理器里搜
        print(f"正在寻找 {loss_func}: {model_path}") 

        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, weights_only=True)['model_state_dict'])
            model.eval()
            model.to(device)
            models[loss_func] = model
        else:
            print(f"警告: 找不到 {loss_func} 的模型文件: {model_path}")
    
    if not models:
        print("错误: 没有成功加载任何模型!")
        return
    
    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # 处理原始图像和标签
            original_image = images[0].cpu().numpy().transpose(1, 2, 0)
            label_image = labels[0].cpu().numpy().squeeze()
            
            # 获取所有模型的预测结果
            predictions = {}
            for loss_func, model in models.items():
                output = model(images)
                pred = output[0].cpu().numpy().squeeze()
                pred = (pred >= 0.5).astype(np.float32)
                predictions[loss_func] = pred
            
            # 创建对比图 - 根据损失函数数量动态调整布局
            num_losses = len(loss_functions)
            cols = num_losses + 2  # 原始图像 + 真实标签 + 所有预测结果
            fig_width = min(20, 3 * cols)  # 限制最大宽度
            
            plt.figure(figsize=(fig_width, 4))
            
            # 原始图像
            plt.subplot(1, cols, 1)
            plt.title('Original Image')
            plt.imshow(np.clip(original_image, 0, 1))
            plt.axis('off')
            
            # 真实标签
            plt.subplot(1, cols, 2)
            plt.title('Ground Truth')
            plt.imshow(label_image, cmap='gray')
            plt.axis('off')
            
            # 各个损失函数的预测结果
            for i, loss_func in enumerate(loss_functions):
                if loss_func in predictions:
                    plt.subplot(1, cols, i + 3)
                    plt.title(f'{loss_func} Loss')
                    plt.imshow(predictions[loss_func], cmap='gray')
                    plt.axis('off')
            
            plt.tight_layout()
            img_save_path = os.path.join(r"D:\Work\Python\_MSDT\compare_img\XIAORONG", f'comparison_{test_dataset.sample_id}.png')
            print(img_save_path)
            plt.savefig(img_save_path, dpi=300, bbox_inches='tight')
            # plt.show()
            plt.close()
            
            # # 可以限制显示的样本数量
            # if idx >= 4:  # 只显示前5个样本
            #     break

if __name__ == "__main__":
    # 新的训练集和测试集路径
    TEST_DIR = r"d:\Work\Python\_MSDT\images_split\test"
    
    # 定义要比较的5个损失函数
    loss_functions = ['BCE', 'Dice', 'Focal', 'Tversky', 'Combo']
    selected_losses=['Baseline', 'AdamW', 'TverskyHD' ,'Combo']
    
    # 开始测试模型
    test_multiple_loss_functions(base_dir=TEST_DIR, loss_functions=selected_losses)