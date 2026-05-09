# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from myDataset import MYDataset
from myModel import UNet
import matplotlib.pyplot as plt


def test_model(base_dir, wd="0"):
    # 定义训练和测试集的数据变换
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

    # 加载测试集
    test_dataset = MYDataset(base_dir=base_dir, transform=transform)
    # 定义数据加载器
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_adam = UNet()
    model_adamw = UNet()

    Adam_model_path = os.path.join(r"D:\Work\Python\_MSDT\saved_results", f"Adam_wd_{wd}[Seed_48]", "models\\best_model.pth")
    AdamW_model_path = os.path.join(r"D:\Work\Python\_MSDT\saved_results", f"AdamW_wd_{wd}[Seed_48]", "models\\best_model.pth")

    model_adam.load_state_dict(torch.load(Adam_model_path, weights_only=True)['model_state_dict'])
    model_adamw.load_state_dict(torch.load(AdamW_model_path, weights_only=True)['model_state_dict'])
    
    model_adam.eval()
    model_adamw.eval()
    model_adam.to(device)
    model_adamw.to(device)
    
    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # 获取两个模型的预测结果
            outputs_adam = model_adam(images)
            outputs_adamw = model_adamw(images)
            
            # 处理单个样本
            image = images[0].cpu().numpy().transpose(1, 2, 0)
            label_image = labels[0].cpu().numpy().squeeze()
            
            output_adam = outputs_adam[0].cpu().numpy().squeeze()
            output_adam = (output_adam >= 0.5).astype(np.float32)
            
            output_adamw = outputs_adamw[0].cpu().numpy().squeeze()
            output_adamw = (output_adamw >= 0.5).astype(np.float32)
            
            # 创建对比图
            plt.figure(figsize=(12, 10))
            rows, cols = (1, 4)
            
            #原始图像（可选，如果需要显示原始图像）
            plt.subplot(rows, cols, 1)
            plt.title('Original Image')
            plt.imshow(np.clip(image, 0, 1))
            plt.axis('off')
            
            plt.subplot(rows, cols, 2)
            plt.title('Ground Truth')
            plt.imshow(label_image, cmap='gray')
            plt.axis('off')
            
            plt.subplot(rows, cols, 3)
            plt.title('Adam Prediction')
            plt.imshow(output_adam, cmap='gray')
            plt.axis('off')
            
            plt.subplot(rows, cols, 4)
            plt.title('AdamW Prediction')
            plt.imshow(output_adamw, cmap='gray')
            plt.axis('off')
            
            plt.tight_layout()
            img_save_path = os.path.join(r"D:\Work\Python\_MSDT\compare_img\Adam-AdamW", f'comparison_{test_dataset.sample_id}.png')
            # print(img_save_path)
            # plt.savefig(img_save_path, dpi=300, bbox_inches='tight')
            plt.show()
            plt.close() 
            
            # # 可以限制显示的样本数量
            # if idx >= 4:  # 只显示前5个样本
            #     break

if __name__ == "__main__":
    # 新的训练集和测试集路径
    TEST_DIR = r"d:\Work\Python\_MSDT\images_split\test"
    
    # 开始测试模型
    test_model(base_dir=TEST_DIR)