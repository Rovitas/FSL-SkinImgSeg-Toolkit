# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from myDataset import MYDataset
from myModel import UNet
import matplotlib.pyplot as plt

# 测试模型
def test_model(test_loader, task_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet()
    model.eval()  # 设置为评估模式
    model_path = os.path.join(r"D:\Work\Python\_MSDT\saved_results", task_name, "models\\best_model.pth")
    checkpoint = torch.load(model_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            print("outputs.size():")
            print(outputs.size())
            # 这里可以展示输出的图像
            for t in range(outputs.size()[0]):
                image = images[t].cpu().numpy().transpose(1, 2, 0)
                output_image = outputs[t].cpu().numpy().squeeze()
                output_image = (output_image >= 0.5).astype(np.float32)
                label_image = labels[t].cpu().numpy().squeeze()

                plt.figure(figsize=(10, 5))
                #原始图像（可选，如果需要显示原始图像）
                plt.subplot(2, 2, 1)
                plt.title('Original Image')
                plt.imshow(np.clip(image, 0, 1))
                plt.axis('off')

                plt.subplot(1, 2, 1)
                plt.title('Predicted Segmentation')
                plt.imshow(output_image, cmap='gray')

                plt.subplot(1, 2, 2)
                plt.title('Ground Truth Label')
                plt.imshow(label_image, cmap='gray')
                plt.show()
                plt.close() 

if __name__ == "__main__":
    # 定义训练和测试集的数据变换
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

    # 新的训练集和测试集路径
    TEST_DIR = r"d:\Work\Python\_MSDT\images_split\test"
    # 加载测试集
    test_dataset = MYDataset(base_dir=TEST_DIR, transform=transform)
    # 定义数据加载器
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 开始测试模型
    test_model(test_loader=test_loader, task_name="AdamW_wd_0[Seed_48]")
