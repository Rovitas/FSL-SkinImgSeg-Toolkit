# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from myDataset import MYDataset
from myModel import UNet
import matplotlib.pyplot as plt

# --- 全局配置区 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOSS_FUNCTIONS_MAP = {
    'Baseline': "Adam_wd_1e-05", 'AdamW': "AdamW_wd_1e-05", 'BCE': "AdamW_wd_1e-05",
    'Dice': "DiceLoss", 'Focal': "FocalLoss(a0.25_g2)", 'Tversky': "TverskyLoss(a0.3_b0.7)",
    'Combo': "TverskyHD(a0.3_b0.7w0.8_0.2)", 'TverskyHD': "TverskyHD(a0.3_b0.7w0.8_0.2)single"
}

def test_multiple_loss_functions(base_dir, loss_functions):
    test_loader = DataLoader(MYDataset(base_dir=base_dir, transform=transforms.Compose([
        transforms.Resize((256, 256)), transforms.ToTensor()
    ])), batch_size=1, shuffle=False)

    models = {}
    for loss_func in loss_functions:
        if loss_func not in LOSS_FUNCTIONS_MAP: continue
        
        model_path = os.path.join(r"D:\Work\Python\_MSDT\saved_results", f"{LOSS_FUNCTIONS_MAP[loss_func]}[Seed_48]", "models", "best_model.pth")
        print(f"Loading {loss_func}: {model_path}") 
        
        if os.path.exists(model_path):
            model = UNet().to(DEVICE)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True)['model_state_dict'])
            model.eval()
            models[loss_func] = model

    if not models:
        print("❌ 错误: 未加载任何模型!")
        return
    
    save_dir = r"D:\Work\Python\_MSDT\compare_img\XIAORONG"
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # 推理并二值化
            preds = {name: (model(images)[0].cpu().numpy().squeeze() >= 0.5).astype(np.float32) 
                     for name, model in models.items()}
            
            # 绘图布局动态计算
            cols = len(models) + 2
            plt.figure(figsize=(min(20, 3 * cols), 4))
            
            # 原图与标签
            plot_items = [
                ('Original Image', np.clip(images[0].cpu().numpy().transpose(1, 2, 0), 0, 1), None),
                ('Ground Truth', labels[0].cpu().numpy().squeeze(), 'gray')
            ]
            # 预测图加入列表
            plot_items.extend([(f'{name} Loss', pred, 'gray') for name, pred in preds.items()])
            
            for i, (title, img, cmap) in enumerate(plot_items, 1):
                plt.subplot(1, cols, i)
                plt.title(title)
                plt.imshow(img, cmap=cmap)
                plt.axis('off')
            
            plt.tight_layout()
            img_save_path = os.path.join(save_dir, f'comparison_{test_loader.dataset.sample_id}.png')
            plt.savefig(img_save_path, dpi=300, bbox_inches='tight')
            plt.close()

if __name__ == "__main__":
    TEST_DIR = r"d:\Work\Python\_MSDT\images_split\test"
    selected_losses = ['Baseline', 'AdamW', 'TverskyHD' ,'Combo']
    test_multiple_loss_functions(base_dir=TEST_DIR, loss_functions=selected_losses)