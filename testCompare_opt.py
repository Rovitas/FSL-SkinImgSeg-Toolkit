# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from myDataset import MYDataset
from myModel import UNet
import matplotlib.pyplot as plt

# ================= 全局配置区 =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOSS_FUNCTIONS_MAP = {
    'Baseline': "Adam_wd_1e-05", 
    'AdamW': "AdamW_wd_1e-05", 
    'BCE': "AdamW_wd_1e-05",
    'Dice': "DiceLoss", 
    'Focal': "FocalLoss(a0.25_g2)", 
    'Tversky': "TverskyLoss(a0.3_b0.7)",
    'Combo': "TverskyHD(a0.3_b0.7w0.8_0.2)", 
    'TverskyHD': "TverskyHD(a0.3_b0.7w0.8_0.2)single"
}
# ==============================================

# 模块 1：模型管家 (专门负责从硬盘加载和初始化模型)
def load_models(model_names, results_base_dir):
    """根据提供的名称列表，批量加载模型权重并返回模型字典"""
    models = {}
    for name in model_names:
        if name not in LOSS_FUNCTIONS_MAP:
            print(f"⚠️ 警告: {name} 不在映射表中，已跳过。")
            continue
            
        # 路径拼接逻辑独立出来，不依赖外部环境
        folder_name = f"{LOSS_FUNCTIONS_MAP[name]}[Seed_48]"
        model_path = os.path.join(results_base_dir, folder_name, "models", "best_model.pth")
        
        if os.path.exists(model_path):
            print(f"Loading {name}: {model_path}") 
            model = UNet().to(DEVICE)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True)['model_state_dict'])
            model.eval()
            models[name] = model
        else:
            print(f"❌ 找不到模型文件: {model_path}")
            
    return models

# 模块 2：画图引擎 (纯粹的画图逻辑，不沾染任何 PyTorch 的张量计算)
def plot_and_save_comparison(original_img, ground_truth, predictions_dict, save_path):
    """
    处理单张样本的绘图和保存
    original_img: numpy array (H, W, C)
    ground_truth: numpy array (H, W)
    predictions_dict: {'ModelName': numpy array (H, W)}
    """
    cols = len(predictions_dict) + 2
    plt.figure(figsize=(min(20, 3 * cols), 4))
    
    # 准备要画的数据队列
    plot_items = [
        ('Original Image', np.clip(original_img, 0, 1), None),
        ('Ground Truth', ground_truth, 'gray')
    ]
    plot_items.extend([(f'{name}', pred, 'gray') for name, pred in predictions_dict.items()])
    
    # 循环画图
    for i, (title, img, cmap) in enumerate(plot_items, 1):
        plt.subplot(1, cols, i)
        plt.title(title, pad=10, fontweight='bold')
        plt.imshow(img, cmap=cmap)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# 模块 3：核心业务流 (负责拉取数据、调用推理、分发画图任务)
def run_qualitative_analysis(test_loader, models_dict, save_dir):
    """执行定性分析的主循环"""
    if not models_dict:
        print("❌ 错误: 未提供任何可用模型!")
        return

    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # 推理并二值化转换为 Numpy
            preds_dict = {}
            for name, model in models_dict.items():
                pred = model(images)[0].cpu().numpy().squeeze()
                preds_dict[name] = (pred >= 0.5).astype(np.float32)
            
            # 提取原图和 GT 转换为 Numpy
            orig_img = images[0].cpu().numpy().transpose(1, 2, 0)
            gt_img = labels[0].cpu().numpy().squeeze()
            
            # 兼容处理：如果你未来的 dataset 去掉了 sample_id，这里不会报错
            sample_id = getattr(test_loader.dataset, 'sample_id', f'sample_{idx}')
            save_path = os.path.join(save_dir, f'comparison_{sample_id}.png')
            
            # 交给专门的画图函数去处理
            plot_and_save_comparison(orig_img, gt_img, preds_dict, save_path)
            # print(f"✅ Saved: comparison_{sample_id}.png")

# ================= 控制入口 =================
if __name__ == "__main__":
    # 1. 所有的硬编码路径统一提到最外层，方便后期维护
    TEST_DIR = r"d:\Work\Python\_MSDT\images_split\test"
    RESULTS_BASE_DIR = r"D:\Work\Python\_MSDT\saved_results"
    SAVE_DIR = r"D:\Work\Python\_MSDT\compare_img\Adam-AdamW"
    # SAVE_DIR = r"D:\Work\Python\_MSDT\compare_img\Adam-AdamW"
    # SAVE_DIR = r"D:\Work\Python\_MSDT\compare_img\XIAORONG"
    
    selected_losses = ['Baseline', 'AdamW', 'TverskyHD' ,'Combo']
    
    # 2. 初始化数据流
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    test_dataset = MYDataset(base_dir=TEST_DIR, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # 3. 组装并运行管线
    print("⏳ 开始加载模型...")
    loaded_models = load_models(selected_losses, RESULTS_BASE_DIR)
    
    print(f"🚀 开始生成对比图，保存至: {SAVE_DIR}")
    run_qualitative_analysis(test_loader, loaded_models, SAVE_DIR)
    print("🎉 定性分析完成！")