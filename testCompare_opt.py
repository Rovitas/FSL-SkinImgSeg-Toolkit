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

# 字体与布局控制
TITLE_FONT_SIZE = 14      # 每张子图标题的字体大小
FIG_HEIGHT = 5            # 画布的高度 (单位：英寸)
WIDTH_PER_COL = 3.5       # 每一列图像占用的宽度 (单位：英寸)
# ============================================

# 模块 1: 模型加载引擎 (精确匹配)
def load_selected_models(experiments_dict, results_base_dir):
    """
    根据配置字典中的精确文件夹名称，直接加载对应的模型权重
    """
    models = {}
    
    for display_name, exact_folder_name in experiments_dict.items():
        # 精确拼接路径
        model_path = os.path.join(results_base_dir, exact_folder_name, "models", "best_model.pth")
        
        if os.path.exists(model_path):
            print(f"Loading {display_name}: {model_path}") 
            model = UNet().to(DEVICE)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True)['model_state_dict'])
            model.eval()
            models[display_name] = model
        else:
            print(f"❌ 警告: 找不到指定的模型文件: {model_path}")
            
    return models

# 模块 2: 定性绘图引擎
def plot_qualitative_comparison(original_img, ground_truth, preds_dict, save_path, 
                                title_fs=TITLE_FONT_SIZE, fig_h=FIG_HEIGHT, w_per_col=WIDTH_PER_COL):
    """
    负责单张对比图的排版与保存
    """
    # 计算列数：原图 + GT + 各个模型结果
    n_cols = len(preds_dict) + 2
    total_width = n_cols * w_per_col
    
    # 动态创建画布大小
    fig, axes = plt.subplots(1, n_cols, figsize=(total_width, fig_h))
    
    # 准备待显示的数据列表
    items = [
        ('Original Image', np.clip(original_img, 0, 1), None),
        ('Ground Truth', ground_truth, 'gray')
    ]
    items.extend([(f'{name}', pred, 'gray') for name, pred in preds_dict.items()])
    
    # 循环渲染子图
    for i, (title, img, cmap) in enumerate(items):
        ax = axes[i]
        ax.set_title(title, fontsize=title_fs, fontweight='bold', pad=12)
        ax.imshow(img, cmap=cmap)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# 模块 3: 业务流水线
def run_qualitative_pipeline(test_loader, models_dict, save_dir):
    """
    遍历测试集并分发绘图任务
    """
    if not models_dict:
        print("❌ 错误: 没有成功加载任何可用的模型。")
        return

    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # 批量获取所有模型对该图的预测结果
            current_preds = {}
            for name, model in models_dict.items():
                out = model(images)[0].cpu().numpy().squeeze()
                current_preds[name] = (out >= 0.5).astype(np.float32)
            
            # 转换原图和标签为 Numpy 格式
            raw_img = images[0].cpu().numpy().transpose(1, 2, 0)
            gt_mask = labels[0].cpu().numpy().squeeze()
            
            # 确定保存文件名
            sample_name = getattr(test_loader.dataset, 'sample_id', f'img_{idx}')
            save_name = os.path.join(save_dir, f'comparison_{sample_name}.png')
            
            # 调用绘图引擎
            plot_qualitative_comparison(raw_img, gt_mask, current_preds, save_name)

# ================= 核心控制入口 =================
if __name__ == "__main__":
    # --- 1. 路径设置 ---
    TEST_DIR = r"d:\Work\Python\_MSDT\images_split\test"
    RESULTS_ROOT = r"D:\Work\Python\_MSDT\saved_results"
    SAVE_FOLDER = r"D:\Work\Python\_MSDT\compare_img\Losses_Comparison"
    
    # --- 2. 在这里配置你需要对比的模型 (精确匹配) ---
    # 格式: '图表上显示的简称': '对应的完整精确文件夹名'
    EXPERIMENTS_TO_COMPARE = {
        'Baseline': "Adam_wd_1e-05[Seed_48]",          
        'BCE': "AdamW_wd_1e-05[Seed_48]",              
        'Dice': "DiceLoss[Seed_48]",
        'Focal': "FocalLoss(a0.25_g2)[Seed_48]",
        'Tversky': "TverskyLoss(a0.3_b0.7)[Seed_48]",
        # 'TverskyHD(0.5:0.5)': "TverskyHD(a0.3_b0.7w0.5_0.5)[Seed_48]",
        'TverskyHD(0.8:0.2)': "TverskyHD(a0.3_b0.7w0.8_0.2)[Seed_48]",
        # 'TverskyHD(0.9:0.1)': "TverskyHD(a0.3_b0.7w0.9_0.1)[Seed_48]",
        # 'Combo': "TverskyHD(a0.3_b0.7w0.8_0.2)[Seed_48]"
    }
    
    # --- 3. 初始化数据加载 ---
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    test_set = MYDataset(base_dir=TEST_DIR, transform=transform)
    loader = DataLoader(test_set, batch_size=1, shuffle=False)
    
    # --- 4. 运行流水线 ---
    print("⏳ 正在根据配置精确加载模型...")
    active_models = load_selected_models(EXPERIMENTS_TO_COMPARE, RESULTS_ROOT)
    
    print(f"\n🚀 开始定性对比分析，保存路径: {SAVE_FOLDER}")
    run_qualitative_pipeline(loader, active_models, SAVE_FOLDER)
    
    print("\n🎉 定性分析完成！请检查输出文件夹。")