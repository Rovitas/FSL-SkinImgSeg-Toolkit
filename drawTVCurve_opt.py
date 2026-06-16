import os
import re
import matplotlib.pyplot as plt
import numpy as np

def smooth_loss_ewma(losses, alpha=0.3):
    """指数加权移动平均（EWMA）平滑"""
    smoothed = [losses[0]]
    for i in range(1, len(losses)):
        smoothed.append(alpha * losses[i] + (1 - alpha) * smoothed[-1])
    return np.array(smoothed)

def parse_log_file(log_path):
    """解析 training.log 文件"""
    epochs, train_losses, val_losses = [], [], []
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            losses = re.search(r'→ Train Loss: ([\d.]+) \| Val Loss: ([\d.]+)', line)
            if losses:
                epochs.append(len(train_losses) + 1)
                train_losses.append(float(losses.group(1)))
                val_losses.append(float(losses.group(2)))
    return epochs, train_losses, val_losses

def draw_crave(dicts, smooth_method='ewma', train_a=0.3, val_a=0.3, save_path=None, show_plot=True):
    # --- 颜色配置 ---
    # 为不同的 WD 配置分配颜色
    wd_colors = {
        '0': '#e41a1c',       # 红色 - WD=0
        '1e-5': '#377eb8',    # 蓝色 - WD=1e-5
        '5e-5': '#4daf4a',    # 绿色 - WD=5e-5
        '1e-4': '#984ea3'     # 紫色 - WD=1e-4
    }
    
    # --- 提取并整理数据 ---
    # 我们将数据整理为 {优化器: {wd值: [[loss1, loss2...], ...]}}
    data_dict = {'Adam': {}, 'AdamW': {}}
    
    for exp_name, log_paths in dicts.items():
        # 判断优化器类型
        if exp_name.startswith('AdamW'):
            opt = 'AdamW'
            # 提取 WD 值 (例如 "AdamW + WD=1e-5" -> '1e-5')
            wd_match = re.search(r'WD[=:]([0-9e\-.]+)', exp_name)
            wd_key = wd_match.group(1) if wd_match else '0'
        elif 'Adam' in exp_name:
            opt = 'Adam'
            wd_match = re.search(r'WD[=:]([0-9e\-.]+)', exp_name)
            wd_key = wd_match.group(1) if wd_match else '0'
        else:
            continue

        # 初始化列表
        if wd_key not in data_dict[opt]:
            data_dict[opt][wd_key] = {'train': [], 'val': []}

        # 收集日志数据
        for log_path in log_paths:
            full_path = os.path.join(r"D:\Work\Python\_MSDT\logs_summary", log_path)
            if not os.path.exists(full_path):
                print(f"路径不存在，已跳过: {full_path}")
                continue
                
            eps, train_l, val_l = parse_log_file(full_path)
            if train_l: # 确保有数据
                data_dict[opt][wd_key]['train'].append(train_l)
            if val_l:
                data_dict[opt][wd_key]['val'].append(val_l)

    # --- 创建画布 ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 13), constrained_layout=True)
    # plt.subplots_adjust(wspace=0.3, hspace=0.3) # 调整间距

    # --- 定义绘图逻辑 ---
    def plot_subplot(ax, opt_data, loss_type, title):
        ax.set_title(title, fontsize=20, fontweight='bold', pad=15)
        ax.set_xlabel('Epoch', fontsize=20)
        ax.set_ylabel('Loss', fontsize=20)
        ax.grid(True, linestyle='--', alpha=0.8)

        # 遍历该优化器下的所有 WD 配置
        for wd, losses_dict in opt_data.items():
            all_losses = losses_dict[loss_type]
            if not all_losses:
                continue
                
            # 1. 长度对齐
            max_epochs = max(len(l) for l in all_losses)
            padded_losses = []
            for l in all_losses:
                if len(l) < max_epochs:
                    # 填充最后一个值
                    padded_losses.append(l + [l[-1]] * (max_epochs - len(l)))
                else:
                    padded_losses.append(l)
            
            # 2. 计算均值
            avg_loss = np.mean(padded_losses, axis=0)
            epochs = np.arange(1, max_epochs + 1)
            
            # 3. 平滑
            if smooth_method == 'ewma':
                smoothed = smooth_loss_ewma(avg_loss, alpha=train_a if loss_type=='train' else val_a)
            else:
                smoothed = avg_loss

            # 4. 绘制曲线
            color = wd_colors.get(wd, '#000000') # 默认黑色
            label = f'WD={wd}'
            ax.plot(epochs, smoothed, label=label, color=color, linewidth=2.5, alpha=0.9)
            ax.tick_params(axis='both', labelsize=16)  # 设置x轴和y轴刻度数字的字体大小

        ax.legend(title='Weight Decay', fontsize=19, title_fontsize=19)

    # --- 绘制四张子图 ---
    
    # 1. Adam - Train
    plot_subplot(axes[0, 0], data_dict['Adam'], 'train', 'Adam: Training Loss')
    
    # 2. AdamW - Train
    plot_subplot(axes[0, 1], data_dict['AdamW'], 'train', 'AdamW: Training Loss')
    
    # 3. Adam - Validation
    plot_subplot(axes[1, 0], data_dict['Adam'], 'val', 'Adam: Validation Loss')
    
    # 4. AdamW - Validation
    plot_subplot(axes[1, 1], data_dict['AdamW'], 'val', 'AdamW: Validation Loss')

    # plt.suptitle('Optimizer Comparison (Adam vs AdamW) with Different Weight Decay', fontsize=16, fontweight='bold', y=0.98)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"✅ Saved plot to: {save_path}")
    if show_plot: plt.show()
    plt.close()

if __name__ == '__main__':
    experiments = {
        "Baseline (Adam + WD=0)": [
            r"Baseline_seed_38.log",
            r"Baseline_seed_39.log",
            r"Baseline_seed_40.log",
            r"Baseline_seed_48.log",
        ],
        "Adam + WD=1e-5": [
            r"Adam_wd_1e-05[Seed_48].log",
            r"Adam_wd_1e-05[Seed_50].log",
        ],
        "Adam + WD=5e-5": [
            r"Adam_wd_5e-05[Seed_48].log",
            r"Adam_wd_5e-05[Seed_49].log",
        ],
        "Adam + WD=1e-4": [
            r"Adam_wd_1e-04[Seed_48].log",
            r"Adam_wd_1e-04[Seed_49].log",
        ],
        "AdamW + WD=0": [
            r"AdamW_wd_0[Seed_48].log",
            r"AdamW_wd_0[Seed_49].log",
            r"AdamW_wd_0[Seed_50].log",
        ],
        "AdamW + WD=1e-5": [
            r"AdamW_wd_1e-05[Seed_48].log",
            r"AdamW_wd_1e-05[Seed_49].log",
            r"AdamW_wd_1e-05[Seed_50].log",
        ],
        "AdamW + WD=5e-5": [
            r"AdamW_wd_5e-05[Seed_48].log",
            r"AdamW_wd_5e-05[Seed_49].log",
            r"AdamW_wd_5e-05[Seed_50].log",
        ],
        "AdamW + WD=1e-4": [
            r"AdamW_wd_1e-04[Seed_48].log",
            r"AdamW_wd_1e-04[Seed_49].log",
            r"AdamW_wd_1e-04[Seed_50].log",
        ],
        # 可视情况解开其他配置...
    }
    IMG_PATH=r'D:\Study\毕业设计\相关图片\Curve.png'
    # IMG_PATH=None
    SHOW_FLAG=False
    # SHOW_FLAG=True
    draw_crave(experiments, smooth_method='ewma', save_path=IMG_PATH, show_plot=SHOW_FLAG, train_a=0.3, val_a=0.1)