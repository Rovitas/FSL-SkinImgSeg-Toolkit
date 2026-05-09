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

def draw_crave(dicts, smooth_method='ewma', train_a=0.3, val_a=0.3, save_path=None, figsize=(12, 6), show_plot=True):
    plt.figure(figsize=figsize)
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']
    
    for idx, (exp_name, log_paths) in enumerate(dicts.items()):
        all_train_losses, all_val_losses = [], []
        
        for log_path in log_paths:
            if not os.path.exists(log_path):
                continue
            _, train_l, val_l = parse_log_file(log_path)
            all_train_losses.append(train_l)
            all_val_losses.append(val_l)
            
        if not all_train_losses: continue
        max_epochs = max(len(l) for l in all_train_losses)
        
        # 长度对齐与均值计算
        padded_trains = [l + [l[-1]] * (max_epochs - len(l)) for l in all_train_losses]
        padded_vals = [l + [l[-1]] * (max_epochs - len(l)) for l in all_val_losses]
        
        avg_train = np.mean(padded_trains, axis=0)
        avg_val = np.mean(padded_vals, axis=0)
        epochs = np.arange(1, max_epochs + 1)
        
        # 平滑处理
        if smooth_method == 'ewma':
            train_smooth = smooth_loss_ewma(avg_train, alpha=train_a)
        else:
            train_smooth = avg_train
            
        label = f"{exp_name} (Train)" if len(all_train_losses) == 1 else f"{exp_name} (Train, avg)"
        plt.plot(epochs, train_smooth, label=label, color=colors[idx % len(colors)], linestyle='-', linewidth=2.0, alpha=0.9)
    
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Loss', fontsize=12, fontweight='bold')
    plt.title('Training Loss Curves', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7, linewidth=0.8)
    plt.legend(fontsize=10, loc='upper right', frameon=True, fancybox=True, shadow=True)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    if show_plot: plt.show()
    else: plt.close()

if __name__ == '__main__':
    experiments = {
        "Baseline (Adam + WD=0)": [
            r"D:\Work\Python\_MSDT\saved_results\Baseline_seed_38\Baseline_seed_38.log",
            r"D:\Work\Python\_MSDT\saved_results\Baseline_seed_39\Baseline_seed_39.log",
            r"D:\Work\Python\_MSDT\saved_results\Baseline_seed_40\Baseline_seed_40.log",
            r"D:\Work\Python\_MSDT\saved_results\Baseline_seed_48\Baseline_seed_48.log",
        ],
        "Adam + WD=1e-5": [
            r"D:\Work\Python\_MSDT\saved_results\Adam_wd_1e-05[Seed_48]\Adam_wd_1e-05[Seed_48].log",
            r"D:\Work\Python\_MSDT\saved_results\Adam_wd_1e-05[Seed_50]\Adam_wd_1e-05[Seed_50].log",
        ],
        # 可视情况解开其他配置...
    }
    draw_crave(experiments, smooth_method='ewma', train_a=0.3, val_a=0.1)