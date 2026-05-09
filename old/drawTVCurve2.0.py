import os
import re
import matplotlib.pyplot as plt
import numpy as np

def smooth_loss_ewma(losses, alpha=0.3):
    """
    指数加权移动平均（EWMA）平滑
    alpha ∈ (0,1]: 越小越平滑，越大越接近原始数据
    """
    smoothed = [losses[0]]
    for i in range(1, len(losses)):
        smoothed.append(alpha * losses[i] + (1 - alpha) * smoothed[-1])
    return np.array(smoothed)

def draw_crave(
    dicts,
    smooth_method='ewma',
    train_a=0.3,
    val_a=0.3,
    save_path=None,
    figsize=(12, 6),
    show_plot=True
):

    plt.figure(figsize=figsize)
    
    # 专业配色方案（ColorBrewer Set1）
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', 
              '#ff7f00', '#ffff33', '#a65628', '#f781bf']
    markers = ['o', 's', '^', 'D', 'v', '*', 'x', '+']
    
    for idx, (exp_name, log_paths) in enumerate(dicts.items()):
        color = colors[idx % len(colors)]
        
        all_train_losses = []
        all_val_losses = []
        max_epochs = 0
        
        # 解析所有日志文件
        for log_path in log_paths:
            if not os.path.exists(log_path):
                print(f"⚠️ Warning: {log_path} not found, skipping.")
                continue
            epochs, train_losses, val_losses = parse_log_file(log_path)
            all_train_losses.append(train_losses)
            all_val_losses.append(val_losses)
            max_epochs = max(max_epochs, len(train_losses))
        
        if not all_train_losses:
            continue
        
        # === 单次运行 vs 多次运行 ===
        if len(all_train_losses) == 1:
            # 单次运行：直接平滑
            train_loss = np.array(all_train_losses[0])
            val_loss = np.array(all_val_losses[0])
            epochs = np.arange(1, len(train_loss) + 1)
            
            if smooth_method == 'ewma':
                train_smooth = smooth_loss_ewma(train_loss, alpha=train_a)
                val_smooth = smooth_loss_ewma(val_loss, alpha=val_a)
            else:
                train_smooth, val_smooth = train_loss, val_loss
                
            label_train = f"{exp_name} (Train)"
            label_val = f"{exp_name} (Val)"
            
        else:
            # 多次运行：先平均再平滑
            def pad_to_max(lst, max_len):
                # print(len(lst), len(lst)) 
                return lst + [lst[-1]] * (max_len - len(lst))
            
            padded_trains = [pad_to_max(l, max_epochs) for l in all_train_losses]
            padded_vals = [pad_to_max(l, max_epochs) for l in all_val_losses]
            
            avg_train = np.mean(padded_trains, axis=0)
            avg_val = np.mean(padded_vals, axis=0)
            epochs = np.arange(1, max_epochs + 1)
            
            if smooth_method == 'ewma':
                train_smooth = smooth_loss_ewma(avg_train, alpha=train_a)
                val_smooth = smooth_loss_ewma(avg_val, alpha=val_a)
            else:
                train_smooth, val_smooth = avg_train, avg_val
                
            label_train = f"{exp_name} (Train, avg)"
            label_val = f"{exp_name} (Val, avg)"
        
        # === 绘制平滑曲线 ===
        plt.plot(epochs, train_smooth, 
                 label=label_train,
                 color=color, linestyle='-', linewidth=2.0, alpha=0.9)
        # plt.plot(epochs, val_smooth, 
        #          label=label_val,
        #          color=color, linestyle='--', linewidth=2.0, alpha=0.9)
    
    # === 图表美化 ===
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Loss', fontsize=12, fontweight='bold')
    # plt.title('Training and Validation Loss Curves', fontsize=14, fontweight='bold')
    plt.title('Training Loss Curves', fontsize=14, fontweight='bold')
    # plt.title('Validation Loss Curves', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7, linewidth=0.8)
    plt.legend(fontsize=10, loc='upper right', frameon=True, fancybox=True, shadow=True)
    plt.tight_layout()
    
    # === 保存矢量图 ===
    if save_path:
        plt.savefig(save_path, 
                   bbox_inches='tight', 
                   dpi=300,
                   format=save_path.split('.')[-1])
        print(f"✅ Saved vector plot to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

# =============================================================================
# 辅助函数：解析日志文件（你需要根据实际日志格式实现）
# =============================================================================
def parse_log_file(log_path):
    """
    解析单个 training.log 文件，返回 epochs, train_losses, val_losses
    """
    epochs = []
    train_losses = []
    val_losses = []

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 匹配日志行：→ Train Loss: 0.428365 | Val Loss: 0.469316
            losses = re.search(r'→ Train Loss: ([\d.]+) \| Val Loss: ([\d.]+)', line)
            if losses:
                train_loss = float(losses.group(1))
                val_loss = float(losses.group(2))
                # 当前 epoch = 已记录数量 + 1
                epoch = len(train_losses) + 1
                epochs.append(epoch)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
    return epochs, train_losses, val_losses


# =============================================================================
# 使用示例
# =============================================================================
if __name__ == '__main__':
    # 配置实验
    experiments = {
       "Baseline (Adam + WD=0)": [
            r"D:\Work\Python\_MSDT\saved_results\Baseline_seed_38\Baseline_seed_38.log",   # seed38
            r"D:\Work\Python\_MSDT\saved_results\Baseline_seed_39\Baseline_seed_39.log",   # seed39
            r"D:\Work\Python\_MSDT\saved_results\Baseline_seed_40\Baseline_seed_40.log",   # seed40
            r"D:\Work\Python\_MSDT\saved_results\Baseline_seed_48\Baseline_seed_48.log",   # seed48
        ],
        "Adam + WD=1e-5": [
            r"D:\Work\Python\_MSDT\saved_results\Adam_wd_1e-05[Seed_48]\Adam_wd_1e-05[Seed_48].log",   # seed48
            r"D:\Work\Python\_MSDT\saved_results\Adam_wd_1e-05[Seed_50]\Adam_wd_1e-05[Seed_50].log",   # seed50
        ],
        "Adam + WD=5e-5": [
            r"D:\Work\Python\_MSDT\saved_results\Adam_wd_5e-05[Seed_48]\Adam_wd_5e-05[Seed_48].log",   # seed48
            r"D:\Work\Python\_MSDT\saved_results\Adam_wd_5e-05[Seed_49]\Adam_wd_5e-05[Seed_49].log",   # seed49
        ],
        "Adam + WD=1e-4": [
            r"D:\Work\Python\_MSDT\saved_results\Adam_wd_1e-04[Seed_48]\Adam_wd_1e-04[Seed_48].log",   # seed48
            r"D:\Work\Python\_MSDT\saved_results\Adam_wd_1e-04[Seed_49]\Adam_wd_1e-04[Seed_49].log",   # seed48
        ],
        # "AdamW + WD=0": [
        #     r"D:\Work\Python\_MSDT\saved_results\AdamW_wd_0[Seed_48]\AdamW_wd_0[Seed_48].log",   # seed48
        #     r"D:\Work\Python\_MSDT\saved_results\AdamW_wd_0[Seed_49]\AdamW_wd_0[Seed_49].log",   # seed49
        #     r"D:\Work\Python\_MSDT\saved_results\AdamW_wd_0[Seed_50]\AdamW_wd_0[Seed_50].log",   # seed50
        # ],
        # "AdamW + WD=1e-5": [
        #     r"D:\Work\Python\_MSDT\saved_results\AdamW_wd_1e-05[Seed_48]\AdamW_wd_1e-05[Seed_48].log",   # seed48
        #     r"D:\Work\Python\_MSDT\saved_results\AdamW_wd_1e-05[Seed_49]\AdamW_wd_1e-05[Seed_49].log",   # seed49
        #     r"D:\Work\Python\_MSDT\saved_results\AdamW_wd_1e-05[Seed_50]\AdamW_wd_1e-05[Seed_50].log",   # seed50
        # ],
        # "AdamW + WD=5e-5": [
        #     r"D:\Work\Python\_MSDT\saved_results\AdamW_wd_5e-05[Seed_48]\AdamW_wd_5e-05[Seed_48].log",   # seed48
        #     r"D:\Work\Python\_MSDT\saved_results\AdamW_wd_5e-05[Seed_49]\AdamW_wd_5e-05[Seed_49].log",   # seed49
        #     r"D:\Work\Python\_MSDT\saved_results\AdamW_wd_5e-05[Seed_50]\AdamW_wd_5e-05[Seed_50].log",   # seed50
        # ],
        # "AdamW + WD=1e-4": [
        #     r"D:\Work\Python\_MSDT\saved_results\AdamW_wd_1e-04[Seed_48]\AdamW_wd_1e-04[Seed_48].log",   # seed48
        #     r"D:\Work\Python\_MSDT\saved_results\AdamW_wd_1e-04[Seed_49]\AdamW_wd_1e-04[Seed_49].log",   # seed49
        #     r"D:\Work\Python\_MSDT\saved_results\AdamW_wd_1e-04[Seed_50]\AdamW_wd_1e-04[Seed_50].log",   # seed50
        # ],
    }
    
    # # 绘制并保存（推荐用于论文）
    # draw_crave(
    #     experiments,
    #     smooth_method='ewma',
    #     alpha=0.2,
    #     save_path="loss_curves_smoothed.pdf"  # 或 .svg
    # )
    
    # 如果只想显示不保存
    draw_crave(experiments, smooth_method='ewma', train_a=0.3, val_a=0.1, save_path=None)