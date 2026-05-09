import os
import re
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# ================= 1. 数据解析模块 =================
def parse_evaluation_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到文件: {file_path}")
        
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    results = {}
    metric_patterns = {
        'Val Loss': r'Val Loss:\s*([0-9.]+)',
        'Dice (DSC)': r'Dice \(DSC\):\s*([0-9.]+)',
        'IoU': r'IoU:\s*([0-9.]+)',
        'Accuracy': r'Accuracy:\s*([0-9.]+)',
        'Recall': r'Recall:\s*([0-9.]+)',
        'Precision': r'Precision:\s*([0-9.]+)',
        'PR-AUC': r'PR-AUC:\s*([0-9.N/A]+)',
        'Average Surface Distance (ASD)': r'Average Surface Distance \(ASD\):\s*([0-9.]+)',
        'Boundary F1 Score': r'Boundary F1 Score:\s*([0-9.]+)'
    }
    
    for exp in content.split('------------------------------------------------------------'):
        exp_name_match = re.search(r'Experiment:\s*(.+)', exp)
        if not exp_name_match: continue
        
        exp_name = exp_name_match.group(1).strip()
        metrics = {}
        for metric, pattern in metric_patterns.items():
            match = re.search(pattern, exp)
            if match:
                val = match.group(1)
                metrics[metric] = np.nan if val == 'N/A' else float(val)
        results[exp_name] = metrics
    return results

# ================= 2. 数据计算模块 =================
def format_wd(wd_value):
    if wd_value == 0: return "0"
    return f"{wd_value:.0e}".replace("e-0", "e-")

def calculate_loss_averages(results_dict, config_dict):
    averaged = {}
    for display_name, target_keyword in config_dict.items():
        matched_metrics = [metrics for exp_name, metrics in results_dict.items() if target_keyword in exp_name]
        if matched_metrics:
            avg_metrics = {}
            for metric_name in matched_metrics[0].keys():
                vals = [m[metric_name] for m in matched_metrics if not np.isnan(m[metric_name])]
                avg_metrics[metric_name] = np.mean(vals) if vals else np.nan
            averaged[display_name] = avg_metrics
    return averaged

def calculate_optimizer_averages(results_dict):
    grouped = {'Adam': {}, 'AdamW': {}}
    for exp_name, metrics in results_dict.items():
        opt = 'AdamW' if 'AdamW' in exp_name else 'Adam' if 'Adam' in exp_name else None
        if not opt: continue
        wd_match = re.search(r'wd_([0-9e.-]+)', exp_name)
        if wd_match:
            wd = float(wd_match.group(1))
            grouped[opt].setdefault(wd, []).append(metrics)
            
    averaged = {'Adam': {}, 'AdamW': {}}
    for opt, wd_groups in grouped.items():
        for wd, metrics_list in wd_groups.items():
            avg_metrics = {}
            for metric_name in metrics_list[0].keys():
                vals = [m[metric_name] for m in metrics_list if not np.isnan(m[metric_name])]
                avg_metrics[metric_name] = np.mean(vals) if vals else np.nan
            averaged[opt][wd] = avg_metrics
    return averaged

# ================= 3. 核心绘图引擎 (完美复刻参考图排版) =================
def create_reference_style_chart(averaged_data, title="Metrics Comparison", save_path=None):
    models = list(averaged_data.keys())
    if not models: return

    # 为了符合从上到下的阅读习惯，将模型列表翻转 (因为 barh 是从底往上画的)
    models_reversed = list(reversed(models))
    
    # 按照 3x3 矩阵规划 9 个指标
    metrics_to_plot = [
        'Dice (DSC)', 'IoU', 'Accuracy', 
        'PR-AUC', 'Recall', 'Precision', 
        'Boundary F1 Score', 'Average Surface Distance (ASD)', 'Val Loss'
    ]

    # 创建 3行3列 画布
    fig, axes = plt.subplots(3, 3, figsize=(22, 18))
    axes = axes.flatten()

    # 复刻参考图的高级渐变色 (深紫 -> 黄绿)
    cmap = plt.get_cmap('viridis')
    # 生成渐变色数组，0.1 到 0.9 避开极亮或极暗
    colors = cmap(np.linspace(0.1, 0.9, len(models_reversed)))

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        # 提取数据并翻转以匹配 Y 轴顺序
        vals = [averaged_data[m].get(metric, np.nan) for m in models_reversed]
        
        # 绘制粗壮的水平条形图
        bars = ax.barh(models_reversed, vals, height=0.75, color=colors, alpha=0.9)
        
        # 子图美化：标题居中
        ax.set_title(metric, fontsize=16, fontweight='bold', pad=15)
        
        # 开启X轴的垂直网格线 (复刻参考图)
        ax.grid(axis='x', linestyle='--', alpha=0.6, color='gray')
        ax.set_axisbelow(True) # 让网格线沉入柱子下方
        
        # Y轴标签 (模型名字)
        ax.set_yticks(np.arange(len(models_reversed)))
        ax.set_yticklabels(models_reversed, fontsize=12, fontweight='bold')
        ax.tick_params(axis='y', which='major', pad=8)
        
        # 去掉顶部和右侧的边框线，让画面更干净
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # 动态设置 X 轴范围
        if metric in ['Average Surface Distance (ASD)', 'Val Loss']:
            max_val = max([v for v in vals if not np.isnan(v)] or [1.0])
            ax.set_xlim(0, max_val * 1.2)  # 给右侧留20%空间
        else:
            ax.set_xlim(0, 1.05)
            
        # 【精简的数值标签】
        # 参考图没有数字。如果你怕评审老师挑刺，我帮你加了一层极简的、紧贴在柱子屁股后面的灰色小数字。
        # 如果你完全不要数字，把下面这段 for 循环注释掉即可。
        for bar in bars:
            w = bar.get_width()
            if not np.isnan(w) and w > 0:
                fmt = '{:.2f}' if metric in ['Average Surface Distance (ASD)'] else ('{:.4f}' if metric == 'Val Loss' else '{:.3f}')
                ax.text(w + (ax.get_xlim()[1] * 0.02), bar.get_y() + bar.get_height()/2., 
                        fmt.format(w), ha='left', va='center', fontsize=10, color='#333333')

    # 添加大标题
    fig.suptitle(title, fontsize=24, fontweight='bold', y=0.96)
    
    # 调整布局间距
    plt.tight_layout(pad=4.0, w_pad=3.0, h_pad=3.5)
    fig.subplots_adjust(top=0.90) # 给大标题留出空间
    
    if save_path: 
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path + ".png", dpi=300, bbox_inches='tight')
        print(f"✅ 图表已完美保存至: {save_path}.png")
    plt.show()

# ================= 4. 控制入口 =================
if __name__ == "__main__":
    BASE_PATH = r"D:\Work\Python\_MSDT\calcuMetrics"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    IMG_PATH = os.path.join(BASE_PATH, 'image', f'metrics_reference_style_{timestamp}')
    
    # [模式切换] : 'loss' 或 'optimizer'
    PLOT_MODE = 'optimizer' 
    
    if PLOT_MODE == 'loss':
        EXPERIMENTS_TO_PLOT = {
            'Baseline': "Adam_wd_1e-05",
            'TverskyHD55': "TverskyHD(a0.3_b0.7w0.5_0.5)",
            'TverskyHD82': "TverskyHD(a0.3_b0.7w0.8_0.2)",
            'TverskyHD91': "TverskyHD(a0.3_b0.7w0.9_0.1)"
        }
        
        file_path = os.path.join(BASE_PATH, "Losses_Comparsion.txt")
        all_results = parse_evaluation_file(file_path)
        averaged_data = calculate_loss_averages(all_results, EXPERIMENTS_TO_PLOT)
        
        create_reference_style_chart(averaged_data, title="Loss Functions Evaluation", save_path=IMG_PATH)
            
    elif PLOT_MODE == 'optimizer':
        adam_res = parse_evaluation_file(os.path.join(BASE_PATH, "Adam.txt"))
        adamw_res = parse_evaluation_file(os.path.join(BASE_PATH, "AdamW.txt"))
        all_results = {**adam_res, **adamw_res}
        
        raw_opt_data = calculate_optimizer_averages(all_results)
        
        flattened_opt_data = {}
        wd_values_sorted = sorted(set(list(raw_opt_data['Adam'].keys()) + list(raw_opt_data['AdamW'].keys())))
        
        # 按照 Adam -> AdamW 的顺序组织数据
        for wd in wd_values_sorted:
            formatted_wd = format_wd(wd)
            if wd in raw_opt_data['Adam']:
                flattened_opt_data[f"Adam (WD={formatted_wd})"] = raw_opt_data['Adam'][wd]
        for wd in wd_values_sorted:
            formatted_wd = format_wd(wd)
            if wd in raw_opt_data['AdamW']:
                flattened_opt_data[f"AdamW (WD={formatted_wd})"] = raw_opt_data['AdamW'][wd]
                
        create_reference_style_chart(flattened_opt_data, title="Optimizer Performance Analysis", save_path=IMG_PATH)