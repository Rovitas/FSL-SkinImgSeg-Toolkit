import os
import re
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# ================= 1. 数据解析模块 =================
def parse_evaluation_file(file_path):
    """解析评估结果txt文件，提取实验数据"""
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
def calculate_loss_averages(results_dict, config_dict):
    """提取损失函数对比均值"""
    averaged = {}
    for display_name, target_keyword in config_dict.items():
        matched_metrics = []
        for exp_name, metrics in results_dict.items():
            if target_keyword in exp_name:
                matched_metrics.append(metrics)
                
        if matched_metrics:
            avg_metrics = {}
            for metric_name in matched_metrics[0].keys():
                vals = [m[metric_name] for m in matched_metrics if not np.isnan(m[metric_name])]
                avg_metrics[metric_name] = np.mean(vals) if vals else np.nan
            averaged[display_name] = avg_metrics
    return averaged

def calculate_optimizer_averages(results_dict):
    """提取优化器对比均值"""
    grouped = {'Adam': {}, 'AdamW': {}}
    for exp_name, metrics in results_dict.items():
        opt = 'AdamW' if 'AdamW' in exp_name else 'Adam' if 'Adam' in exp_name else None
        if not opt: continue
        wd_match = re.search(r'wd_(\d+e?-?\d+)', exp_name)
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

# ================= 3. 核心绘图引擎 (分组并排版) =================
def create_combined_metrics_chart(averaged_data, title="Metrics Comparison", save_path=None):
    """
    高度集成的混合分组图表引擎：
    上半部分：7个主指标（百分比类型）的分组柱状图
    下半部分：Val Loss 和 ASD（绝对数值类型）的对比柱状图
    """
    models = list(averaged_data.keys())
    if not models:
        print("⚠️ 警告: 没有匹配到任何数据，无法作图。")
        return

    # 指标分类：百分比类型 vs 绝对值类型
    metrics_main = ['Dice (DSC)', 'IoU', 'Accuracy', 'PR-AUC', 'Recall', 'Precision', 'Boundary F1 Score']
    metrics_loss = 'Val Loss'
    metrics_asd = 'Average Surface Distance (ASD)'

    # 构建布局：上半幅占1.5倍高度跨满屏，下半幅2个子图各占一半
    fig = plt.figure(figsize=(20, 13))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.6, 1], hspace=0.4, wspace=0.15)
    
    ax_main = fig.add_subplot(gs[0, :])
    ax_loss = fig.add_subplot(gs[1, 0])
    ax_asd = fig.add_subplot(gs[1, 1])

    # 颜色库
    colors = ['#ff7f00', '#377eb8', '#4daf4a', '#984ea3', '#e7298a', '#a65628', '#f781bf', '#999999']
    
    n_bars = len(models)
    
    # -----------------------
    # 绘制上半部分 (主指标集合)
    # -----------------------
    n_groups = len(metrics_main)
    bar_width = 0.8 / n_bars
    x_main = np.arange(n_groups)

    for i, model in enumerate(models):
        vals = [averaged_data[model].get(m, np.nan) for m in metrics_main]
        x_pos = x_main - 0.4 + (i + 0.5) * bar_width
        
        bars = ax_main.bar(x_pos, vals, width=bar_width, label=model, 
                           color=colors[i % len(colors)], edgecolor='white', linewidth=0.5)
        
        # 智能标签逻辑：如果模型数>=5，标签垂直旋转以防重叠
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h) and h > 0:
                rot = 90 if n_bars >= 5 else 0
                offset = 0.02 if rot == 0 else 0.04
                ax_main.text(bar.get_x() + bar.get_width()/2., h + offset, f'{h:.3f}',
                             ha='center', va='bottom', fontsize=11, fontweight='bold', rotation=rot)

    ax_main.set_xticks(x_main)
    ax_main.set_xticklabels(metrics_main, fontsize=14, fontweight='bold')
    ax_main.set_ylim(0, 1.25)  # 给顶部图例留出空间
    ax_main.grid(True, linestyle='--', alpha=0.6, axis='y')
    ax_main.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=n_bars, fontsize=14, frameon=True)
    ax_main.set_title(f"{title} (Percentage Metrics)", pad=60, fontsize=20, fontweight='bold')

    # -----------------------
    # 绘制左下角 (Val Loss)
    # -----------------------
    x_other = np.arange(n_bars)
    vals_loss = [averaged_data[m].get(metrics_loss, np.nan) for m in models]
    
    bars_loss = ax_loss.bar(x_other, vals_loss, color=[colors[i%len(colors)] for i in range(n_bars)], width=0.55)
    ax_loss.set_xticks(x_other)
    ax_loss.set_xticklabels(models, fontsize=12, rotation=15, ha='right')
    ax_loss.set_title(metrics_loss, fontsize=16, fontweight='bold')
    ax_loss.grid(True, linestyle='--', alpha=0.6, axis='y')
    
    max_loss = max([v for v in vals_loss if not np.isnan(v)] or [1.0])
    ax_loss.set_ylim(0, max_loss * 1.25)
    for bar in bars_loss:
        h = bar.get_height()
        if not np.isnan(h) and h > 0:
            ax_loss.text(bar.get_x() + bar.get_width()/2., h + max_loss*0.02, f'{h:.4f}',
                         ha='center', va='bottom', fontsize=12, fontweight='bold')

    # -----------------------
    # 绘制右下角 (ASD)
    # -----------------------
    vals_asd = [averaged_data[m].get(metrics_asd, np.nan) for m in models]
    
    bars_asd = ax_asd.bar(x_other, vals_asd, color=[colors[i%len(colors)] for i in range(n_bars)], width=0.55)
    ax_asd.set_xticks(x_other)
    ax_asd.set_xticklabels(models, fontsize=12, rotation=15, ha='right')
    ax_asd.set_title(metrics_asd, fontsize=16, fontweight='bold')
    ax_asd.grid(True, linestyle='--', alpha=0.6, axis='y')
    
    max_asd = max([v for v in vals_asd if not np.isnan(v)] or [10.0])
    ax_asd.set_ylim(0, max_asd * 1.25)
    for bar in bars_asd:
        h = bar.get_height()
        if not np.isnan(h) and h > 0:
            ax_asd.text(bar.get_x() + bar.get_width()/2., h + max_asd*0.02, f'{h:.2f}',
                         ha='center', va='bottom', fontsize=12, fontweight='bold')

    # -----------------------
    # 保存与显示
    # -----------------------
    if save_path: 
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path + ".png", dpi=300, bbox_inches='tight')
        print(f"✅ 图表已保存至: {save_path}.png")
    plt.show()

# ================= 4. 控制入口 =================
if __name__ == "__main__":
    # 统一路径配置
    BASE_PATH = r"D:\Work\Python\_MSDT\calcuMetrics"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    IMG_PATH = os.path.join(BASE_PATH, 'image', f'comparison_metrics_{timestamp}')
    
    # [模式切换] : 'loss' 或 'optimizer'
    PLOT_MODE = 'optimizer' 
    
    if PLOT_MODE == 'loss':
        EXPERIMENTS_TO_PLOT = {
            'Baseline': "Adam_wd_1e-05",
            # 'Combo': "TverskyHD(a0.3_b0.7w0.8_0.2)",
            'TverskyHD55': "TverskyHD(a0.3_b0.7w0.5_0.5)",
            'TverskyHD82': "TverskyHD(a0.3_b0.7w0.8_0.2)",
            'TverskyHD91': "TverskyHD(a0.3_b0.7w0.9_0.1)"
        }
        
        file_path = os.path.join(BASE_PATH, "Losses_Comparsion.txt")
        all_results = parse_evaluation_file(file_path)
        averaged_data = calculate_loss_averages(all_results, EXPERIMENTS_TO_PLOT)
        
        # 统一绘图引擎调用
        create_combined_metrics_chart(
            averaged_data, 
            title="Loss Functions Evaluation",
            save_path=IMG_PATH
        )
            
    elif PLOT_MODE == 'optimizer':
        adam_res = parse_evaluation_file(os.path.join(BASE_PATH, "Adam.txt"))
        adamw_res = parse_evaluation_file(os.path.join(BASE_PATH, "AdamW.txt"))
        all_results = {**adam_res, **adamw_res}
        
        raw_opt_data = calculate_optimizer_averages(all_results)
        
        # 将嵌套的字典扁平化，生成例如 "Adam (WD=1e-5)" 的名字，喂给同一个统一引擎
        flattened_opt_data = {}
        wd_values_sorted = sorted(set(list(raw_opt_data['Adam'].keys()) + list(raw_opt_data['AdamW'].keys())))
        for wd in wd_values_sorted:
            wd_str = f'{wd:.0e}' if wd not in [0, 1e-5, 5e-5, 1e-4] else str(wd)
            if wd in raw_opt_data['Adam']:
                flattened_opt_data[f"Adam (WD={wd_str})"] = raw_opt_data['Adam'][wd]
            if wd in raw_opt_data['AdamW']:
                flattened_opt_data[f"AdamW (WD={wd_str})"] = raw_opt_data['AdamW'][wd]
                
        # 统一绘图引擎调用
        create_combined_metrics_chart(
            flattened_opt_data,
            title="Optimizer Performance Analysis",
            save_path=IMG_PATH
        )