import os
import re
import matplotlib.pyplot as plt
import numpy as np

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

# 需要绘制的所有指标列表（保持顺序）
METRICS_TO_PLOT = [
    'Dice (DSC)', 'IoU', 'Accuracy', 'PR-AUC', 'Val Loss', 
    'Recall', 'Precision', 'Average Surface Distance (ASD)', 'Boundary F1 Score'
]

# ================= 2. 数据计算模块 =================
def calculate_loss_averages(results_dict, config_dict):
    """
    根据用户传入的 config_dict (图例名: 实验匹配词) 去原始数据中提取并取均值。
    彻底解决匹配优先级冲突，按 config_dict 的顺序生成数据。
    """
    averaged = {}
    for display_name, target_keyword in config_dict.items():
        matched_metrics = []
        
        # 遍历解析出的所有实验
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
    """专门为 Adam vs AdamW 准备的计算函数"""
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

# ================= 3. 通用图表组件 =================
def _add_value_labels(bars, ax, metric_name):
    """通用添加数值标签方法"""
    max_h = max([b.get_height() for b in ax.patches if not np.isnan(b.get_height())] or [1])
    offset = max_h * (0.025 if metric_name in ['Average Surface Distance (ASD)', 'Val Loss'] else 0.015)
    fmt = '{:.2f}' if metric_name in ['Average Surface Distance (ASD)', 'Val Loss'] else '{:.3f}'
    
    for bar in bars:
        h = bar.get_height()
        if not np.isnan(h) and h > 0:
            ax.text(bar.get_x() + bar.get_width()/2., h + offset, fmt.format(h),
                    ha='center', va='bottom', fontsize=7, fontweight='normal')

def _format_axis(ax, metric_name, values):
    """通用坐标轴格式化"""
    if metric_name in ['Average Surface Distance (ASD)', 'Val Loss']:
        max_val = max([v for v in values if not np.isnan(v)] or [10])
        ax.set_ylim(0, max_val * 1.15)
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    else:
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.grid(True, linestyle='--', alpha=0.6, axis='y')
    ax.tick_params(axis='both', which='major', labelsize=10)

# ================= 4. 绘图执行模块 =================
def create_spacious_loss_comparison_chart(averaged_data, save_path=None, figsize=(18, 12)):
    # 动态获取图例名称（自动遵循传入字典的顺序）
    losses = list(averaged_data.keys())
    if not losses:
        print("⚠️ 警告: 没有匹配到任何数据，无法作图。")
        return
        
    fig, axes = plt.subplots(3, 3, figsize=figsize)
    # 颜色库 (超出会自动循环)
    colors = ['#ff7f00', '#377eb8', '#4daf4a', '#984ea3', '#e7298a', '#a65628', '#f781bf']
    bar_colors = [colors[i % len(colors)] for i in range(len(losses))]
    
    for idx, metric in enumerate(METRICS_TO_PLOT):
        ax = axes.flatten()[idx]
        vals = [averaged_data[l].get(metric, np.nan) for l in losses]
        x = np.arange(len(losses))
        
        bars = ax.bar(x, vals, color=bar_colors, width=0.6)
        ax.set_title(metric, fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(losses, fontsize=11)
        
        _add_value_labels(bars, ax, metric)
        _format_axis(ax, metric, vals)

    # 隐藏多余的子图
    for idx in range(len(METRICS_TO_PLOT), 9): 
        axes.flatten()[idx].set_visible(False)
    
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=bar_colors[i], label=losses[i]) for i in range(len(losses))]
    fig.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.02, 0.5), title='Loss Functions', fontsize=12, frameon=True)
    plt.tight_layout(pad=3.0, h_pad=2.5, w_pad=2.5)
    fig.subplots_adjust(right=0.85, top=0.92)
    
    if save_path: 
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 图表已保存至: {save_path}")
    plt.show()

def create_all_metrics_optimizer_chart(averaged_data, save_path="all_metrics_comparison.png"):
    wd_values_sorted = sorted(set(list(averaged_data['Adam'].keys()) + list(averaged_data['AdamW'].keys())))
    wd_labels = [f'{wd:.0e}' if wd not in [0, 1e-5, 5e-5, 1e-4] else str(wd) for wd in wd_values_sorted]

    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    
    for idx, metric in enumerate(METRICS_TO_PLOT):
        ax = axes.flatten()[idx]
        adam_vals = [averaged_data['Adam'].get(wd, {}).get(metric, np.nan) for wd in wd_values_sorted]
        adamw_vals = [averaged_data['AdamW'].get(wd, {}).get(metric, np.nan) for wd in wd_values_sorted]
        x = np.arange(len(wd_values_sorted))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, adam_vals, width, color='#ff7f00', alpha=0.8)
        bars2 = ax.bar(x + width/2, adamw_vals, width, color='#377eb8', alpha=0.8)
        
        ax.set_title(metric, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(wd_labels)
        
        _add_value_labels(bars1, ax, metric)
        _add_value_labels(bars2, ax, metric)
        _format_axis(ax, metric, adam_vals + adamw_vals)

    for idx in range(len(METRICS_TO_PLOT), 9): 
        axes.flatten()[idx].set_visible(False)
    
    handles = [plt.Rectangle((0,0),1,1, color='#ff7f00', alpha=0.8), plt.Rectangle((0,0),1,1, color='#377eb8', alpha=0.8)]
    fig.legend(handles, ['Adam', 'AdamW'], loc='center right', bbox_to_anchor=(0.98, 0.5), fontsize=12, frameon=True)
    plt.subplots_adjust(left=0.08, right=0.85, bottom=0.15, top=0.92, wspace=0.3, hspace=0.4)
    
    if save_path: 
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 图表已保存至: {save_path}")
    plt.show()

# ================= 5. 控制入口 =================
if __name__ == "__main__":
    # 统一路径配置
    BASE_PATH = r"D:\Work\Python\_MSDT\calcuMetrics"
    
    # [模式切换] : 'loss' 或 'optimizer'
    PLOT_MODE = 'loss' 
    
    if PLOT_MODE == 'loss':
        # ---------------------------------------------------------
        # 【修改这里即可控制出图！】
        # 键 (Key)   : 最终显示在图表 X 轴和图例上的名字
        # 值 (Value) : 在日志 txt 文件中用于匹配的关键字 (会自动避开随机种子)
        # 字典的顺序就是柱状图从左到右显示的顺序，随便增减不会互相覆盖
        # ---------------------------------------------------------
        EXPERIMENTS_TO_PLOT = {
            'Baseline': "Adam_wd_1e-05",
            # 'Combo': "TverskyHD(a0.3_b0.7w0.8_0.2)", # 如果不想画Combo，注销掉这一行就行
            'TverskyHD55': "TverskyHD(a0.3_b0.7w0.5_0.5)",
            'TverskyHD82': "TverskyHD(a0.3_b0.7w0.8_0.2)",
            'TverskyHD91': "TverskyHD(a0.3_b0.7w0.9_0.1)"
        }
        
        file_path = os.path.join(BASE_PATH, "Losses_Comparsion.txt")
        all_results = parse_evaluation_file(file_path)
        
        # 核心解耦：直接把你的字典传进去捞数据
        averaged_data = calculate_loss_averages(all_results, EXPERIMENTS_TO_PLOT)
        
        create_spacious_loss_comparison_chart(
            averaged_data, 
            save_path="loss_comparison_all_metrics_spacious.png"
        )
            
    elif PLOT_MODE == 'optimizer':
        adam_res = parse_evaluation_file(os.path.join(BASE_PATH, "Adam.txt"))
        adamw_res = parse_evaluation_file(os.path.join(BASE_PATH, "AdamW.txt"))
        all_results = {**adam_res, **adamw_res}
        
        averaged_data = calculate_optimizer_averages(all_results)
        create_all_metrics_optimizer_chart(
            averaged_data,
            save_path="all_metrics_comparison.png"
        )