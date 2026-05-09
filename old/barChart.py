import re
import matplotlib.pyplot as plt
import numpy as np

def parse_evaluation_file(file_path):
    """
    解析评估结果txt文件，提取实验数据
    
    Args:
        file_path: txt文件路径
        
    Returns:
        dict: {experiment_name: {metric: value}}
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 按实验分割
    experiments = content.split('------------------------------------------------------------')
    
    results = {}
    
    for exp in experiments:
        if 'Experiment:' not in exp:
            continue
            
        # 提取实验名称
        exp_name_match = re.search(r'Experiment:\s*(.+)', exp)
        if not exp_name_match:
            continue
        exp_name = exp_name_match.group(1).strip()
        
        # 提取所有指标
        metrics = {}
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
        
        for metric_name, pattern in metric_patterns.items():
            match = re.search(pattern, exp)
            if match:
                value_str = match.group(1)
                if value_str == 'N/A':
                    metrics[metric_name] = np.nan
                else:
                    metrics[metric_name] = float(value_str)
        
        results[exp_name] = metrics
    
    return results

def extract_wd_and_optimizer(experiment_name):
    """
    从实验名称中提取优化器类型和权重衰减值
    
    Args:
        experiment_name: 实验名称字符串
        
    Returns:
        tuple: (optimizer, wd_value)
    """
    # 匹配优化器类型 (Adam 或 AdamW)
    if 'AdamW' in experiment_name:
        optimizer = 'AdamW'
        # 提取WD值
        wd_match = re.search(r'wd_(\d+e?-?\d+)', experiment_name)
        if wd_match:
            wd_str = wd_match.group(1)
            # 处理科学计数法
            if 'e' in wd_str:
                wd_value = float(wd_str)
            else:
                wd_value = float(wd_str)
        else:
            wd_value = 0.0
    elif 'Adam' in experiment_name:
        optimizer = 'Adam'
        # 提取WD值
        wd_match = re.search(r'wd_(\d+e?-?\d+)', experiment_name)
        if wd_match:
            wd_str = wd_match.group(1)
            if 'e' in wd_str:
                wd_value = float(wd_str)
            else:
                wd_value = float(wd_str)
        else:
            wd_value = 0.0
    else:
        return None, None
    
    return optimizer, wd_value

def group_experiments_by_optimizer_and_wd(results_dict):
    """
    将实验按优化器和权重衰减值分组
    
    Args:
        results_dict: {experiment_name: metrics}
        
    Returns:
        dict: {'Adam': {wd_value: [metrics_list]}, 'AdamW': {wd_value: [metrics_list]}}
    """
    grouped = {'Adam': {}, 'AdamW': {}}
    
    for exp_name, metrics in results_dict.items():
        optimizer, wd_value = extract_wd_and_optimizer(exp_name)
        if optimizer is None:
            continue
            
        if wd_value not in grouped[optimizer]:
            grouped[optimizer][wd_value] = []
        grouped[optimizer][wd_value].append(metrics)
    
    return grouped

def calculate_average_metrics(grouped_data):
    """
    计算每个优化器-权重衰减组合的平均指标
    
    Args:
        grouped_data: 分组后的数据
        
    Returns:
        dict: {'Adam': {wd_value: avg_metrics}, 'AdamW': {wd_value: avg_metrics}}
    """
    averaged = {'Adam': {}, 'AdamW': {}}
    
    for optimizer in ['Adam', 'AdamW']:
        for wd_value, metrics_list in grouped_data[optimizer].items():
            if not metrics_list:
                continue
                
            # 计算每个指标的平均值
            avg_metrics = {}
            metric_keys = metrics_list[0].keys()
            
            for key in metric_keys:
                values = [m[key] for m in metrics_list if not np.isnan(m[key])]
                if values:
                    avg_metrics[key] = np.mean(values)
                else:
                    avg_metrics[key] = np.nan
                    
            averaged[optimizer][wd_value] = avg_metrics
    
    return averaged

def create_comparison_bar_chart(averaged_data, metric_name='Dice (DSC)', 
                               save_path=None, figsize=(12, 8)):
    """
    创建分组柱状图比较Adam和AdamW在不同权重衰减下的性能
    
    Args:
        averaged_data: 平均指标数据
        metric_name: 要绘制的指标名称
        save_path: 保存路径（可选）
        figsize: 图表大小
    """
    # 获取所有权重衰减值并排序
    all_wd_values = set()
    for optimizer in ['Adam', 'AdamW']:
        all_wd_values.update(averaged_data[optimizer].keys())
    wd_values_sorted = sorted(all_wd_values)
    
    # 准备数据
    adam_values = []
    adamw_values = []
    
    for wd in wd_values_sorted:
        # Adam 数据
        if wd in averaged_data['Adam'] and metric_name in averaged_data['Adam'][wd]:
            adam_values.append(averaged_data['Adam'][wd][metric_name])
        else:
            adam_values.append(0)  # 或者 np.nan，但柱状图不显示nan
            
        # AdamW 数据
        if wd in averaged_data['AdamW'] and metric_name in averaged_data['AdamW'][wd]:
            adamw_values.append(averaged_data['AdamW'][wd][metric_name])
        else:
            adamw_values.append(0)
    
    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)
    
    # 设置柱子位置
    x = np.arange(len(wd_values_sorted))
    width = 0.35
    
    # 绘制柱状图
    bars1 = ax.bar(x - width/2, adam_values, width, label='Adam', color='#ff7f00', alpha=0.8)
    bars2 = ax.bar(x + width/2, adamw_values, width, label='AdamW', color='#377eb8', alpha=0.8)
    
    # 设置标签和标题
    ax.set_xlabel('Weight Decay Coefficient', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
    ax.set_title(f'{metric_name} Comparison: Adam vs AdamW with Different Weight Decay', 
                 fontsize=14, fontweight='bold')
    
    # 设置x轴刻度
    wd_labels = []
    for wd in wd_values_sorted:
        if wd == 0:
            wd_labels.append('0')
        elif wd == 1e-5:
            wd_labels.append('1e-5')
        elif wd == 5e-5:
            wd_labels.append('5e-5')
        elif wd == 1e-4:
            wd_labels.append('1e-4')
        else:
            wd_labels.append(f'{wd:.0e}')
    
    ax.set_xticks(x)
    ax.set_xticklabels(wd_labels)
    
    # 添加数值标签
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height) and height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    
    # 网格和图例
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')
    ax.legend(fontsize=11, loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Chart saved to: {save_path}")
    
    plt.show()

def extract_loss_function(experiment_name):
    """
    从实验名称中提取损失函数类型
    假设实验名称格式如: "Loss_BCE", "Loss_Dice", "Loss_Focal" 等
    """
    # 定义支持的损失函数类型
    loss_types = ['BCE', 'Dice', 'Focal', 'Tversky', 'Combo']
    
    LOSS_FUNCTIONS = {
        'BCE': "AdamW_wd_1e-05",
        'Dice': "DiceLoss",
        'Focal': "FocalLoss(a0.25_g2)",
        'Tversky': "TverskyLoss(a0.3_b0.7)",
        'Combo': "TverskyHD(a0.3_b0.7w0.8_0.2)"
    }
    
    for loss_type in loss_types:
        if f'Loss_{loss_type}' in experiment_name or f'loss_{loss_type}' in experiment_name:
            return loss_type
    
    # 如果没找到，返回默认值
    return "Unknown"

def group_experiments_by_loss_function(results_dict):
    """
    将实验按损失函数类型分组
    """
    grouped = {}
    
    for exp_name, metrics in results_dict.items():
        loss_func = extract_loss_function(exp_name)
        if loss_func not in grouped:
            grouped[loss_func] = []
        grouped[loss_func].append(metrics)
    
    return grouped

def main():
    """
    主函数：读取文件、处理数据、生成图表
    """
    # 文件路径（请根据实际情况修改）
    adam_file = r"D:\Work\Python\_MSDT\calcuMetrics\Adam.txt"      # Adam的结果文件
    adamw_file = r"D:\Work\Python\_MSDT\calcuMetrics\AdamW.txt"    # AdamW的结果文件
    
    try:
        # 解析两个文件
        print("Parsing Adam results...")
        adam_results = parse_evaluation_file(adam_file)
        print(f"Found {len(adam_results)} Adam experiments")
        
        print("Parsing AdamW results...")
        adamw_results = parse_evaluation_file(adamw_file)
        print(f"Found {len(adamw_results)} AdamW experiments")
        
        # 合并结果
        all_results = {**adam_results, **adamw_results}
        
        # 分组和平均
        grouped_data = group_experiments_by_optimizer_and_wd(all_results)
        averaged_data = calculate_average_metrics(grouped_data)
        
        # 打印平均结果用于验证
        print("\nAverage Results:")
        print("=" * 60)
        for optimizer in ['Adam', 'AdamW']:
            print(f"\n{optimizer}:")
            for wd in sorted(averaged_data[optimizer].keys()):
                dsc = averaged_data[optimizer][wd].get('Dice (DSC)', 'N/A')
                print(f"  WD={wd}: DSC={dsc:.4f}")
        
          # 创建所有指标的子图
        metrics_to_plot = [
            'Dice (DSC)',
            'IoU', 
            'Accuracy',
            'PR-AUC',
            'Val Loss',
            'Recall',
            'Precision',
            'Average Surface Distance (ASD)',
            'Boundary F1 Score'
        ]

        # 计算子图布局
        n_metrics = len(metrics_to_plot)
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 12))
        axes = axes.flatten() if n_metrics > 1 else [axes]

        # 绘制所有子图（不加图例）
        for idx, metric in enumerate(metrics_to_plot):
            # 获取所有权重衰减值并排序
            all_wd_values = set()
            for optimizer in ['Adam', 'AdamW']:
                all_wd_values.update(averaged_data[optimizer].keys())
            wd_values_sorted = sorted(all_wd_values)
            
            # 准备数据
            adam_values = []
            adamw_values = []
            
            for wd in wd_values_sorted:
                if wd in averaged_data['Adam'] and metric in averaged_data['Adam'][wd]:
                    adam_values.append(averaged_data['Adam'][wd][metric])
                else:
                    adam_values.append(0)
                    
                if wd in averaged_data['AdamW'] and metric in averaged_data['AdamW'][wd]:
                    adamw_values.append(averaged_data['AdamW'][wd][metric])
                else:
                    adamw_values.append(0)
            
            # 绘制子图
            x = np.arange(len(wd_values_sorted))
            width = 0.35
            
            bars1 = axes[idx].bar(x - width/2, adam_values, width, color='#ff7f00', alpha=0.8)
            bars2 = axes[idx].bar(x + width/2, adamw_values, width, color='#377eb8', alpha=0.8)
            
            # 添加数值标签
            def add_value_labels(bars, ax, metric_name):
                for bar in bars:
                    height = bar.get_height()
                    if not np.isnan(height) and height > 0:
                        # 获取当前子图中所有有效高度的最大值
                        all_heights = []
                        for b in ax.patches:
                            h = b.get_height()
                            if not np.isnan(h) and h > 0:
                                all_heights.append(h)
                        max_height = max(all_heights) if all_heights else 1
                        
                        # 根据指标类型设置不同的格式和偏移
                        if metric_name == 'Average Surface Distance (ASD)' or metric_name == 'Val Loss':
                            offset = max_height * 0.03
                            text = f'{height:.2f}'
                        else:
                            offset = max_height * 0.02
                            text = f'{height:.3f}'
                        
                        ax.text(bar.get_x() + bar.get_width()/2., height + offset,
                               text, ha='center', va='bottom', fontsize=6, fontweight='bold')
            
            add_value_labels(bars1, axes[idx], metric)
            add_value_labels(bars2, axes[idx], metric)
            
            # 设置标签
            wd_labels = []
            for wd in wd_values_sorted:
                if wd == 0:
                    wd_labels.append('0')
                elif wd == 1e-5:
                    wd_labels.append('1e-5')
                elif wd == 5e-5:
                    wd_labels.append('5e-5')
                elif wd == 1e-4:
                    wd_labels.append('1e-4')
                else:
                    wd_labels.append(f'{wd:.0e}')
            
            axes[idx].set_xticks(x)
            axes[idx].set_xticklabels(wd_labels)
            axes[idx].set_title(metric, fontweight='bold')
            axes[idx].grid(True, linestyle='--', alpha=0.7, axis='y')

        # 隐藏多余的子图
        for idx in range(n_metrics, len(axes)):
            axes[idx].set_visible(False)

        # 创建全局图例（放在图表右侧）
        handles = [plt.Rectangle((0,0),1,1, color='#ff7f00', alpha=0.8), 
                   plt.Rectangle((0,0),1,1, color='#377eb8', alpha=0.8)]
        labels = ['Adam', 'AdamW']
        fig.legend(handles, labels, loc='center right', bbox_to_anchor=(0.98, 0.5), 
                   fontsize=12, frameon=True)

        plt.subplots_adjust(
            left=0.08,      # 左边距
            right=0.85,     # 右边距（给图例留空间）
            bottom=0.15,    # 下边距（给x轴标签留空间）
            top=0.92,       # 上边距
            wspace=0.3,     # 子图水平间距
            hspace=0.4      # 子图垂直间距
        )

        plt.savefig("all_metrics_comparison.png", dpi=300, bbox_inches='tight')
        print("All metrics chart saved to: all_metrics_comparison.png")
        plt.show()
            
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        print("Please make sure the input files exist and update the file paths in the code.")
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    main()