# -*- coding: utf-8 -*-
import os
import re
import csv

def parse_txt_to_csv(txt_file_path, csv_file_path):
    """
    将评估结果的 txt 文件解析并保存为 csv 文件
    """
    if not os.path.exists(txt_file_path):
        print(f"❌ 错误: 找不到文件 {txt_file_path}")
        return

    with open(txt_file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 清理可能由于复制粘贴产生的无关换行或系统标记（如 ）
    # 保证类似 "Average Surface Distance (ASD):" 不会被意外截断
    content = re.sub(r'\\s*', '', content)
    content = content.replace('Average \nSurface Distance', 'Average Surface Distance')

    # 按实验分隔符切分文本块
    experiments = re.split(r'-{50,}', content)
    
    all_results = []
    
    # 定义 CSV 的表头（严格对应要提取的字段）
    fieldnames = [
        'Experiment', 'Val Loss', 'Dice (DSC)', 'IoU', 
        'Accuracy', 'Recall', 'Precision', 'PR-AUC', 
        'Average Surface Distance (ASD)', 'Boundary F1 Score'
    ]

    # 定义正则匹配规则
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

    for exp_block in experiments:
        if 'Experiment:' not in exp_block:
            continue
            
        result_dict = {}
        
        # 1. 提取实验名称
        exp_match = re.search(r'Experiment:\s*(.+)', exp_block)
        if exp_match:
            result_dict['Experiment'] = exp_match.group(1).strip()
        else:
            continue # 如果连实验名字都没有，跳过

        # 2. 提取所有指标
        for key, pattern in metric_patterns.items():
            match = re.search(pattern, exp_block)
            if match:
                result_dict[key] = match.group(1).strip()
            else:
                result_dict[key] = "N/A" # 缺失值处理
                
        all_results.append(result_dict)

    if not all_results:
        print("⚠️ 警告: 未能在文件中解析到任何实验数据。请检查文件格式。")
        return

    # 3. 写入 CSV 文件
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader() # 写入表头
        for row in all_results:
            writer.writerow(row) # 写入数据行

    print(f"✅ 转换成功！共处理 {len(all_results)} 个实验，结果已保存至: {csv_file_path}")

if __name__ == '__main__':
    # === 配置输入和输出路径 ===
    # 替换为你实际的 txt 文件路径
    INPUT_TXT = r"D:\Work\Python\_MSDT\calcuMetrics\Losses_Comparsion.txt" 
    
    # 自动在同级目录下生成同名的 csv 文件
    OUTPUT_CSV = INPUT_TXT.replace('.txt', '.csv')
    
    parse_txt_to_csv(INPUT_TXT, OUTPUT_CSV)