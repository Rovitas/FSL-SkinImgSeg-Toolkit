# -*- coding: utf-8 -*-

import os
import shutil
import random

def split_ph2_dataset(ph2_root, output_root=None, seed=37):
    # 设置随机种子
    random.seed(seed)
    
    # 确定输出目录
    if output_root is None:
        output_root = os.path.join(ph2_root, "image_split")
    
    # 原始 images 目录
    src_images_dir = os.path.join(ph2_root, "imagesO")
    if not os.path.exists(src_images_dir):
        raise FileNotFoundError(f"目录未找到: {src_images_dir}")
    
    # 获取所有有效样本 ID（IMDxxx 文件夹）
    all_samples = []
    for item in sorted(os.listdir(src_images_dir)):
        item_path = os.path.join(src_images_dir, item)
        if os.path.isdir(item_path) and item.startswith("IMD"):
            # 验证关键文件存在（可选，增强健壮性）
            img_path = os.path.join(item_path, f"{item}_Dermoscopic_Image", f"{item}.bmp")
            mask_path = os.path.join(item_path, f"{item}_lesion", f"{item}_lesion.bmp")
            if os.path.exists(img_path) and os.path.exists(mask_path):
                all_samples.append(item)
    
    if not all_samples:
        raise ValueError("No valid PH2 samples found!")
    
    print(f"\nFound {len(all_samples)} valid samples.")
    
    # 打乱顺序
    random.shuffle(all_samples)
    
    # 按 6:2:2 划分
    n_total = len(all_samples)
    n_train = int(0.6 * n_total)
    n_test = int(0.2 * n_total)
    n_val = n_total - n_train - n_test  # 确保总数一致
    
    train_samples = all_samples[:n_train]
    test_samples = all_samples[n_train:n_train + n_test]
    val_samples = all_samples[n_train + n_test:]
    
    print(f"Train: {len(train_samples)}, Test: {len(test_samples)}, Val: {len(val_samples)}")
    
    # 定义目标目录
    def make_split_dir(split_name):
        dst_dir = os.path.join(output_root, "image_split", split_name)
        os.makedirs(dst_dir, exist_ok=True)
        return dst_dir
    
    train_dst = make_split_dir("train")
    test_dst = make_split_dir("test")
    val_dst = make_split_dir("val")
    
    # 复制函数
    def copy_samples(samples, dst_dir):
        for sample_id in samples:
            src = os.path.join(src_images_dir, sample_id)
            dst = os.path.join(dst_dir, sample_id)
            if os.path.exists(dst):
                shutil.rmtree(dst)  # 避免重复运行报错
            shutil.copytree(src, dst)
            # print(f"Copied {sample_id} to {dst_dir}")
    
    # 执行复制
    copy_samples(train_samples, train_dst)
    copy_samples(test_samples, test_dst)
    copy_samples(val_samples, val_dst)
    
    print("\n✅ 数据集随机分配完成!")
    print(f"输出目录: {output_root}")


if __name__ == '__main__':
    image_dir = r"d:\Work\Python\_MSDT"   # PH2 原始数据集路径
    # 执行划分
    split_ph2_dataset(ph2_root=image_dir)