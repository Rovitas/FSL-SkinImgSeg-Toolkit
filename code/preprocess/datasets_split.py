import os
import random
import shutil
from pathlib import Path


def split_ph2_dataset(root_path, output_root=None, seed=42, ratio = {'train':0.6, 'test':0.2, 'val':0.2}):
    random.seed(seed)
    
    if output_root is None:
        output_root = os.path.join(root_path, "images_split")
    os.makedirs(output_root, exist_ok=True)

    src_images_dir = os.path.join(root_path, "images")
    if not os.path.exists(src_images_dir):
        os.makedirs(src_images_dir, exist_ok=True)
        raise FileNotFoundError(f"## Path not found: {src_images_dir}")
    
    all_samples = []
    for item in sorted(os.listdir(src_images_dir)):
        item_path = os.path.join(src_images_dir, item)
        if os.path.isdir(item_path) and item.startswith("IMD"):
            img_path = os.path.join(item_path, f"{item}_Dermoscopic_Image", f"{item}.bmp")
            mask_path = os.path.join(item_path, f"{item}_lesion", f"{item}_lesion.bmp")
            if os.path.exists(img_path) and os.path.exists(mask_path):
                all_samples.append(item)
    
    if not all_samples:
        raise ValueError("## No valid PH2 samples found!")
    
    print(f"## Found {len(all_samples)} valid samples.")
    
    random.shuffle(all_samples)
    
    n_total = len(all_samples)
    n_train = int(ratio.get('train') * n_total)
    n_test = int(ratio.get('test') * n_total)
    n_val = n_total - n_train - n_test
    
    train_samples = all_samples[:n_train]
    test_samples = all_samples[n_train:n_train + n_test]
    val_samples = all_samples[n_train + n_test:]
    
    print(f"## Samples amount: Train: {len(train_samples)}, Test: {len(test_samples)}, Val: {len(val_samples)}")
    
    def make_split_dir(split_name):
        dst_dir = os.path.join(output_root, split_name)
        os.makedirs(dst_dir, exist_ok=True)
        return dst_dir
    
    train_dst = make_split_dir("train")
    test_dst = make_split_dir("test")
    val_dst = make_split_dir("val")
    
    def copy_samples(samples, dst_dir):
        for sample_id in samples:
            src = os.path.join(src_images_dir, sample_id)
            dst = os.path.join(dst_dir, sample_id)
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            # print(f"Copied {sample_id} to {dst_dir}")
    
    copy_samples(train_samples, train_dst)
    copy_samples(test_samples, test_dst)
    copy_samples(val_samples, val_dst)
    
    print("## Random spliting of Datasets has been finished!")
    print(f"## Output path: {output_root}")


if __name__ == '__main__':
    current_path = Path(__file__).resolve()
    root = current_path.parents[2]

    split_ratio = {'train':0.6, 'test':0.2, 'val':0.2}

    split_ph2_dataset(root_path=root, ratio=split_ratio)