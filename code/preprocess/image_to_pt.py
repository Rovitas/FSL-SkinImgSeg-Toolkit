import os
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms

code_root = Path(__file__).resolve().parents[1]
sys.path.append(str(code_root))

from train.my_dataset import MYDataset


def preprocess_and_save_ph2(root_dir, save_name, category, use_roi=False, target_size=(256, 256), do_aug=False):
    image_dir = os.path.join(root_dir, "images_split", category)
    output_root = os.path.join(root_dir, "pt_dataset")
    all_images = []
    all_masks = []
    all_ids = []
    save_path = os.path.join(output_root, save_name)

    resize_img = transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BILINEAR)
    resize_mask = transforms.Resize(target_size, interpolation=transforms.InterpolationMode.NEAREST)

    os.makedirs(output_root, exist_ok=True)

    dataset = MYDataset(base_dir=image_dir, use_roi=use_roi, transform=None)

    for i in range(len(dataset)):
        img, mask = dataset[i]
        item_id = dataset.sample_ids[i]

        img_resized = resize_img(img)
        mask_resized = resize_mask(mask)

        img_tensor = torch.from_numpy(np.array(img_resized)).permute(2, 0, 1)  # [3, H, W], uint8
        mask_tensor = torch.from_numpy(np.array(mask_resized)).unsqueeze(0)    # [1, H, W], uint8

        all_images.append(img_tensor)
        all_masks.append(mask_tensor)
        all_ids.append(item_id)

        if do_aug:
            # 水平翻转
            all_images.append(TF.hflip(img_tensor))
            all_masks.append(TF.hflip(mask_tensor))
            all_ids.append(item_id + "_H")
            
            # 垂直翻转
            all_images.append(TF.vflip(img_tensor))
            all_masks.append(TF.vflip(mask_tensor))
            all_ids.append(item_id + "_V")

            # 对角翻转（水平+垂直）
            all_images.append(TF.vflip(TF.hflip(img_tensor)))
            all_masks.append(TF.vflip(TF.hflip(mask_tensor)))
            all_ids.append(item_id + "_HV")

    images_stacked = torch.stack(all_images, dim=0)   # [N, 3, H, W]
    masks_stacked = torch.stack(all_masks, dim=0)     # [N, H, W]

    torch.save({
        "images": images_stacked,
        "masks": masks_stacked,
        "ids": all_ids
    }, save_path)

    print(f"## {len(all_ids)} samples have already saved to {save_path}")


if __name__ == '__main__':
    current_path = Path(__file__).resolve()
    root = current_path.parents[2]

    preprocess_and_save_ph2(root_dir=root, save_name="train.pt", category="train", use_roi=False, do_aug=False)
    # preprocess_and_save_ph2(root_dir=root, save_name="val.pt", category="val", do_aug=False)
    
    # 生成一个包含离线增强的扩充训练集
    # preprocess_and_save_ph2(root_dir=root, save_name="train_aug.pt", category="train", do_aug=True)