import os
import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF  # 新增：用于翻转操作
from tqdm import tqdm
import numpy as np

def preprocess_and_save_ph2(root_dir, save_name, category, use_roi=False, target_size=(256, 256), do_aug=False):
    image_dir = os.path.join(root_dir, "images_split", category)
    all_images = []
    all_masks = []
    all_ids = []
    save_path = os.path.join(root_dir, "dataset", save_name)

    resize_img = transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BILINEAR)
    resize_mask = transforms.Resize(target_size, interpolation=transforms.InterpolationMode.NEAREST)

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    for item in sorted(os.listdir(image_dir)):
        if not item.startswith("IMD") or not os.path.isdir(os.path.join(image_dir, item)):
            continue

        derm_folder = os.path.join(image_dir, item, f"{item}_Dermoscopic_Image")
        lesion_folder = os.path.join(image_dir, item, f"{item}_lesion")
        roi_folder = os.path.join(image_dir, item, f"{item}_roi")

        img_path = os.path.join(derm_folder, f"{item}.bmp")
        if not os.path.exists(img_path):
            continue

        if use_roi:
            r1 = os.path.join(roi_folder, f"{item}_R1_Label.bmp")
            r2 = os.path.join(roi_folder, f"{item}_R2_Label.bmp")
            mask_path = r1 if os.path.exists(r1) else (r2 if os.path.exists(r2) else None)
        else:
            mask_path = os.path.join(lesion_folder, f"{item}_lesion.bmp")

        if not mask_path or not os.path.exists(mask_path):
            continue

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        img_resized = resize_img(img)
        mask_resized = resize_mask(mask)

        img_tensor = torch.from_numpy(np.array(img_resized)).permute(2, 0, 1)  # [3, H, W], uint8
        mask_tensor = torch.from_numpy(np.array(mask_resized)).unsqueeze(0)    # [1, H, W], uint8

        # 1. 保存原图
        all_images.append(img_tensor)
        all_masks.append(mask_tensor)
        all_ids.append(item)

        # 2. 如果开启了离线增强，把翻转后的图也塞进列表里
        if do_aug:
            # 水平翻转
            all_images.append(TF.hflip(img_tensor))
            all_masks.append(TF.hflip(mask_tensor))
            all_ids.append(item + "_H")
            
            # 垂直翻转
            all_images.append(TF.vflip(img_tensor))
            all_masks.append(TF.vflip(mask_tensor))
            all_ids.append(item + "_V")

            # 对角翻转（水平+垂直）
            all_images.append(TF.vflip(TF.hflip(img_tensor)))
            all_masks.append(TF.vflip(TF.hflip(mask_tensor)))
            all_ids.append(item + "_HV")

    images_stacked = torch.stack(all_images, dim=0)   # [N, 3, H, W]
    masks_stacked = torch.stack(all_masks, dim=0)     # [N, H, W]

    torch.save({
        "images": images_stacked,
        "masks": masks_stacked,
        "ids": all_ids
    }, save_path)

    print(f"Saved {len(all_ids)} samples to {save_path}")

if __name__ == '__main__':
    # preprocess_and_save_ph2(root_dir=r"d:\Work\Python\_MSDT", save_name="train.pt", category="train", do_aug=False)
    # preprocess_and_save_ph2(root_dir=r"d:\Work\Python\_MSDT", save_name="val.pt", category="val", do_aug=False)
    
    # 生成一个包含离线增强的扩充训练集
    preprocess_and_save_ph2(root_dir=r"d:\Work\Python\_MSDT", save_name="train_aug.pt", category="train", do_aug=True)