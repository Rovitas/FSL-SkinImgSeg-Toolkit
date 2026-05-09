import os
from PIL import Image
import torch
from torch.utils.data import Dataset


class MYDataset(Dataset):
    def __init__(self, base_dir, pt_file=None, use_roi=False, transform=None):
        #   获取图片文件名列表
        self.base_dir = base_dir
        if pt_file is not None:
            self.image_dir = pt_file
            self.is_save_as_pt = False
        else:
            self.image_dir = base_dir
            self.is_save_as_pt = True
        self.use_roi = use_roi
        self.transform = transform

        #   检查文件路径是否存在
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        
        self.sample_ids = [f for f in os.listdir(self.image_dir)]

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        self.sample_id = sample_id
        print(f"{sample_id} {os.path.basename(self.base_dir) }")
        #   构造路径
        img_path = os.path.join(self.image_dir, sample_id, f"{sample_id}_Dermoscopic_Image", f"{sample_id}.bmp")
        lesion_path = os.path.join(self.image_dir, sample_id, f"{sample_id}_lesion", f"{sample_id}_lesion.bmp")
        roi_folder = os.path.join(self.image_dir, sample_id, f"{sample_id}_roi")

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        # 标签路径：根据 use_roi 决定用 lesion 还是 ROI
        if self.use_roi:
            label_path = os.path.join(roi_folder, f"{sample_id}_R1_Label.bmp")
            if not os.path.exists(label_path):
                # 尝试 R2 作为备选
                label_path = os.path.join(roi_folder, f"{sample_id}_R2_Label.bmp")
                if not os.path.exists(label_path):
                    raise FileNotFoundError(f"Neither R1 nor R2 ROI mask found for {sample_id}")
        else:
            label_path = lesion_path
            if not os.path.exists(label_path):
                raise FileNotFoundError(f"Lesion mask not found: {label_path}")
            
        image = Image.open(img_path).convert("RGB")  # 原始图像
        label = Image.open(label_path).convert("L")  # 标签图像（灰度）

        # 转换为Tensor并应用变换
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image, label
    
class MYDataset_From_Pt(Dataset):
    def __init__(self, pt_file, transform=None):
        data = torch.load(pt_file)
        self.base_dir = pt_file
        self.images = data["images"]      # [N, 3, H, W], uint8
        self.masks = data["masks"]        # [N, H, W], uint8
        self.sample_ids = data["ids"]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        print(f"{sample_id} {os.path.basename(self.base_dir) }")
        img = self.images[idx].float() / 255.0   # [3, H, W]
        mask = self.masks[idx].float() / 255.0   # [1, H, W] ← 已带 channel 维度

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)  # Resize 会保持 [1, H, W]

        return img, mask


# 测试代码
if __name__ == '__main__':
    # 原始图像和标签路径 
    # image_dir = r"d:\Work\Python\_MSDT" # os.path.abspath(__file__).split('code')[0]
    # dataset = MYDataset(image_dir)
    
 
    # 从pt文件读取dataset
    pt_dir = r'd:\Work\Python\_MSDT\dataset\val.pt'
    dataset = MYDataset_From_Pt(pt_dir)

    # 测试数据集
    print(dataset.sample_ids)
    print("length:" + str(len(dataset)))
    for i in range(0, len(dataset)):
        print(dataset[i])
