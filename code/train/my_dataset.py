import os
import re
from PIL import Image
import torch
from torch.utils.data import Dataset


class MYDataset(Dataset):
    def __init__(self, base_dir, use_roi=False, transform=None):
        self.base_dir = base_dir
        self.use_roi = use_roi
        self.transform = transform

        if not os.path.exists(self.base_dir):
            raise FileNotFoundError(f"Image directory not found: {self.base_dir}")
        
        self.sample_ids = sorted([
            f for f in os.listdir(self.base_dir) 
            if f.startswith("IMD") and os.path.isdir(os.path.join(self.base_dir, f))
        ])

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        
        img_path = os.path.join(self.base_dir, sample_id, f"{sample_id}_Dermoscopic_Image", f"{sample_id}.bmp")
        lesion_path = os.path.join(self.base_dir, sample_id, f"{sample_id}_lesion", f"{sample_id}_lesion.bmp")
        roi_folder = os.path.join(self.base_dir, sample_id, f"{sample_id}_roi")

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        if self.use_roi:
            label_path = None
            if os.path.exists(roi_folder):
                roi_files = os.listdir(roi_folder)
                pattern = re.compile(rf"^{sample_id}_R\d+_Label\d+\.bmp$")
                
                matched_files = sorted([f for f in roi_files if pattern.match(f)])
                
                if matched_files:
                    label_path = os.path.join(roi_folder, matched_files[0])
            
            if label_path is None or not os.path.exists(label_path):
                raise FileNotFoundError(f"No valid ROI mask found for {sample_id} matching pattern '{sample_id}_R[X]_Label[Y].bmp'")
        else:
            label_path = lesion_path
            if not os.path.exists(label_path):
                raise FileNotFoundError(f"Lesion mask not found: {label_path}")
            
        image = Image.open(img_path).convert("RGB")
        label = Image.open(label_path).convert("L")

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image, label
    

class MYDataset_From_Pt(Dataset):
    def __init__(self, pt_file, transform=None):
        if not os.path.exists(pt_file):
            raise FileNotFoundError(f"PT file not found: {pt_file}")
            
        data = torch.load(pt_file)
        self.images = data["images"]      # [N, 3, H, W], uint8
        self.masks = data["masks"]        # [N, H, W], uint8
        self.sample_ids = data["ids"]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].float() / 255.0   # [3, H, W]
        mask = self.masks[idx].float() / 255.0   # [1, H, W]

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)  

        return img, mask


if __name__ == '__main__':
    from pathlib import Path
    PROJECT_ROOT = Path( __file__).resolve().parents[2]

    # Pt file test
    pt_dir = os.path.join(PROJECT_ROOT, "pt_dataset", "val.pt")
    if os.path.exists(pt_dir):
        print("\n## Loading PT Dataset...")
        dataset_pt = MYDataset_From_Pt(pt_dir)
        print(f"## Loaded {len(dataset_pt)} samples from PT.")
        print(f"## Sample IDs (first 5): {dataset_pt.sample_ids[:5]}")

    # Raw images test    
    img_dir = os.path.join(PROJECT_ROOT, "images_split", "train")
    if os.path.exists(img_dir):
        print("\n## Loading Raw Image Dataset...")
        dataset_raw = MYDataset(img_dir, use_roi=False)
        print(f"## Found {len(dataset_raw)} raw samples.\n")