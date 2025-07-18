import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms():
    """定义温和与强度两种数据增强策略。"""
    gentle_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    strong_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.7),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2, p=0.7),
        A.CoarseDropout(max_holes=8, max_height=16, max_width=16, min_holes=1, min_height=8, min_width=8, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    return gentle_transforms, strong_transforms

# ----------------------------------------------------------------------------
# 步骤 5: 创建自定义的、感知不确定性的Dataset
class AleatoricAwareDataset(Dataset):
    def __init__(self, original_dataset, uncertainties, threshold):
        self.original_dataset = original_dataset
        self.uncertainties = uncertainties
        self.threshold = threshold
        self.gentle_transforms, self.strong_transforms = get_transforms()
        
        print(f"Aleatoric-aware dataset created. Threshold={self.threshold:.4f}")
        print(f"Number of samples with high uncertainty (gentle aug): {np.sum(self.uncertainties > self.threshold)}")
        print(f"Number of samples with low uncertainty (strong aug): {np.sum(self.uncertainties <= self.threshold)}")

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        # 从原始数据集中获取图像和标签
        # 假设original_dataset[idx]返回的是 (图像numpy数组, 标签)
        image, label = self.original_dataset[idx] 
        
        # 获取预计算的不确定性
        uncertainty = self.uncertainties[idx]

        # 根据不确定性应用不同的增强策略
        if uncertainty > self.threshold:
            # 高不确定性 -> 温和增强
            transformed = self.gentle_transforms(image=image)
        else:
            # 低不确定性 -> 强度增强
            transformed = self.strong_transforms(image=image)
        
        image_tensor = transformed['image']
        
        return image_tensor, label