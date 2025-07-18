import os
import torch
import numpy as np
from PIL import Image
import random
import torchvision.transforms as T
import torch.nn.functional as F

def join_path(path1,path2):
    path = os.path.join(path1,path2)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def CMO_weighted_train_loader(cls_num_list, train_dataset, batch_size, weighted_alpha):
    cls_weight = 1.0 / (np.array(cls_num_list) ** weighted_alpha)
    cls_weight = cls_weight / np.sum(cls_weight) * len(cls_num_list)
    samples_weight = np.array([cls_weight[t] for t in train_dataset.targets])
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    print(samples_weight)
    weighted_sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight),
                                                                replacement=True)
    weighted_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                            pin_memory=True,
                                                            drop_last=True,
                                                            num_workers=0,
                                                            sampler= weighted_sampler)
    return weighted_train_loader

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class ClassMixOut:
    def __init__(self, prob=0.5, beta=1.0, start_epoch=0, end_epoch=80, total_epochs=100):
        self.prob = prob
        self.beta = beta
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch

    def rand_bbox(self, size, lam):
        W, H = size[2], size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def __call__(self, input, target, input2, target2, epoch):
        if not (self.start_epoch < epoch < self.end_epoch):
            return input, target, target2, 1.0, False

        if np.random.rand() > self.prob:
            return input, target, target2, 1.0, False

        lam = np.random.beta(self.beta, self.beta)
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(input.size(), lam)
        input[:, :, bbx1:bbx2, bby1:bby2] = input2[:, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size(-1) * input.size(-2)))

        return input, target, target2, lam, True



class CutPasteTensor:
    def __init__(self, area_ratio=0.1):
        self.area_ratio = area_ratio

    def __call__(self, img_tensor):
        """
        img_tensor: torch.Tensor of shape (C, H, W), values in [0, 1]
        """
        if not isinstance(img_tensor, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        if img_tensor.dim() != 3:
            raise ValueError("Input tensor must have shape (C, H, W)")

        C, H, W = img_tensor.shape
        patch_h = int(H * self.area_ratio)
        patch_w = int(W * self.area_ratio)

        # 1. 随机裁剪 patch
        y1 = random.randint(0, H - patch_h)
        x1 = random.randint(0, W - patch_w)
        patch = img_tensor[:, y1:y1 + patch_h, x1:x1 + patch_w].clone()

        # 2. 随机粘贴到其他位置
        y2 = random.randint(0, H - patch_h)
        x2 = random.randint(0, W - patch_w)
        img_tensor[:, y2:y2 + patch_h, x2:x2 + patch_w] = patch

        return img_tensor




def sub_optimize_low_confidence(num_classes,epoch,num_epochs,inputs, labels, W, model, optimizer,
                                 criterion,
                                 confidence_threshold,
                                 sub_loss_weight,
                                 device='cuda'):
    """
    对当前 batch 中置信度低的样本进行增强并再优化一次。
    
    Args:
        inputs (Tensor): 当前 batch 的输入，shape [B, C, H, W]
        labels (Tensor): 当前 batch 的标签，shape [B]
        outputs (Tensor): 当前 batch 的模型输出，shape [B, num_classes]
        model (nn.Module): 当前模型
        optimizer (Optimizer): 优化器
        augmentation_fn (Callable): 数据增强函数，输入单个 Tensor，输出增强后的 Tensor
        criterion (Callable): 损失函数
        confidence_threshold (float): 判断低置信度的阈值
        sub_loss_weight (float): 子优化 loss 的缩放因子
        device (str): 设备（'cuda' or 'cpu'）
    """
    augmentation_fn = T.Compose([
        # T.RandomHorizontalFlip(),
        # T.ColorJitter(0.2, 0.2, 0.2, 0.1),
        CutPasteTensor()
    ])

    # Step 1: 计算置信度
    with torch.no_grad():
        low_conf_mask = W.squeeze(1) > confidence_threshold
        low_conf_indices = torch.nonzero(low_conf_mask, as_tuple=False).squeeze(1)

    if low_conf_indices.numel() == 0:
        return  # 无低置信度样本，直接返回

    # Step 2: 获取低置信度样本并增强
    low_inputs = inputs[low_conf_indices].detach().to(device)
    low_labels = labels[low_conf_indices].detach().to(device)

    # 如果只有一个样本，防止 stack 报错
    if low_inputs.ndim == 3:
        low_inputs = low_inputs.unsqueeze(1)
        low_labels = low_labels.unsqueeze(1)

    # 增强
    low_inputs_aug = torch.stack([augmentation_fn(img) for img in low_inputs]).to(device)

    # Step 3: 前向 & 计算 loss
    model.train()  # ensure model is in training mode
    optimizer.zero_grad()
    low_outputs = model(low_inputs_aug)
    sub_loss,_ = criterion(low_labels,low_outputs,num_classes,epoch,num_epochs) 
    # Step 4: 反向传播 & 优化
    sub_loss *=  sub_loss_weight
    sub_loss.backward()
    optimizer.step()
    # print(f"Aug samples{low_inputs.shape[0]}")



def sharpen_filter(img_tensor, kernel_size=3):
    # 假设 img_tensor: [C, H, W] (float, range 0~1)
    sharpening_kernel = torch.tensor([[[[0, -1, 0],
                                        [-1, 5, -1],
                                        [0, -1, 0]]]], dtype=torch.float32)  # shape [1,1,3,3]
    # 适用于多通道
    kernel = sharpening_kernel.repeat(img_tensor.shape[0], 1, 1, 1).to(img_tensor.device)
    img_tensor = img_tensor.unsqueeze(0)  # [1, C, H, W]
    sharpened = F.conv2d(img_tensor, kernel, padding=1, groups=img_tensor.shape[1])
    return sharpened.squeeze(0).clamp(0, 1)

class SharpenTransform(torch.nn.Module):
    def forward(self, img):
        return sharpen_filter(img)