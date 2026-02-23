import torch.nn.functional as F
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets, models
import os
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import confusion_matrix
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tools import get_loss, local_dataset, configure_optimizer, CMO_weighted_train_loader, rand_bbox
from torch.special import digamma, gammaln

def edl_entropy_decomposition(alpha: torch.Tensor):
    """
    计算基于熵的 EDL 不确定性分解（Total / Epistemic / Aleatoric）

    参数：
        alpha: Tensor, shape (B, K)
            Dirichlet 参数（通常 = softplus(logits) + 1）

    返回：
        total_entropy:   (B,) 期望分布的熵（总不确定性）
        epistemic_entropy: (B,) Dirichlet 熵（模型不确定性）
        aleatoric_entropy: (B,) 数据不确定性
    """
    alpha = torch.clamp(alpha, min=1e-6) 
    S = torch.sum(alpha)     # (B, 1)
    probs = alpha / S                             # E[p_k], shape (B, K)
    
    # Total entropy of expected probability
    total_entropy = -torch.sum(probs * torch.log(probs))  # (B,)

    # Dirichlet entropy = epistemic
    psi_S = digamma(S + 1)  # [B, 1]
    psi_alpha = digamma(alpha + 1)  # [B, C]
    aleatoric = torch.sum(probs * (psi_S - psi_alpha))

    # Aleatoric = total - epistemic
    epistemic = total_entropy - aleatoric

    return total_entropy.item(), epistemic.item(), aleatoric.item()


def test_single_sample(model, dataloader, device):
    model.eval()
    model.to(device)

    results = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing"):
            inputs, labels = batch  # 必须是3个元素
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            total_entropy, epistemic, aleatoric = edl_entropy_decomposition(outputs)
            for i in range(len(labels)):
                result = {
                    'label': labels[i].item(),
                    'pred': preds[i].item(),
                    'correct': preds[i].item() == labels[i].item(),
                    'Up': total_entropy,
                    'Ue': epistemic,
                    'Ua': aleatoric,
                    'K/s': num_classes/torch.sum(outputs).item()
                }
                results.append(result)

    return results

# 定义一些超参数
model_name = 'resnet50'
dataset_name = 'FGSC'
method_name = 'trust_decomposition'
optim_name='warmup+cosine'
is_pre = False
batch_size = 32
num_epochs = 100   # 训练总轮数
warmup_epochs = 10  # 学习率预热阶段
max_lr = 1e-3      # 预热后的最大学习率0.001
min_lr = 1e-6       # 余弦退火的最小学习率

# save_path = './output/test_W'
save_path = f'./output/{dataset_name}/{ "pretrained" if is_pre else "" }_{model_name}/v2/{method_name}_{optim_name}7-31-sig1-1-a0'
for i in range(100):
# i=1
    model_path = f"./output/FGSC/_resnet50/v2/trust_decomposition_warmup+cosine7-31-sig1-1-a0/models/resnet50_model{i}.pth"
    # 0. 路径管理
    # models_save_path = os.path.join(save_path,'models')
    # os.makedirs(models_save_path,exist_ok=True)
    # logs_save_path = os.path.join(save_path,'logs')
    # os.makedirs(logs_save_path,exist_ok=True)
    results_save_path = os.path.join(save_path,'results')
    os.makedirs(results_save_path,exist_ok=True)
    # results_save_path = os.path.join(save_path,'results')
    # os.makedirs(results_save_path,exist_ok=True)

    # 1. 数据预处理
    image_size=512
    crop_size=448
    train_transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.RandomCrop(crop_size, padding=8),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
    test_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop((crop_size, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    train_path, test_path = local_dataset(dataset_name)   # 替换为你的测试集路径

    # 2. 加载数据集
    train_dataset = datasets.ImageFolder(root=train_path, transform=train_transform)
    test_dataset = datasets.ImageFolder(root=test_path, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    print(train_dataset.classes)  # 类别名称列表
    print(train_dataset.class_to_idx)  # 类别到索引的映射

    # 如果是 Tensor，先转换成 list
    label_list = train_dataset.targets
    if isinstance(label_list, torch.Tensor):
        label_list = label_list.tolist()

    cls_num_list = Counter(label_list)
    print(cls_num_list)

    if 'cmo' in method_name.lower():
        weighted_train_loader = CMO_weighted_train_loader(cls_num_list=list(cls_num_list.values()), train_dataset=train_dataset,
                                                        batch_size=batch_size, weighted_alpha=1)

    # 3. 加载预训练的ResNet50模型
    model = models.resnet50(pretrained=is_pre)

    # 4. 修改最后一层全连接层，以适应你的类别数
    num_classes = len(train_dataset.classes)  # 计算类别数量
    if 'trust' in method_name:
        print('softplus')
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, num_classes),
            nn.Softplus()
        )
    else:
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, num_classes),
        )



    checkpoint = torch.load(model_path)  # 加载保存的模型
    model.load_state_dict(checkpoint)  # 将权重加载到模型中



    # 5. 使用GPU（如果有）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 3. 测试模型
    results = test_single_sample(model, test_loader, device)
    results_train = test_single_sample(model, train_loader, device)
    # 4. 保存结果
    output_csv = os.path.join(results_save_path,f'test_results_{i}.csv')
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"测试完成，结果保存至 {output_csv}")

    output_csv = os.path.join(results_save_path,f'train_results_{i}.csv')
    df = pd.DataFrame(results_train)
    df.to_csv(output_csv, index=False)
    print(f"测试完成，结果保存至 {output_csv}")