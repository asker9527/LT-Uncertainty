import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models
import os
import gc
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import confusion_matrix
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tools import get_loss, local_dataset, configure_optimizer, CMO_weighted_train_loader
from tools import rand_bbox, sub_optimize_low_confidence, SharpenTransform
from tools.models import set_output_layer

# 定义一些超参数
model_name = 'resnet50'     # 'mobilenet_v2','efficientnet_b0','resnet18','resnet50
dataset_name = 'FGSC'           # FGSC, DIOR, DOTA
method_name = 'trust_decomposition'    # CE, trust, w_trust, trust_smooth
optim_name='warmup+cosine'      # warmup+cosine
activation = 'softplus'         # softplus, sigmoid, relu
is_pre = False
batch_size = 32
num_epochs = 100 if dataset_name == 'FGSC' else 30   # 训练总轮数
warmup_epochs = num_epochs/10  # 学习率预热阶段
max_lr = 1e-3       # 预热后的最大学习率0.001
min_lr = 1e-6       # 余弦退火的最小学习率

# save_path = './output/test9'
save_path = f'./output/{dataset_name}/{ "pretrained" if is_pre else "" }_{model_name}/v2/{method_name}_{optim_name}7-31-sig1-1-a0'


# 0. 路径管理
models_save_path = os.path.join(save_path,'models')
os.makedirs(models_save_path,exist_ok=True)
logs_save_path = os.path.join(save_path,'logs')
os.makedirs(logs_save_path,exist_ok=True)
results_save_path = os.path.join(save_path,'results')
os.makedirs(results_save_path,exist_ok=True)

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

# weights = pd.read_csv('./traindata_Ua.txt')
# weights = torch.tensor(weights.values.squeeze(), dtype=torch.float)
# weights = torch.softmax(-weights, dim=0) 
# sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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
if model_name == 'resnet50':
    model = models.resnet50(pretrained=is_pre)
elif model_name == 'vgg16':
    model = models.vgg16(pretrained=is_pre)
elif model_name == 'mobilenet_v2':
    model = models.mobilenet_v2(pretrained=is_pre)
elif model_name == 'efficientnet_b0':
    model = models.efficientnet_b0(pretrained=is_pre)
elif model_name == 'resnet18':
    model = models.resnet18(pretrained=is_pre)
else:
    raise NameError('Model Name Error')


# 4. 修改最后一层全连接层，以适应你的类别数
num_classes = len(train_dataset.classes)  # 计算类别数量
model = set_output_layer(model, num_classes, method_name, activation=activation, )

# model_path = r"E:\Github\LT-Uncertainty\output\FGSC\_reset50\trust_decomposition_warmup+cosine\models\reset50_bestmodel.pth"
# checkpoint = torch.load(model_path)  # 加载保存的模型
# model.load_state_dict(checkpoint)  # 将权重加载到模型中

# 5. 使用GPU（如果有）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(model)

# 6. 定义损失函数和优化器
criterion = get_loss(method_name)
optimizer, scheduler = configure_optimizer(
    model=model,
    optim_name=optim_name,
    max_lr=max_lr,
    min_lr=min_lr,
    num_epochs=num_epochs,
    warmup_epochs=warmup_epochs
)

writer = SummaryWriter(log_dir=logs_save_path)
best_eval_acc =0
best_avg_eval_acc = 0
best_train_acc = 0
best_train_avg_acc = 0

W = None
# 7. 训练模型
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    current_lr = optimizer.param_groups[0]['lr']
    
    if 'cmo' in method_name.lower():
        weighted_train_iter = iter(weighted_train_loader)

    for inputs, labels in tqdm(train_loader, desc="Train"):
        # CMO Data augmentation
        r = np.random.rand(1)
        mixup_prob = 0.5    # default
        beta = 1    # default
        cmo_gate = 0
        if 'cmo' in method_name.lower() and warmup_epochs < epoch < num_epochs - warmup_epochs and r < mixup_prob:
            cmo_gate = 1
            try:
                input2, target2 = next(weighted_train_iter)
            except:
                weighted_train_iter = iter(weighted_train_loader)
                input2, target2 = next(weighted_train_iter)
            # 强制裁剪到和 inputs 一样大小
            bs = inputs.size(0)
            input2 = input2[:bs]
            target2 = target2[:bs]

            # 设备转移
            input2 = input2.to(device)
            target2 = target2.to(device)
            input1, target1 = inputs.to(device), labels.to(device)

            lam = np.random.beta(beta, beta)
            bbx1, bby1, bbx2, bby2 = rand_bbox(input1.size(), lam)
            input1[:, :, bbx1:bbx2, bby1:bby2] = input2[:, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            inputs = input1
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input1.size()[-1] * input1.size()[-2]))
            labels = target1
        else:
            inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        if method_name == 'trust_smooth_cmo' and cmo_gate == 1:
            loss = criterion(target1,outputs,num_classes,epoch,num_epochs,cmo=True) * lam + criterion(target2,outputs,num_classes,epoch,num_epochs,cmo=True) * (1. - lam)
        elif method_name == 'trust_cmo' and cmo_gate == 1:
            loss = criterion(target1,outputs,num_classes,epoch,num_epochs) * lam + criterion(target2,outputs,num_classes,epoch,num_epochs) * (1. - lam)
        elif 'trust' in method_name:
            loss,A, B,C,D= criterion(labels,outputs,num_classes,epoch,num_epochs)
        elif cmo_gate == 1:
            loss = criterion(outputs, target1) * lam + criterion(outputs, target2) * (1. - lam)

        else:
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # 计算准确率
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # 收集预测和真实标签用于计算类别准确率
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        if A:
            with open(os.path.join(results_save_path,'loss.txt'), 'a') as f:  # 使用 'a' 模式追加写入
                        f.write(f"{loss.cpu()} ")
                        # f.write(f"{W.cpu()} ")
                        f.write(f"{A.cpu()} ")
                        f.write(f"{B.cpu()} ")
                        f.write(f"{C.cpu()} ")
                        f.write(f"{D.cpu()} \n")

                    
    if scheduler is not None:
        scheduler.step() 
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    # 计算每个类别的正确率
    classes = torch.unique(torch.tensor(all_labels)).tolist()
    class_correct = [0] * len(classes)
    class_total = [0] * len(classes)

    for label, pred in zip(all_labels, all_preds):
        for i, cls in enumerate(classes):
            if label == cls:
                class_total[i] += 1
                if label == pred:
                    class_correct[i] += 1

    class_acc = [100 * correct / total if total > 0 else 0 for correct, total in zip(class_correct, class_total)]
    avg_class_acc = np.mean(class_acc)
    
    # 打印信息
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, lr: {current_lr:.6f}")
    print(f"Accuracy: {epoch_acc:.2f}%")
    print(f"Average Class Accuracy: {avg_class_acc:.2f}%")
    
    # 记录到TensorBoard
    writer.add_scalar('Loss/train', epoch_loss, epoch)
    writer.add_scalar('Accuracy/train', epoch_acc, epoch)
    writer.add_scalar('Average Class Accuracy/train', avg_class_acc, epoch)
    for i, cls in enumerate(classes):
        writer.add_scalar(f'Class Accuracy/{cls}', class_acc[i], epoch)
    
    # 测试
    if best_train_acc <= epoch_acc or best_train_avg_acc <= avg_class_acc :
        best_train_acc = epoch_acc
        best_train_avg_acc = avg_class_acc

        model.eval()  # 设置模型为评估模式
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 计算每个类别的正确率
        classes = torch.unique(torch.tensor(all_labels)).tolist()
        class_correct = [0] * len(classes)
        class_total = [0] * len(classes)

        for label, pred in zip(all_labels, all_preds):
            for i, cls in enumerate(classes):
                if label == cls:
                    class_total[i] += 1
                    if label == pred:
                        class_correct[i] += 1

        class_acc = [100 * correct / total if total > 0 else 0 for correct, total in zip(class_correct, class_total)]
        avg_class_acc = np.mean(class_acc)
        test_acc = 100 * correct / total
        print(f"Test Accuracy after Epoch {epoch+1}: {test_acc:.2f}%")
        print(f"Avg Accuracy after Epoch {epoch+1}: {avg_class_acc:.2f}%")
        writer.add_scalar('Test Accuracy', test_acc, epoch)
        writer.add_scalar('Test Avg Accuracy', avg_class_acc, epoch)

        if best_eval_acc <= test_acc and best_avg_eval_acc <= avg_class_acc:
            best_eval_acc = test_acc
            best_avg_eval_acc = avg_class_acc
            # === 写入txt文件 ===
            results_file = os.path.join(results_save_path,'results.txt')
            with open(results_file, 'a') as f:  # 使用 'a' 模式追加写入
                f.write(f"Epoch {epoch+1}\n")
                f.write(f"Overall Accuracy: {test_acc:.2f}%\n")
                f.write(f"Average Class Accuracy: {avg_class_acc:.2f}%\n")
                for i, cls in enumerate(classes):
                    f.write(f"Class {cls} Accuracy: {class_acc[i]:.2f}%\n")
                f.write("-" * 30 + "\n")
            # 8. 保存模型
            # model_save_path = os.path.join(models_save_path,f'{model_name}_model{epoch}.pth')
            model_save_path = os.path.join(models_save_path,f'{model_name}_bestmodel.pth')
            torch.save(model.state_dict(), model_save_path)
    model_save_path = os.path.join(models_save_path,f'{model_name}_model{epoch}.pth')
    torch.save(model.state_dict(), model_save_path)
    # torch.cuda.empty_cache()
    # torch.cuda.ipc_collect()    

writer.close()
