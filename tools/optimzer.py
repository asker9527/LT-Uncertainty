from torch import optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, SequentialLR, LinearLR
from .config import config_sgd

def configure_optimizer(model, optim_name, max_lr=1e-4, min_lr=1e-6, num_epochs=100, warmup_epochs=10):
    """
    配置优化器和学习率调度器

    参数:
    - model: 神经网络模型
    - optim_name: 例如 "warmup+cosine"
    - max_lr: 最高学习率
    - min_lr: 最低学习率（仅用于 CosineAnnealing）
    - num_epochs: 总训练轮数
    - warmup_epochs: 预热轮数

    返回:
    - optimizer
    - scheduler
    """
    
    optimizer = optim.Adam(model.parameters(), lr=max_lr)
    scheduler = None

    if optim_name == 'warmup+cosine':
        # 预热调度器
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,    # 初始为 1% 学习率
            end_factor=1.0,
            total_iters=warmup_epochs
        )

        # 余弦退火调度器
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_epochs - warmup_epochs,
            eta_min=min_lr
        )

        # 顺序组合调度器
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )
    if optim_name == 'cosine':
        # 余弦退火调度器
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=min_lr
        )
    if optim_name == 'SGD':
        config = config_sgd
        optimizer = optim.SGD(
            model.parameters(),
            lr=config['lr'],
            momentum=config_sgd.get('momentum', 0.9),   # 0.9 is default value
            weight_decay=config['weight_decay'],
        )
    

    return optimizer, scheduler
