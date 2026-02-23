import torch
from torch import nn 
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from torch.special import digamma, gammaln

def get_loss(method_name):
    if method_name == 'CE' or 'base':
        loss_fuc = nn.CrossEntropyLoss()
    if method_name == 'CE_cmo':
        loss_fuc = nn.CrossEntropyLoss()
    if method_name == 'trust':
        loss_fuc = trust_ce_loss
    if method_name == 'w_trust':
        loss_fuc = wtrust_ce_loss
    if method_name.lower() in ['focal','focal loss','fl']:
        loss_fuc = FocalLoss()
    if method_name == 'fix_trust_and_ce':
        loss_fuc = fix_trust_ce_loss
    if method_name == 'trust_decomposition':
        loss_fuc = uncertainty_weighted_loss
    if method_name == 'trust_cmo':
        loss_fuc = uncertainty_weighted_loss
    if method_name == 'trust_smooth':
        loss_fuc = uncertainty_weighted_smooth_loss
    if method_name == 'trust_smooth_cmo':
        loss_fuc = uncertainty_weighted_smooth_loss
    if method_name == 'soft_label':
        loss_fuc = soft_label_cross_entropy
    print(f"Loss:{method_name}")
    return loss_fuc
    
def KL(alpha, c):
    beta = torch.ones((1, c)).cuda()
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl

def rank_normalize_1d(x: torch.Tensor) -> torch.Tensor:
    """
    对 1D tensor 进行排序归一化，返回值域在 (0, 1] 中。
    """
    ranks = x.argsort().argsort().float() + 1  # rank 从 1 开始
    normalized = ranks / ranks.max()
    return normalized

def norm_exp(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Min-max normalization for a and b
    a_norm = (a - a.min()) / (a.max() - a.min() + 1e-6)
    b_norm = (b - b.min()) / (b.max() - b.min() + 1e-6)
    # a_norm = rank_normalize_1d(a)
    # b_norm = rank_normalize_1d(b)

    return torch.exp(a_norm - b_norm)

# p:label alpha:Dirichlet parameter
def trust_ce_loss(p, alpha, c, global_step, annealing_step):
    alpha = torch.clamp(alpha, min=1)  # 或 1e-4
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p.long(), num_classes=c)
    A = torch.sum(label.cuda() * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
    annealing_coef = min(1, global_step / annealing_step)

    alp = E.cuda() * (1 - label.cuda()) + 1
    B = annealing_coef * KL(alp, c)
    return (A + B).mean(), KL(alp, c).mean(), A.mean(), B.mean()

def trust_mse_loss(p, alpha, c, global_step, annealing_step=1):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    m = alpha / S
    label = F.one_hot(p, num_classes=c)
    A = torch.sum((label - m) ** 2, dim=1, keepdim=True)
    B = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    C = annealing_coef * KL(alp, c)
    return (A + B) + C

# reweight loss based on Dirichlet uncertainty
def wtrust_ce_loss(p, alpha, c, global_step, annealing_step):
    alpha = torch.clamp(alpha, min=1e-6)  # 或 1e-4
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p.long(), num_classes=c)
    A = torch.sum(label.cuda() * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
    annealing_coef = min(1, global_step / annealing_step)
    
    alp = E.cuda() * (1 - label.cuda()) + 1
    B = annealing_coef * KL(alp, c)
    uncertainty_weight = (c / S.detach())**5
    # print(f'w:{uncertainty_weight}\n')
    return (uncertainty_weight*A + B).mean(),uncertainty_weight.mean(), A.mean(), B.mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean', eps=1e-8):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.eps = eps

        if isinstance(alpha, (list, torch.Tensor)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        elif alpha is not None:
            raise TypeError("alpha must be None, list, or torch.Tensor")

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)  # retain gradient
        targets = targets.view(-1, 1)
        pt = probs.gather(1, targets).clamp(min=self.eps)
        log_pt = log_probs.gather(1, targets)

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device).gather(0, targets.squeeze())
            loss = -alpha_t.view(-1, 1) * (1 - pt) ** self.gamma * log_pt
        else:
            loss = -(1 - pt) ** self.gamma * log_pt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss



def fix_trust_ce_loss(p, alpha, c, global_step, annealing_step):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p.long(), num_classes=c)
    A = torch.sum(label.cuda() * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
    annealing_coef = min(1, global_step / annealing_step)
    # print(f'alpha:{alpha}\n')
    alp = E.cuda() * (1 - label.cuda()) + 1
    B = annealing_coef * KL(alp, c)
    uncertainty_weight = (c / torch.clamp(S.detach(), min=1.0))
    if global_step < annealing_step*2/3:
        return (A + B).mean()
    else:
        return (uncertainty_weight*A+B).mean()
    


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
    S = torch.sum(alpha, dim=1, keepdim=True)     # (B, 1)
    probs = alpha / S                             # E[p_k], shape (B, K)
    
    # Total entropy of expected probability
    total_entropy = -torch.sum(probs * torch.log(probs), dim=1)  # (B,)

    # Dirichlet entropy = epistemic
    psi_S = digamma(S + 1)  # [B, 1]
    psi_alpha = digamma(alpha + 1)  # [B, C]
    aleatoric = torch.sum(probs * (psi_S - psi_alpha), dim=1)

    # Aleatoric = total - epistemic
    epistemic = total_entropy - aleatoric

    return total_entropy, epistemic, aleatoric


def adaptive_soft_label(one_hot, uncertainty, alpha=0.2):
    """
    one_hot:      [B, C]      one-hot 标签
    uncertainty:  [B]         已获得的不确定性（如预测方差、entropy 等）
    alpha:        float       平滑强度上限
    """
    # 将不确定性映射到 (0, alpha) 区间
    smooth = alpha * torch.sigmoid(uncertainty/torch.log(torch.tensor(one_hot.size(1)))).unsqueeze(1)  # [B, 1]
    # smooth = alpha * torch.sigmoid(uncertainty).unsqueeze(1)  # [B, 1]

    # 构造 soft label
    soft_label = one_hot * (1 - smooth) + smooth / one_hot.size(1)
    return soft_label


import torch
import torch.nn.functional as F

def soft_label_cross_entropy(preds, targets, reduction='mean', label_smoothing=0.0):
    """
    计算 soft label 的交叉熵损失，自动支持 hard label（类别索引）输入。

    参数:
    - preds: 模型输出 logits，形状 [batch_size, num_classes]
    - targets: 可以是 soft label [B, C]，也可以是 hard label [B]
    - reduction: 'mean' | 'sum' | 'none'
    - label_smoothing: float, 可选的 label smoothing 系数（0 表示不平滑）

    返回:
    - loss: 标量或 shape [B] 的损失
    """
    num_classes = preds.size(-1)

    # 如果是整数类型，就转换为 one-hot（带 smoothing）
    if targets.dtype in [torch.long, torch.int, torch.int64]:
        # 转为 one-hot
        targets = F.one_hot(targets, num_classes=num_classes).float()

    # 如果需要 label smoothing，就应用平滑
    if label_smoothing > 0:
        smooth_value = label_smoothing / num_classes
        targets = (1.0 - label_smoothing) * targets + smooth_value

    # Log softmax
    log_probs = F.log_softmax(preds, dim=-1)

    # 交叉熵公式
    loss = -(targets * log_probs).sum(dim=-1)

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


def uncertainty_weighted_loss(p, alpha, c, global_step, annealing_step, epsilon=1e-6):
    alpha = torch.clamp(alpha, min=epsilon)  # 或 1e-4
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p.long(), num_classes=c)
    # calculate uncertainty by entropy
    Up, Ue, Ua = edl_entropy_decomposition(alpha=alpha)

    A = torch.sum(label.cuda() * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
    annealing_coef = min(1, global_step / annealing_step)

    alp = E.cuda() * (1 - label.cuda()) + 1
    B = annealing_coef * KL(alp, c)

    weights = (c / torch.clamp(S, min=1.0))**1

    weighted_loss = (weights.detach() * A + B).mean()

    return weighted_loss, Up.mean(), Ue.mean(), Ua.mean(), (c / torch.clamp(S, min=1.0)).mean()

def uncertainty_weighted_smooth_loss(p, alpha, c, global_step, annealing_step, epsilon=1e-6,cmo=False):
    alpha = torch.clamp(alpha, min=epsilon)  # 或 1e-4
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1

    # alpha = torch.clamp(alpha, min=epsilon) + 1  # 或 1e-4
    # S = torch.sum(alpha, dim=1, keepdim=True)
    # E = alpha - 1

    label = F.one_hot(p.long(), num_classes=c)
    # calculate uncertainty by entropy
    Up, Ue, Ua = edl_entropy_decomposition(alpha=alpha)
    if cmo:
        soft_label = label
    else:
        soft_label = adaptive_soft_label(label, Ua, alpha=0.2)
        # soft_label = label
    A = torch.sum(soft_label.cuda() * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
    annealing_coef = min(1, global_step / annealing_step)

    alp = E.cuda() * (1 - soft_label.cuda()) + 1
    B = annealing_coef * KL(alp, c)

    weights = (2*c / torch.clamp(S, min=1.0))**3
    if cmo:
        weighted_loss = (A + B).mean()
    else:
        weighted_loss = (weights.detach() * A+B).mean()

    return weighted_loss.mean(), Up.mean(), Ue.mean(), Ua.mean(), (c / torch.clamp(S, min=1.0)).mean()
