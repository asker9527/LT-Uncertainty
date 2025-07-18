import torch.nn as nn

def set_output_layer(model, num_classes, method_name, activation='softplus'):
    """
    自动替换模型的输出层，并可选择添加激活层。

    参数:
    - model: 待修改模型
    - num_classes: 输出类别数
    - activation: 激活层（如 'softplus', 'relu', 'sigmoid'，或 None）

    返回:
    - 修改后的模型
    """
    def build_head(in_features):
        layers = [nn.Linear(in_features, num_classes)]
        if activation == 'softplus':
            layers.append(nn.Softplus())
        elif activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)

    if 'trust' not in method_name.lower():
        activation = None
    # 修改 fc
    if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
        in_features = model.fc.in_features
        model.fc = build_head(in_features)

    # 修改 head
    elif hasattr(model, 'head') and isinstance(model.head, nn.Linear):
        in_features = model.head.in_features
        model.head = build_head(in_features)

    # 修改 classifier
    elif hasattr(model, 'classifier'):
        if isinstance(model.classifier, nn.Linear):
            in_features = model.classifier.in_features
            model.classifier = build_head(in_features)

        elif isinstance(model.classifier, nn.Sequential):
            # 找最后一个 Linear 层替换
            for i in reversed(range(len(model.classifier))):
                if isinstance(model.classifier[i], nn.Linear):
                    in_features = model.classifier[i].in_features
                    model.classifier[i] = build_head(in_features)
                    break
    else:
        raise ValueError("模型结构未知，无法设置输出层")

    return model


