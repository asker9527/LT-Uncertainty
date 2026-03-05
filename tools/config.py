def local_dataset(dataset_name):    
    train_path = f'/picassox/intelligent-cpfs/segmentation/intern_segmentation/dc1/Infinity/data/Asker9527/Remote_Sense_Datasets/{dataset_name}/train'  # 替换为你的训练集路径
    test_path = f'/picassox/intelligent-cpfs/segmentation/intern_segmentation/dc1/Infinity/data/Asker9527/Remote_Sense_Datasets/{dataset_name}/test'  # 替换为你的测试集路径
    
    return train_path, test_path

config_sgd = {
    'lr': 0.01,  # 基于批量大小的调整
    'weight_decay': 1e-4,
    'momentum': 0.9,
    'nesterov': False,
    'lr_schedule': 'cosine'
}
