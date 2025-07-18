def local_dataset(dataset_name):
    if dataset_name == 'HRSC':
        train_path = r'D:\BaiduNetdiskDownload\Remote_Sense_Datasets\HRSC\train'  # 替换为你的训练集路径
        test_path = r'D:\BaiduNetdiskDownload\Remote_Sense_Datasets\HRSC\test'    # 替换为你的测试集路径
    if dataset_name == 'DIOR':
        train_path = r'D:\BaiduNetdiskDownload\Remote_Sense_Datasets\DIOR\train'  # 替换为你的训练集路径
        test_path = r'D:\BaiduNetdiskDownload\Remote_Sense_Datasets\DIOR\test'    # 替换为你的测试集路径
    if dataset_name == 'DOTA':
        train_path = r'D:\BaiduNetdiskDownload\Remote_Sense_Datasets\DOTA\train'  # 替换为你的训练集路径
        test_path = r'D:\BaiduNetdiskDownload\Remote_Sense_Datasets\DOTA\test'    # 替换为你的测试集路径
    if dataset_name == 'FGSC':
        train_path = r'D:\BaiduNetdiskDownload\Remote_Sense_Datasets\FGSC\train'  # 替换为你的训练集路径
        test_path = r'D:\BaiduNetdiskDownload\Remote_Sense_Datasets\FGSC\test'    # 替换为你的测试集路径
    
    return train_path, test_path

config_sgd = {
    'lr': 0.01,  # 基于批量大小的调整
    'weight_decay': 1e-4,
    'momentum': 0.9,
    'nesterov': False,
    'lr_schedule': 'cosine'
}
