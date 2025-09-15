# -*- coding: utf-8 -*-
import gc  # 垃圾回收模块
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from tqdm import tqdm
from Data_Loader import Data_Loader, MultiModalDataset, RGBLoader
from model import *
from Augmentation import HyperspectralAugmentation, HyperspectralRandAugment
import time
from torchvision import transforms
import torch
import copy
from PIL import Image
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
import random
from train import train, evaluate
import torch.optim as optim

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='main.log',
    filemode='a'
)

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
SEED = 0
set_seed(SEED)

activation_dict = {
    'relu': nn.ReLU(),
    'leaky_relu': nn.LeakyReLU(),
    'gelu': nn.GELU(),
    'selu': nn.SELU(),
    'tanh': nn.Tanh(),
    'sigmoid': nn.Sigmoid(),
    'swish': nn.SiLU(),
    'prelu': nn.PReLU(),
    'elu': nn.ELU(alpha=1.0),
    'selu': nn.SELU(),
}

def compute_rgb_stats(paths, sample_size=200):
    sample_paths = random.sample(paths, min(sample_size, len(paths)))
    pixels = []
    for path in sample_paths:
        img = Image.open(path).convert('RGB')
        img = transforms.Resize((224, 224))(img)
        img = transforms.ToTensor()(img)
        pixels.append(img)
    pixels = torch.stack(pixels)
    return pixels.mean(dim=(0, 2, 3)), pixels.std(dim=(0, 2, 3))

def create_optimizer(optimizer_name='Adam', learning_rate=0.001, weight_decay=0.0001, momentum=0.9):
    if optimizer_name == "AdamW":
        return optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'Adam':
        return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        return optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

start = time.time()
        
# 加载固定划分数据
trainval_pairs = torch.load("splits/trainval.pt")
test_pairs     = torch.load("splits/cleantest.pt")
# trainval_pairs = torch.load("splits_clean/trainval.pt")
# test_pairs     = torch.load("splits_clean/test.pt")

rgb_trainval = [p[0] for p in trainval_pairs]
X_trainval   = torch.stack([p[1] for p in trainval_pairs])
y_trainval   = torch.tensor([p[2] for p in trainval_pairs])

rgb_test = [p[0] for p in test_pairs]
X_test   = torch.stack([p[1] for p in test_pairs])
y_test   = torch.tensor([p[2] for p in test_pairs])

mean_hsi = X_trainval.mean(dim=(0, 2, 3))
std_hsi = X_trainval.std(dim=(0, 2, 3))
transform_hsi = transforms.Normalize(mean=mean_hsi.tolist(), std=std_hsi.tolist())


rgb_mean, rgb_std = compute_rgb_stats(rgb_trainval)
transform_rgb = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=rgb_mean, std=rgb_std)
])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16

mymodels = {
    'MultiModalNet': (MultiModalNet, {
        'optimizer': ['Adam'], 'learning_rate': [0.001], 'weight_decay': [0.0001],
        'fuse': ['concat'], 'dropout': [0.5], 'dim_head': [4], 'warmup': [4],
        'cos': [False], 'mid_dim': [256], 'token_dim': [64], 'mlp_dim': [512],
        'act': ['leaky_relu'], 'epochs':[50],
    }),
    # 'MultiModalNet': (MultiModalNetCompare, {
    #     'optimizer': ['Adam'],
    #     'learning_rate': [0.001],
    #     'weight_decay': [0.0001],
    #     'fuse': ['concat', 'rgb', 'hsi'],  # 支持四种模式
    #     'dropout': [0.5],
    #     'heads': [4],          # 统一保留
    #     'dim_head': [16],      # 即使没用也保留
    #     'warmup': [4],         # 预热轮次
    #     'cos': [False],        # 是否使用cosine调度
    #     'mid_dim': [256],      # 中间层维度（有些模型会用到）
    #     'token_dim': [1024],   # encoder输出维度
    #     'mlp_dim': [512],      # MLP hidden dim
    #     'act': ['gelu'],       # 激活函数（统一保留）
    #     'epochs': [50],
    #     'backbone': ['resnet34'], # backbone 选择
    #     'use_vit': [False],       # 是否启用 ViT encoder
    #     'c': [313],               # HSI 通道数
    #     # 'num_classes': [4],       # 分类数
    # }),
}

test_dataset = MultiModalDataset(rgb_test, X_test, y_test,transform_rgb=transform_rgb,transform_hsi=transform_hsi)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

end = time.time()
# print(f"加载数据的时间：{end-start:.4f}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

all_fold_results = []
for model_name, (model_class, grid) in mymodels.items():
    for params in ParameterGrid(grid):
        fold_metrics = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_trainval, y_trainval), 1):
            start = time.time()
            X_train = X_trainval[train_idx]
            y_train = y_trainval[train_idx]
            rgb_train = [rgb_trainval[i] for i in train_idx]

            X_val = X_trainval[val_idx]
            y_val = y_trainval[val_idx]
            rgb_val = [rgb_trainval[i] for i in val_idx]

            train_dataset = MultiModalDataset(rgb_train, X_train, y_train, transform_rgb, transform_hsi)
            val_dataset = MultiModalDataset(rgb_val, X_val, y_val, transform_rgb, transform_hsi)

            # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=4, pin_memory=True, prefetch_factor=2)
            val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                          num_workers=4, pin_memory=True, prefetch_factor=2)

            
            set_seed(1)
            activation = activation_dict[params['act'].lower()]
            model = model_class(
                fuse_method=params['fuse'],
                dropout=params['dropout'],
                heads=params['token_dim'] // params['dim_head'],
                dim_head=params['dim_head'],
                token_dim=params['token_dim'],
                mlp_dim=params['mlp_dim'],
                activation=activation,
                c=313,
                # backbone=params['backbone'],
                # use_vit=params['use_vit'],
            ).to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = create_optimizer(
                optimizer_name=params['optimizer'],
                learning_rate=params['learning_rate'],
                weight_decay=params.get('weight_decay', 0),
            )
            
            end = time.time()
            # print(f"循环中加载数据的时间：{end-start:.4f}")
            result = train(
                model=model,
                model_name=model_name,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                num_epochs=params['epochs'],
                criterion=criterion,
                device=device,
                warmup_epochs=params.get('warmup_epochs', params['warmup']),
                use_cosine_decay=params.get('cos', params['cos']),
                run=fold
            )
            start = time.time()
            best_model = result['model']
            os.makedirs('saved_models', exist_ok=True)
            torch.save({
                'model_state_dict': best_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f'saved_models/{model_name}_fold{fold}_best.pth')

            print(f"Saved best model to saved_models/{model_name}_fold{fold}_best.pth")

            # test_metrics = evaluate(best_model, test_loader, device=device)
            class_names = ['Healthy', '2dpi', '4dpi', '6dpi'] 
            test_metrics = evaluate(best_model, test_loader, class_names=class_names, device=device)
            fold_metrics.append(test_metrics)

            logging.info(f"Fold {fold} Test Accuracy: {test_metrics['test_accuracy']:.4f}")

            del model, result, train_loader, val_loader, train_dataset, val_dataset
            torch.cuda.empty_cache()
            gc.collect()
            
            # print(f"训练之外的时间：{end-start:.4f}")
            print("clean")
        mean_metrics = {}
    
    # 提取所有折的同一个指标，然后计算均值和标准差
    for key in fold_metrics[0].keys():
        # 我们只对数值类型的指标进行计算
        if isinstance(fold_metrics[0][key], (int, float)):
            all_values = [m[key] for m in fold_metrics]
            mean_metrics[f'mean_{key}'] = np.mean(all_values)
            mean_metrics[f'std_{key}'] = np.std(all_values)

    # 将这组参数的最终结果（包含每一折的详情和平均值）保存起来
    current_model_result = {
        'model': model_name,
        'params': params,
        'fold_metrics': fold_metrics,
        'mean_metrics': mean_metrics  # 保存所有指标的均值和标准差
    }
    # all_fold_results.append(current_model_result) # 假设 all_fold_results 在外部定义

    # --- 将当前模型的结果以追加模式写入文本文件 ---
    # 这是你要求的核心部分
    output_filepath = "store/5fold_results.txt"
    print(f"\nAppending results for model {model_name} to {output_filepath}")
    
    with open(output_filepath, "a+") as f:
        f.write("="*80 + "\n")
        f.write(f"Model: {current_model_result['model']}\n")
        f.write(f"Params: {current_model_result['params']}\n\n")
        
        # --- 写入平均指标 (格式化，更适合论文) ---
        f.write("[Overall 5-Fold Results (Mean ± Std)]\n")
        f.write(f"  - Accuracy:           {mean_metrics['mean_test_accuracy']:.4f} ± {mean_metrics['std_test_accuracy']:.4f}\n")
        f.write(f"  - AUROC (Macro OvR):    {mean_metrics.get('mean_test_auroc_ovr_macro', float('nan')):.4f} ± {mean_metrics.get('std_test_auroc_ovr_macro', float('nan')):.4f}\n")
        f.write(f"  - F1 (Weighted):        {mean_metrics['mean_test_f1_weighted']:.4f} ± {mean_metrics['std_test_f1_weighted']:.4f}\n")
        f.write(f"  - Precision (Weighted): {mean_metrics['mean_test_precision_weighted']:.4f} ± {mean_metrics['std_test_precision_weighted']:.4f}\n")
        f.write(f"  - Recall (Weighted):    {mean_metrics['mean_test_recall_weighted']:.4f} ± {mean_metrics['std_test_recall_weighted']:.4f}\n\n")

        f.write("  [Per-class Accuracy (Recall)]\n")
        for name in class_names:
            key_mean = f'mean_test_acc_{name.replace(" ", "_").lower()}'
            key_std = f'std_test_acc_{name.replace(" ", "_").lower()}'
            f.write(f"    - {name}: {mean_metrics[key_mean]:.4f} ± {mean_metrics[key_std]:.4f}\n")
        f.write("\n")
        
        f.write("  [Binary Metrics (Healthy vs Diseased)]\n")
        f.write(f"    - Accuracy:  {mean_metrics['mean_test_binary_accuracy']:.4f} ± {mean_metrics['std_test_binary_accuracy']:.4f}\n")
        f.write(f"    - AUROC:     {mean_metrics.get('mean_test_binary_auroc', float('nan')):.4f} ± {mean_metrics.get('std_test_binary_auroc', float('nan')):.4f}\n")
        f.write(f"    - F1-score:  {mean_metrics['mean_test_binary_f1']:.4f} ± {mean_metrics['std_test_binary_f1']:.4f}\n")
        f.write(f"    - Precision: {mean_metrics['mean_test_binary_precision']:.4f} ± {mean_metrics['std_test_binary_precision']:.4f}\n")
        f.write(f"    - Recall:    {mean_metrics['mean_test_binary_recall']:.4f} ± {mean_metrics['std_test_binary_recall']:.4f}\n\n")

        # --- 写入每一折的详细指标 ---
        f.write("[Detailed Fold-by-Fold Metrics]\n")
        for i, metrics in enumerate(current_model_result['fold_metrics'], 1):
            f.write(f"  Fold {i}: Acc={metrics['test_accuracy']:.4f}, "
                    f"AUROC={metrics.get('test_auroc_ovr_macro', float('nan')):.4f}, "
                    f"F1={metrics['test_f1_weighted']:.4f}, "
                    f"Bin_Acc={metrics['test_binary_accuracy']:.4f}, "
                    f"Bin_F1={metrics['test_binary_f1']:.4f}\n")
        f.write("="*80 + "\n\n")

    print("Results successfully appended.")

#         all_fold_results.append({
#             'model': model_name,
#             'params': params,
#             'fold_metrics': fold_metrics,
#             'mean_accuracy': np.mean([m['test_accuracy'] for m in fold_metrics])
#         })

#         with open("store/5fold_results.txt", "a+") as f:
#             for result in all_fold_results:
#                 f.write(f"\nModel: {result['model']}\n")
#                 f.write(f"Params: {result['params']}\n")
#                 f.write(f"5-Fold Mean Accuracy: {result['mean_accuracy']:.4f}\n")
#                 for fold, metrics in enumerate(result['fold_metrics'], 1):
#                     f.write(f"Fold {fold}: Acc={metrics['test_accuracy']:.4f}, "
#                             f"Precision={metrics['test_precision']:.4f}, "
#                             f"Recall={metrics['test_recall']:.4f}, "
#                             f"F1={metrics['test_f1']:.4f}\n")
