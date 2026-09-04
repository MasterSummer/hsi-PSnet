
# -*- coding: utf-8 -*-
import os
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
from Data_Loader import Data_Loader, MultiModalDataset, RGBLoader
from model import *
from Augmentation import HyperspectralAugmentation, HyperspectralRandAugment
# from Methods.ELM import ELMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
import time
from torchvision import datasets  
import torch
import copy
from PIL import Image
import seaborn as sns
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, TensorDataset
import logging
import random
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split, ParameterGrid
from train import train, evaluate
import torch.optim as optim
from torch.utils.data import Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='main.log',  # 指定日志文件名
    filemode='a'  # 指定文件模式，'a' 表示追加，'w' 表示覆盖
)
spa_col = [59, 92, 97, 95, 49, 28, 182, 68, 44, 8, 111, 168, 305,
           3, 140, 300, 55, 288, 149, 82, 278, 307, 312, 283, 101, 158,
           0, 21, 266, 129, 226]

def load_rgb_images(image_paths, resize=(224, 224)):
    """加载 RGB 图像并转换为 Tensor"""
    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
    ])
    images = []
    for path in image_paths:
        img = Image.open(path).convert('RGB')  # 确保是 RGB 格式
        img = transform(img)  # 转换为 Tensor [C, H, W]
        images.append(img)
    return torch.stack(images)  # [N, C, H, W]

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def create_optimizer(optimizer_name='Adam', learning_rate=0.001, weight_decay=0.0001, momentum=0.9):
    if optimizer_name == "AdamW":
        return optim.AdamW( model.parameters(), lr=learning_rate,weight_decay=weight_decay)
    elif optimizer_name == 'Adam':
        return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        return optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")



# ------------------------data prepare----------------------------------------------------------------------
SEED = 42
timestamp = int(time.time())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16  # 设置批量大小
activation_dict = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'gelu': nn.GELU(),
            'selu': nn.SELU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'swish': nn.SiLU(),  # SiLU就是Swish激活函数
            'prelu': nn.PReLU(),                              # 可学习参数的 LeakyReLU
            'elu': nn.ELU(alpha=1.0),                         # alpha 默认为 1
            'selu': nn.SELU(),                                # 自归一化 ELU
        }
data_loader = Data_Loader()
data_loader.load_data()
X, y = data_loader.get_data()
n, h, w, c = X.shape
X_tensor = torch.tensor(X, dtype=torch.float32)  # (n, h, w, c)
X = X_tensor.permute(0, 3, 1, 2)  # 转换为 (n, c, h, w)
y = torch.tensor(y, dtype=torch.long)  # 适用于分类任务
assert len(X) == len(y), "X and y must have the same number of samples"

# 2️⃣ **获取 RGB 图像路径**
rgb_loader = RGBLoader(data_root="RGB")  # rgb_root需要定义或使用默认值
rgb_image_paths, rgb_labels = rgb_loader.load_paths()
# 3️⃣ **确保 RGB 和 HSI 数据对齐**
assert len(rgb_image_paths) == len(X), "RGB 和 HSI 数量不匹配！"
assert rgb_labels == y.tolist(), "RGB 和 HSI 标签不匹配！"

data_pairs = list(zip(rgb_image_paths, X, y))  # [(RGB路径, HSI, 标签), ...]
train_pairs, temp_pairs = train_test_split(data_pairs, test_size=0.2, stratify=y, random_state=SEED)
val_pairs, test_pairs = train_test_split(temp_pairs, test_size=0.5, stratify=[p[2] for p in temp_pairs], random_state=SEED)

trainval_pairs = train_pairs + val_pairs
# # 保存为文件（方便后续加载）
# os.makedirs("splits", exist_ok=True)
# torch.save(trainval_pairs, "splits/trainval.pt")
# torch.save(test_pairs, "splits/test.pt")
# print("✅ Saved fixed splits: trainval.pt and test.pt")
# test_pairs = torch.load("splits/test.pt")

# **解包数据**
rgb_train, X_train, y_train = zip(*train_pairs)
rgb_val, X_val, y_val = zip(*val_pairs)
rgb_test, X_test, y_test = zip(*test_pairs)

# 转换为 Tensor
X_train, y_train = torch.stack(X_train), torch.tensor(y_train)
X_val, y_val = torch.stack(X_val), torch.tensor(y_val)
X_test, y_test = torch.stack(X_test), torch.tensor(y_test)

mean_hsi = X.mean(dim=(0, 2, 3))  # 计算通道维度的 均值
std_hsi = X.std(dim=(0, 2, 3))    # 计算通道维度的标准差
transform_hsi = transforms.Compose([
    transforms.Normalize(mean=mean_hsi.tolist(), std=std_hsi.tolist()),
    # AddSaltPepperNoise(salt_prob=0.01, pepper_prob=0.01)  # 添加噪声
])
transform_rgb = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0175, 0.0130, 0.0146], std=[0.0113, 0.0120, 0.0113]),
])


train_dataset = MultiModalDataset(rgb_train, X_train, y_train, transform_rgb, transform_hsi)
val_dataset = MultiModalDataset(rgb_val, X_val, y_val, transform_rgb, transform_hsi)
test_dataset = MultiModalDataset(rgb_test, X_test, y_test, transform_rgb, transform_hsi)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


# ==================== 验证数据集结构 ====================
print(f"\n最终数据集划分:")
print(f"训练集: {len(train_dataset)}个样本")
print(f"验证集: {len(val_dataset)}个样本") 
print(f"测试集: {len(test_dataset)}个样本（新数据）")

# 验证标签分布
print("\n类别分布:")
for name, loader in [("训练集", train_loader), ("验证集", val_loader), ("测试集", test_loader)]:
    labels = []
    for _, _, batch_labels in loader:
        labels.extend(batch_labels.tolist())
    print(f"{name}: {np.bincount(labels)}")


# ------------------------main------------------------------------------------


mymodels = {
    # 'MultiModalNet': (MultiModalNet,{ 'optimizer': ['Adam'], 'learning_rate': [0.001], 'weight_decay': [0.0001], 'fuse':[ 'concat'], 'dropout':[0.5], 'dim_head':[16],'heads':[4], 'warmup':[5], 'cos':[False], 'mid_dim':[256] ,'token_dim':[64] ,'mlp_dim':[512], 'act':['leaky_relu'] }),
    'SimpleMultiModalNet': (SimpleMultiModalNet,{ 'optimizer': ['Adam'], 'learning_rate': [0.001], 'weight_decay': [0.0001], 'fuse':[ 'concat'], 'dropout':[0.5], 'dim_head':[16],'heads':[4], 'warmup':[5], 'cos':[False], 'mid_dim':[256] ,'token_dim':[64] ,'mlp_dim':[512], 'act':['leaky_relu'] }),
    # 'MultiModalNet2': (MultiModalNet2,{ 'optimizer': ['Adam'], 'learning_rate': [0.001], 'weight_decay': [0.0001], 'fuse':[ 'concat'], 'dropout':[0.5], 'dim_head':[16],'heads':[4], 'warmup':[3], 'cos':[False], 'mid_dim':[256] ,'token_dim':[48] ,'mlp_dim':[512]}), 
#     'MultiModalNet1': (MultiModalNet1,{ 'optimizer': ['Adam'], 'learning_rate': [0.001], 'weight_decay': [0.0001], 'fuse':[ 'concat'], 'dropout':[0.1], 'dim_head':[32 ],'heads':[4] }),

}



# -----------------------------Model training and comparison------------------------


best_model = None
for model_name, (model_class, grid) in mymodels.items():
      for params in ParameterGrid(grid):
        logging.info(f"Training {model_name} with params: {params}")

        if model_name not in ['Line', 'KNN', 'SVM', 'ELM']:
            # 初始化模型
            # if torch.cuda.device_count() >= 2:
            #     logging.info(f"Using {torch.cuda.device_count()} GPUs!")
            # else:
            #     logging.info("Using single GPU or CPU")
                
            torch.cuda.empty_cache()
            repeats = 1
            all_run_metrics = []
            for run in range(1, 1+repeats):#(1, repeats + 1)
                logging.info(f"Starting run {run}/{repeats}...")
                set_seed(run)  # 每次运行使用不同种子
                
                activation = activation_dict[params['act'].lower()]
                model = model_class(fuse_method=params['fuse'], dropout=params['dropout'], heads=params['heads'], dim_head=params['dim_head'], token_dim=params['token_dim'], mlp_dim=params['mlp_dim'], activation=activation, c=c)
           
                model = model.to(device)
                # 使用DataParallel包装模型
                # if torch.cuda.device_count() >= 2:
                #     model = nn.DataParallel(model)
                
                model = model.to(device)
                criterion = nn.CrossEntropyLoss()
                optimizer = create_optimizer(
                    optimizer_name=params['optimizer'],
                    learning_rate=params['learning_rate'],
                    weight_decay=params.get('weight_decay', 0),
                )
                # 训练模型（使用修改后的train函数）
                result = train(
                    model=model,
                    model_name=model_name,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    optimizer=optimizer,
                    criterion=criterion,
                    device=device,
                    warmup_epochs=params.get('warmup_epochs', params['warmup']),
                    use_cosine_decay=params.get('cos', params['cos']),
                )
                best_model = result['model']
                history = result['history']
                # 测试集评估
                test_metrics = evaluate(best_model, test_loader, device=device)
                all_run_metrics.append(test_metrics)
                # 打印测试结果
                logging.info(f"Test Accuracy: {test_metrics['test_accuracy']:.4f} | "
                             f"Test Precision: {test_metrics['test_precision']:.4f} | "
                             f"Test Recall: {test_metrics['test_recall']:.4f} | "
                             f"Test F1 Score: {test_metrics['test_f1']:.4f}")

                # 计算统计指标
            metrics_summary = {}
            for metric in ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']:
                values = [run[metric] for run in all_run_metrics]
                metrics_summary[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'raw': values
                }

            # 保存结果到文件
            with open(f"store/{model_name}_results.txt", "a+") as f:
                f.write(f"\n修改:\nModel: {model_name}\n")
                f.write(f"Runs: {repeats}\n")

                # 写入测试集结果
                f.write("\nTest Metrics:\n")
                for metric in ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']:
                    mean = metrics_summary[metric]['mean']
                    std = metrics_summary[metric]['std']
                    f.write(f"{metric}: {mean:.4f} ± {std:.4f}\n")

                f.write("\nAll Test Accuracies:\n")
                test_accuracies = metrics_summary['test_accuracy']['raw']  # 获取所有test_accuracy值
                for i, acc in enumerate(test_accuracies, 1):  # 编号从1开始
                    f.write(f"Run {i}: {acc:.4f}\n")
                # # 写入原始数据
                # f.write("\nRaw Data:\n")
                # for i, run in enumerate(all_run_metrics, 1):
                #     f.write(f"Run {i}: " +
                #             ", ".join([f"{k}: {v:.4f}" for k, v in run.items()]) + "\n")
            del model, best_model, history
            torch.cuda.empty_cache()  # 清空缓存

            logging.info(f"Final results saved to {model_name}_results.txt")
        else:
            # 传统机器学习模型处理
            X_feature = feature_extraction(
                X,
                method=spectral_method,
                texture_method=texture_method,
                fusion_method=fusion_method,
                fusion_weight=0.5
            )

            # 自动调优模型
            model = auto_tune_model(spectral_method, model_class, param_grid, X_feature, y)
            metrics = train_evaluate_model(model, X_feature, y, n_splits=5, stratified=True)

            # 记录模型结果
            best_model = model
            best_metrics = metrics


