# -*- coding: utf-8 -*-
import os
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
import time
from model import *
import torch
import seaborn as sns
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, TensorDataset
import logging
import copy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
import torch.optim as optim
from copy import deepcopy
from torch.cuda.amp import GradScaler, autocast
import math
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, ReduceLROnPlateau
import logging
from datetime import datetime
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, roc_curve, auc
import json
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, roc_curve, auc
# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
spa_col = [ 59,  92,  97,  95,  49,  28, 182,  68,  44,   8, 111, 168, 305,
    3, 140, 300,  55, 288, 149,  82, 278, 307, 312, 283, 101, 158,
    0,  21, 266, 129, 226]

def visualize_train_tsne(features, labels, epoch, save_path, title=None):
    """支持多特征类型的训练集可视化"""
    try:
        # 1. 维度验证
        assert len(features) == len(labels), f"特征数({len(features)})≠标签数({len(labels)})"
        
        # 2. t-SNE降维 (自动处理NaN/Inf)
        features = np.nan_to_num(features)  # 处理异常值
        tsne = TSNE(n_components=2, perplexity=min(30, len(features)-1))
        features_2d = tsne.fit_transform(features)
        
        # 3. 绘图
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            features_2d[:, 0], features_2d[:, 1],
            c=labels, cmap='tab20', alpha=0.6, s=30,
            edgecolors='w', linewidths=0.3
        )
        
        # 4. 添加颜色条
        cbar = plt.colorbar(scatter)
        cbar.set_label('Class')
        
        # 5. 标题和保存
        plt.title(title or f"Train t-SNE (Epoch {epoch})")
        plt.grid(alpha=0.2)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_file = f"{save_path}_epoch{epoch}.png"
        plt.savefig(save_file, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"成功保存: {os.path.abspath(save_file)}")
        
    except Exception as e:
        print(f" 可视化失败: {str(e)}")
        if 'plt' in locals():
            plt.close()
import time
from copy import deepcopy
import torch

from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, roc_curve, auc

def train(
        model,
        model_name,
        train_loader,
        val_loader,
        optimizer,
        num_epochs=100,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        criterion=None,
        save_path='store',
        max_grad_norm=1.0,
        warmup_epochs=3,
        use_cosine_decay=False,
        run=1
):
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'train_binary_acc': [],  # 新增：二分类准确率
        'val_binary_acc': []     # 新增：二分类准确率
    }
    best_model = None
    best_model_state = None
    best_loss = float('inf')
    
    # 初始化学习率调度器
    warmup_scheduler, cosine_scheduler, reduce_lr_scheduler = None, None, None
    if warmup_epochs > 0:
        warmup_scheduler = LambdaLR(
            optimizer, 
            lambda e: min(1.0, (e + 1) / warmup_epochs)
        )
    
    if use_cosine_decay:
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_epochs - warmup_epochs,
            eta_min=1e-6
        )
    else:
        reduce_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=3, verbose=True
        )
    
    scaler = torch.amp.GradScaler(enabled=(device == 'cuda'))

    for epoch in range(num_epochs):
        epoch_start = time.time()
        # ========== 训练阶段 ==========
        
        model.train()
        train_loss, train_correct, total_samples = 0.0, 0, 0
        train_binary_correct = 0  # 新增：二分类正确计数
        
        ct=0
        add = 0
        for rgb, hsi, labels in train_loader:
            t0=time.time()
            rgb, hsi, labels = rgb.to(device), hsi.to(device), labels.to(device)
            
            
            with torch.amp.autocast(device_type='cuda', enabled=True):
                outputs = model(rgb, hsi)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
           
            train_loss += loss.item() * labels.size(0)
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == labels).sum().item()
            # 新增：二分类预测（健康 vs 病害）
            binary_labels = (labels > 0).long()  # 0=健康, 1/2/3=病害 -> 1
            binary_preds = (preds > 0).long()    # 0=健康, 1/2/3=病害 -> 1
            train_binary_correct += (binary_preds == binary_labels).sum().item()
            
            total_samples += labels.size(0)
            t1=time.time()-t0
            ct+=1
            # print(f"{ct}:  {t1:.2f} ")
            add+=t1
             
        t_train = time.time()-epoch_start
          

        train_loss /= total_samples
        train_acc = train_correct / total_samples
        train_binary_acc = train_binary_correct / total_samples  # 二分类准确率
        # ========== 验证阶段（加条件判断） ==========
        t_val = 0
        t0 = time.time()
        if val_loader is not None:
            model.eval()
            val_loss, val_correct, total_val = 0.0, 0, 0
            val_binary_correct = 0  # 新增：二分类正确计数

            if epoch in [50]:
                val_feats_rgb, val_feats_hsi, val_feats_cat = [], [], []
                val_labels = []

            with torch.no_grad():
                for rgb, hsi, labels in val_loader:
                    rgb, hsi, labels = rgb.to(device), hsi.to(device), labels.to(device)
                    outputs = model(rgb, hsi)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * labels.size(0)
                    preds = torch.argmax(outputs, dim=1)
                    val_correct += (preds == labels).sum().item()
                    # 新增：二分类预测
                    binary_labels = (labels > 0).long()
                    binary_preds = (preds > 0).long()
                    val_binary_correct += (binary_preds == binary_labels).sum().item()
                    
                    total_val += labels.size(0)

              
                

            val_loss /= total_val
            val_acc = val_correct / total_val
            val_binary_acc = val_binary_correct / total_val
        else:
            val_loss = float('nan')
            val_acc = float('nan')
            val_binary_acc = float('nan')
       
        # ========== 学习率调整 ==========
        if warmup_scheduler and epoch < warmup_epochs:
            warmup_scheduler.step()
        elif epoch >= warmup_epochs:
            if use_cosine_decay:
                cosine_scheduler.step()
            elif val_loader is not None:
                reduce_lr_scheduler.step(val_loss)

        # ========== 记录和保存 ==========
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_binary_acc'].append(train_binary_acc)  # 记录二分类准确率
        history['val_binary_acc'].append(val_binary_acc)      # 记录二分类准确率
        

        if val_loader is not None and val_loss < best_loss:
            best_loss = val_loss
            best_model_state = deepcopy(model.state_dict())
       
        logging.info(
            f"Epoch {epoch + 1}: "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
            f"Train Binary Acc: {train_binary_acc:.4f} | Val Binary Acc: {val_binary_acc:.4f}")  # 新增二分类日志

 
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return {
        'model': model,
        'history': history,
    }

# def evaluate(model, test_loader, class_names=None, save_dir='results', device=None):
#     """增强的测试集评估函数（所有指标添加test_前缀），支持多分类和二分类"""
    
#     # 设备处理
#     if device is None:
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     # 创建保存目录
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     save_path = os.path.join(save_dir, f"eval_{timestamp}")
#     os.makedirs(save_path, exist_ok=True)
    
#     # 初始化
#     model = model.to(device)
#     model.eval()
#     all_preds, all_labels = [], []
#     all_probs = []
    
#     # 推理
#     with torch.no_grad():
#         for rgb, hsi, labels in test_loader:
#             rgb, hsi, labels = rgb.to(device), hsi.to(device), labels.to(device)
#             outputs = model(rgb, hsi)
#             probs = torch.softmax(outputs.cpu(), dim=1)
#             preds = torch.argmax(probs, dim=1)
#             all_preds.extend(preds.numpy())
#             all_labels.extend(labels.cpu().numpy())
#             all_probs.extend(probs.numpy())
    
#     # 转换为numpy
#     all_labels = np.array(all_labels)
#     all_preds = np.array(all_preds)
#     all_probs = np.array(all_probs)
    
#     # 生成二分类标签和预测（0=健康，1=病害）
#     binary_labels = (all_labels > 0).astype(int)  # 0->0, 1/2/3->1
#     binary_preds = (all_preds > 0).astype(int)    # 0->0, 1/2/3->1
    
#     # 处理类别名称
#     if class_names is None:
#         class_names = [str(i) for i in range(np.max(all_labels)+1)] if len(all_labels) > 0 else []
    
#     # 二分类类别名称
#     binary_class_names = ['healthy', 'diseased']
    
#     # 多分类评估指标
#     test_metrics = {
#         'test_accuracy': accuracy_score(all_labels, all_preds),
#         'test_precision': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
#         'test_recall': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
#         'test_f1': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
#         'class_names': class_names,
        
#         # 二分类评估指标
#         'test_binary_accuracy': accuracy_score(binary_labels, binary_preds),
#         'test_binary_precision': precision_score(binary_labels, binary_preds, average='binary', zero_division=0),
#         'test_binary_recall': recall_score(binary_labels, binary_preds, average='binary', zero_division=0),
#         'test_binary_f1': f1_score(binary_labels, binary_preds, average='binary', zero_division=0),
#         'binary_class_names': binary_class_names
#     }
    
#     # 多分类分类报告
#     multi_report = classification_report(
#         all_labels, all_preds, 
#         target_names=class_names, 
#         output_dict=True
#     )
#     pd.DataFrame(multi_report).transpose().to_csv(os.path.join(save_path, 'multiclass_classification_report.csv'))
#     test_metrics['multiclass_classification_report'] = multi_report
    
#     # 二分类分类报告
#     binary_report = classification_report(
#         binary_labels, binary_preds, 
#         target_names=binary_class_names, 
#         output_dict=True
#     )
#     pd.DataFrame(binary_report).transpose().to_csv(os.path.join(save_path, 'binary_classification_report.csv'))
#     test_metrics['binary_classification_report'] = binary_report
    
#     # 多分类混淆矩阵
#     multi_cm = confusion_matrix(all_labels, all_preds)
#     test_metrics['multiclass_confusion_matrix'] = multi_cm.tolist()
    
#     plt.figure(figsize=(10, 8))
#     df_multi_cm = pd.DataFrame(multi_cm, index=class_names, columns=class_names)
#     sns.heatmap(df_multi_cm, annot=True, fmt='d', cmap='Blues')
#     plt.title('Multiclass Confusion Matrix')
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
#     plt.savefig(os.path.join(save_path, 'multiclass_confusion_matrix.png'))
#     plt.close()
#     pd.DataFrame(multi_cm, index=class_names, columns=class_names).to_csv(
#         os.path.join(save_path, 'multiclass_confusion_matrix.csv'))
    
#     # 二分类混淆矩阵
#     binary_cm = confusion_matrix(binary_labels, binary_preds)
#     test_metrics['binary_confusion_matrix'] = binary_cm.tolist()
    
#     plt.figure(figsize=(8, 6))
#     df_binary_cm = pd.DataFrame(binary_cm, index=binary_class_names, columns=binary_class_names)
#     sns.heatmap(df_binary_cm, annot=True, fmt='d', cmap='Blues')
#     plt.title('Binary Classification Confusion Matrix')
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
#     plt.savefig(os.path.join(save_path, 'binary_confusion_matrix.png'))
#     plt.close()
#     pd.DataFrame(binary_cm, index=binary_class_names, columns=binary_class_names).to_csv(
#         os.path.join(save_path, 'binary_confusion_matrix.csv'))
    
#     # 保存预测结果（包含多分类和二分类）
#     results_df = pd.DataFrame({
#         'true_label': all_labels,
#         'predicted_label': all_preds,
#         'true_binary_label': binary_labels,
#         'predicted_binary_label': binary_preds,
#         **{f'prob_{name}': all_probs[:, i] for i, name in enumerate(class_names)}
#     })
#     results_df.to_csv(os.path.join(save_path, 'test_predictions.csv'), index=False)
    
#     # 保存所有指标
#     with open(os.path.join(save_path, 'test_metrics.json'), 'w') as f:
#         json.dump(test_metrics, f, indent=4)
    
#     # 打印结果摘要
#     print(f"评估结果已保存至: {save_path}")
#     print(f"多分类准确率: {test_metrics['test_accuracy']:.4f}")
#     print(f"二分类准确率: {test_metrics['test_binary_accuracy']:.4f}")
#     print(f"二分类F1分数: {test_metrics['test_binary_f1']:.4f}")
    
#     return test_metrics

import os
import json
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,  # 新增
    classification_report,
    confusion_matrix
)
from sklearn.preprocessing import label_binarize # 新增

def evaluate(model, test_loader, class_names=None, save_dir='results', device=None):
    """
    增强的测试集评估函数，计算多分类和二分类的全面指标。
    """
    
    # --- 1. 初始化和准备 (与你原代码相同) ---
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"eval_{timestamp}")
    os.makedirs(save_path, exist_ok=True)
    
    model = model.to(device)
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for rgb, hsi, labels in test_loader:
            rgb, hsi, labels = rgb.to(device), hsi.to(device), labels.to(device)
            outputs = model(rgb, hsi)
            # 使用 .cpu() 来避免GPU同步问题
            probs = torch.softmax(outputs, dim=1).cpu()
            preds = torch.argmax(probs, dim=1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.numpy())
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # 处理类别名称
    if class_names is None:
        # 确保class_names的顺序与标签0, 1, 2, ...对应
        unique_labels = np.unique(all_labels)
        class_names = [str(i) for i in range(np.max(unique_labels) + 1)] if len(unique_labels) > 0 else []
    
    # --- 2. 派生二分类数据 ---
    binary_labels = (all_labels > 0).astype(int)
    binary_preds = (all_preds > 0).astype(int)
    # 二分类的概率是所有患病类别的概率之和
    binary_probs_diseased = all_probs[:, 1:].sum(axis=1) 
    binary_class_names = ['healthy', 'diseased']

    # --- 3. 计算所有指标 ---
    
    # 初始化指标字典
    test_metrics = {}

    # --- 3.1 多分类指标 ---
    num_classes = len(class_names)
    test_metrics['test_accuracy'] = accuracy_score(all_labels, all_preds)
    test_metrics['test_precision_weighted'] = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    test_metrics['test_recall_weighted'] = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    test_metrics['test_f1_weighted'] = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    # **新增: 多分类AUROC (OvR, macro average)**
    # 确保有多个类别且预测概率维度正确
    if num_classes > 1 and all_probs.shape[1] == num_classes:
        # 将标签二值化
        y_true_binarized = label_binarize(all_labels, classes=range(num_classes))
        test_metrics['test_auroc_ovr_macro'] = roc_auc_score(y_true_binarized, all_probs, average='macro', multi_class='ovr')
    else:
        test_metrics['test_auroc_ovr_macro'] = float('nan')

    # **新增: 每个类别的准确率 (即召回率)**
    multi_report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0)
    for class_name in class_names:
        key = f'test_acc_{class_name.replace(" ", "_").lower()}' # e.g., 'test_acc_healthy', 'test_acc_2_dpi'
        test_metrics[key] = multi_report_dict[class_name]['recall']
    
    # --- 3.2 二分类指标 ---
    test_metrics['test_binary_accuracy'] = accuracy_score(binary_labels, binary_preds)
    test_metrics['test_binary_precision'] = precision_score(binary_labels, binary_preds, zero_division=0)
    test_metrics['test_binary_recall'] = recall_score(binary_labels, binary_preds, zero_division=0)
    test_metrics['test_binary_f1'] = f1_score(binary_labels, binary_preds, zero_division=0)

    # **新增: 二分类AUROC**
    # 确保存在正负两类
    if len(np.unique(binary_labels)) > 1:
        test_metrics['test_binary_auroc'] = roc_auc_score(binary_labels, binary_probs_diseased)
    else:
        test_metrics['test_binary_auroc'] = float('nan')

    # --- 4. 生成报告、混淆矩阵和文件保存 (大部分与你原代码相同) ---
    
    # 分类报告
    pd.DataFrame(multi_report_dict).transpose().to_csv(os.path.join(save_path, 'multiclass_classification_report.csv'))
    test_metrics['multiclass_classification_report'] = multi_report_dict

    binary_report_dict = classification_report(binary_labels, binary_preds, target_names=binary_class_names, output_dict=True, zero_division=0)
    pd.DataFrame(binary_report_dict).transpose().to_csv(os.path.join(save_path, 'binary_classification_report.csv'))
    test_metrics['binary_classification_report'] = binary_report_dict
    
    # 混淆矩阵
    multi_cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(pd.DataFrame(multi_cm, index=class_names, columns=class_names), annot=True, fmt='d', cmap='Blues')
    plt.title('Multiclass Confusion Matrix'); plt.xlabel('Predicted'); plt.ylabel('True')
    plt.savefig(os.path.join(save_path, 'multiclass_confusion_matrix.png'))
    plt.close()
    
    binary_cm = confusion_matrix(binary_labels, binary_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(pd.DataFrame(binary_cm, index=binary_class_names, columns=binary_class_names), annot=True, fmt='d', cmap='Blues')
    plt.title('Binary Classification Confusion Matrix'); plt.xlabel('Predicted'); plt.ylabel('True')
    plt.savefig(os.path.join(save_path, 'binary_confusion_matrix.png'))
    plt.close()
    
    # 保存预测详情
    results_df = pd.DataFrame({
        'true_label': all_labels,
        'predicted_label': all_preds,
        'true_binary_label': binary_labels,
        'predicted_binary_label': binary_preds,
        'prob_diseased': binary_probs_diseased,
        **{f'prob_class_{i}': all_probs[:, i] for i in range(num_classes)}
    })
    results_df.to_csv(os.path.join(save_path, 'test_predictions.csv'), index=False)
    
    # 保存所有指标
    # 使用自定义函数处理Numpy类型，使其可以被JSON序列化
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
        
    with open(os.path.join(save_path, 'test_metrics.json'), 'w') as f:
        json.dump(test_metrics, f, indent=4, default=convert_numpy)
    
    # --- 5. 打印结果摘要 ---
    print(f"\n--- Evaluation Results (saved to {save_path}) ---")
    print("\n[Multiclass Metrics]")
    print(f"  Accuracy: {test_metrics['test_accuracy']:.4f}")
    print(f"  AUROC (Macro OvR): {test_metrics.get('test_auroc_ovr_macro', float('nan')):.4f}")
    print(f"  F1 (Weighted): {test_metrics['test_f1_weighted']:.4f}")
    print("  Per-class Accuracy (Recall):")
    for name in class_names:
        key = f'test_acc_{name.replace(" ", "_").lower()}'
        print(f"    - {name}: {test_metrics[key]:.4f}")

    print("\n[Binary Metrics (Healthy vs Diseased)]")
    print(f"  Accuracy: {test_metrics['test_binary_accuracy']:.4f}")
    print(f"  AUROC: {test_metrics.get('test_binary_auroc', float('nan')):.4f}")
    print(f"  Precision: {test_metrics['test_binary_precision']:.4f}")
    print(f"  Recall: {test_metrics['test_binary_recall']:.4f}")
    print(f"  F1-score: {test_metrics['test_binary_f1']:.4f}")
    
    return test_metrics
# def evaluate(model, test_loader, class_names=None, save_dir='results', device=None):
#     """增强的测试集评估函数（所有指标添加test_前缀），支持多分类和二分类，额外记录错误样本"""
    
#     # 设备处理
#     if device is None:
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     # 创建保存目录
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     save_path = os.path.join(save_dir, f"eval_{timestamp}")
#     os.makedirs(save_path, exist_ok=True)
    
#     # 初始化
#     model = model.to(device)
#     model.eval()
#     all_preds, all_labels = [], []
#     all_probs, all_indices = [], []
    
#     # 推理
#     with torch.no_grad():
#         for batch_idx, (rgb, hsi, labels) in enumerate(test_loader):
#             rgb, hsi, labels = rgb.to(device), hsi.to(device), labels.to(device)
#             outputs = model(rgb, hsi)
#             probs = torch.softmax(outputs.cpu(), dim=1)
#             preds = torch.argmax(probs, dim=1)

#             all_preds.extend(preds.numpy())
#             all_labels.extend(labels.cpu().numpy())
#             all_probs.extend(probs.numpy())

#             # 保存样本在原始数据集中的索引
#             if hasattr(test_loader.dataset, "indices"):  
#                 # 兼容Subset
#                 start = batch_idx * test_loader.batch_size
#                 end = start + len(labels)
#                 batch_indices = test_loader.dataset.indices[start:end]
#             else:
#                 # 直接顺序索引
#                 start = batch_idx * test_loader.batch_size
#                 end = start + len(labels)
#                 batch_indices = list(range(start, end))
#             all_indices.extend(batch_indices)
    
#     # 转换为numpy
#     all_labels = np.array(all_labels)
#     all_preds = np.array(all_preds)
#     all_probs = np.array(all_probs)
#     all_indices = np.array(all_indices)
    
#     # 生成二分类标签和预测（0=健康，1=病害）
#     binary_labels = (all_labels > 0).astype(int)  
#     binary_preds = (all_preds > 0).astype(int)    
    
#     # 处理类别名称
#     if class_names is None:
#         class_names = [str(i) for i in range(np.max(all_labels)+1)] if len(all_labels) > 0 else []
#     binary_class_names = ['healthy', 'diseased']
    
#     # 多分类评估指标
#     test_metrics = {
#         'test_accuracy': accuracy_score(all_labels, all_preds),
#         'test_precision': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
#         'test_recall': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
#         'test_f1': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
#         'class_names': class_names,
        
#         # 二分类评估指标
#         'test_binary_accuracy': accuracy_score(binary_labels, binary_preds),
#         'test_binary_precision': precision_score(binary_labels, binary_preds, average='binary', zero_division=0),
#         'test_binary_recall': recall_score(binary_labels, binary_preds, average='binary', zero_division=0),
#         'test_binary_f1': f1_score(binary_labels, binary_preds, average='binary', zero_division=0),
#         'binary_class_names': binary_class_names
#     }
    
#     # 分类报告 & 混淆矩阵保存（略，保持你原来的代码）
#     # ------------------------------------------------------
#     # ... 这里不变，和你之前一样 ...
#     # ------------------------------------------------------
    
#     # 保存预测结果（额外加上是否错误）
#     misclassified = (all_preds != all_labels).astype(int)
#     results_df = pd.DataFrame({
#         'sample_index': all_indices,
#         'true_label': all_labels,
#         'predicted_label': all_preds,
#         'true_binary_label': binary_labels,
#         'predicted_binary_label': binary_preds,
#         'misclassified': misclassified,
#         **{f'prob_{name}': all_probs[:, i] for i, name in enumerate(class_names)}
#     })
#     results_df.to_csv(os.path.join(save_path, 'test_predictions.csv'), index=False)
    
#     # 找出总是错的样本
#     always_wrong = results_df[results_df['misclassified'] == 1]['sample_index'].tolist()
#     with open(os.path.join(save_path, 'always_wrong_samples.txt'), 'w') as f:
#         for idx in always_wrong:
#             f.write(str(idx) + "\n")
#     test_metrics['always_wrong_samples'] = always_wrong
    
#     # 保存所有指标
#     with open(os.path.join(save_path, 'test_metrics.json'), 'w') as f:
#         json.dump(test_metrics, f, indent=4)
    
#     # 打印结果摘要
#     print(f"评估结果已保存至: {save_path}")
#     print(f"多分类准确率: {test_metrics['test_accuracy']:.4f}")
#     print(f"二分类准确率: {test_metrics['test_binary_accuracy']:.4f}")
#     print(f"二分类F1分数: {test_metrics['test_binary_f1']:.4f}")
#     print(f"总是被分错的样本数: {len(always_wrong)}")
    
#     return test_metrics
