import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from PIL import Image
from Data_Loader import Data_Loader, MultiModalDataset, RGBLoader
from sklearn.decomposition import PCA
from sklearn.gaussian_process.kernels import Matern

import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import Matern, RBF, RationalQuadratic
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from collections import Counter
# ---------------------
# 数据读取与预处理
# ---------------------




def preprocess_data(X, y, test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split
    
    # 如果数据维度大于2，使用PCA降维
    if X.ndim > 2:
        print("原始数据维度:", X.shape)
        X = X.reshape(X.shape[0], -1)  # 展平高维数据
        print("展平后维度:", X.shape)
        
        # 使用PCA保留95%的方差
        pca = PCA(n_components=50000, random_state=random_state)
        X = pca.fit_transform(X)
        print("PCA降维后:", X.shape)
    
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # 标准化数据
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def balanced_sample(X, y, n_samples=600, random_state=42):
    """
    均衡采样函数，确保每个类别都有代表性样本
    
    参数:
        X: 特征数据 (n_samples, n_features)
        y: 标签数据 (n_samples,)
        n_samples: 总采样数
        random_state: 随机种子
    
    返回:
        X_balanced: 均衡采样后的特征数据
        y_balanced: 均衡采样后的标签数据
    """
    np.random.seed(random_state)
    
    # 获取类别信息
    unique_classes, class_counts = np.unique(y, return_counts=True)
    n_classes = len(unique_classes)
    print(f"原始数据类别个数: {n_classes}")
    
    # 计算每个类别的目标样本数
    samples_per_class = n_samples // n_classes
    remaining_samples = n_samples % n_classes
    
    sampled_indices = []
    
    # 对每个类别进行采样
    for i, cls in enumerate(unique_classes):
        # 获取当前类别的所有样本索引
        cls_indices = np.where(y == cls)[0]
        
        # 计算当前类别应该采样的数量
        n_cls_samples = samples_per_class + (1 if i < remaining_samples else 0)
        
        # 如果当前类别样本不足，则全部取用
        n_cls_samples = min(n_cls_samples, len(cls_indices))
        
        # 随机采样
        sampled_idx = np.random.choice(cls_indices, size=n_cls_samples, replace=False)
        sampled_indices.extend(sampled_idx)
    
    # 打乱样本顺序
    np.random.shuffle(sampled_indices)
    
    # 采样后类别统计
    y_balanced = y[sampled_indices]
    balanced_counts = Counter(y_balanced)
    print(f"采样后类别个数: {len(balanced_counts)}")
    print(f"采样后各类别样本数: {dict(balanced_counts)}")
    
    return X[sampled_indices], y_balanced

# 主程序
def main():
    SEED = 42
    data_loader = Data_Loader()
    data_loader.load_data()
    X, y = data_loader.get_data()
    n, h, w, c = X.shape

     # 预处理数据
    X_train_scaled, X_test_scaled, y_train, y_test = preprocess_data(X, y, random_state=SEED)
    
    # 检查处理后类别分布
    unique, counts = np.unique(y_train, return_counts=True)
    print("SMOTE处理后训练集类别分布:", dict(zip(unique, counts)))
    
    # 定义核函数集合
    kernels = {
        'Matern1.5': Matern(length_scale=1.0, nu=1.5),
        'Matern2.5': Matern(length_scale=1.0, nu=2.5),
        'RBF': RBF(length_scale=1.0),
        'RationalQuadratic': RationalQuadratic(length_scale=1.0, alpha=0.1)
    }
    
    best_score = 0
    best_model = None
    best_kernel = None
    
    # 测试不同核函数
    for name, kernel in kernels.items():
        print(f"\n=== 使用核函数: {name} ===")
        
        # 创建并训练模型
        gpc = GaussianProcessClassifier(kernel=kernel, random_state=SEED)
        gpc.fit(X_train_scaled, y_train)
        
        # 训练集评估
        y_pred_train = gpc.predict(X_train_scaled)
        print("\n训练集分类报告:")
        print(classification_report(y_train, y_pred_train))
        
        # 测试集评估
        y_pred = gpc.predict(X_test_scaled)
        print("测试集分类报告:")
        report = classification_report(y_test, y_pred, output_dict=True)
        print(classification_report(y_test, y_pred))
        
        # 计算加权平均f1分数
        weighted_f1 = report['weighted avg']['f1-score']
        
        # 更新最佳模型
        if weighted_f1 > best_score:
            best_score = weighted_f1
            best_model = gpc
            best_kernel = name
    
    # 输出最终结果
    print("\n=== 最佳模型 ===")
    print(f"最佳核函数: {best_kernel}")
    print(f"最佳加权F1分数: {best_score:.4f}")
    
    # 最佳模型混淆矩阵
    print("\n最佳模型混淆矩阵:")
    y_pred_best = best_model.predict(X_test_scaled)
    print(confusion_matrix(y_test, y_pred_best))
    
    # 输出各类别详细预测情况
    print("\n各类别预测详情:")
    for cls in np.unique(y_test):
        cls_idx = np.where(y_test == cls)[0]
        accuracy = np.mean(y_pred_best[cls_idx] == y_test[cls_idx])
        print(f"类别 {cls}: 准确率 = {accuracy:.2%}")
if __name__ == "__main__":
    main()
