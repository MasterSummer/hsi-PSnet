import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from scipy.stats import wasserstein_distance
from PIL import Image
import seaborn as sns
from tqdm import tqdm
from Data_Loader import Data_Loader, MultiModalDataset, RGBLoader

# 1. 第一批数据加载
print("加载第一批数据...")
data_loader = Data_Loader()
data_loader.load_data()
X, y = data_loader.get_data()
X = torch.tensor(X, dtype=torch.float32).permute(0, 3, 1, 2)  # (n, c, h, w)
y = torch.tensor(y, dtype=torch.long)
print(f"run1 num: {X.shape[0]}") 
rgb_loader = RGBLoader(data_root="RGB")
rgb_image_paths, rgb_labels = rgb_loader.load_paths()

# 2. 第二批数据加载
print("加载第二批数据...")
data_loader_new = Data_Loader()
data_loader_new.load_data(hsi_root="run2")
X_new, y_new = data_loader_new.get_data()
X_new = torch.tensor(X_new, dtype=torch.float32).permute(0, 3, 1, 2)
y_new = torch.tensor(y_new, dtype=torch.long)
print(f"run2 num: {X_new.shape[0]}") 
rgb_loader_new = RGBLoader(data_root="RGB2")
rgb_image_paths_new, rgb_labels_new = rgb_loader_new.load_paths()
print(f"run2 rbg num: {len(rgb_image_paths)}") 
# 3. 基础验证
def basic_validation():
    print("\n=== 基础验证 ===")
    # 第一批检查
    assert len(X) == len(rgb_image_paths), "第一批HSI-RGB数量不匹配"
    assert torch.all(y == torch.tensor(rgb_labels)), "第一批标签不一致"
    
    # 第二批检查
    assert len(X_new) == len(rgb_image_paths_new), "第二批HSI-RGB数量不匹配"
    assert torch.all(y_new == torch.tensor(rgb_labels_new)), "第二批标签不一致"
    
    # 维度检查
    assert X.shape[1:] == X_new.shape[1:], f"数据形状不匹配: 第一批{X.shape[1:]}, 第二批{X_new.shape[1:]}"

    print("基础验证通过")

basic_validation()

# 4. 光谱特征分析
def spectral_analysis():
    
    plt.figure(figsize=(15, 6))
    
    # 第一批光谱
    plt.subplot(1, 2, 1)
    for cls in torch.unique(y):
        idx = (y == cls).nonzero()[:5]  # 每类取5个样本
        specs = X[idx].mean(dim=(0, 2, 3))  # (5, c)
        for spec in specs:
            plt.plot(spec, label=f'Class {cls}', alpha=0.5)
    plt.title("第一批光谱特征")
    plt.xlabel("波段")
    plt.ylabel("反射率")
    plt.legend()
    
    # 第二批光谱
    plt.subplot(1, 2, 2)
    for cls in torch.unique(y_new):
        idx = (y_new == cls).nonzero()[:5]
        specs = X_new[idx].mean(dim=(0, 2, 3))
        for spec in specs:
            plt.plot(spec, label=f'Class {cls}', alpha=0.5)
    plt.title("第二批光谱特征")
    plt.xlabel("波段")
    
    plt.tight_layout()
    plt.savefig('spectral_comparison.png')
    plt.close()
    print(" 光谱分析完成 - 结果保存到 spectral_comparison.png")

spectral_analysis()

# 5. 空间分布检测
def spatial_distribution():
    # 随机采样1000个点（避免内存不足）
    n_samples = min(1000, len(X), len(X_new))
    idx1 = np.random.choice(len(X), n_samples, replace=False)
    idx2 = np.random.choice(len(X_new), n_samples, replace=False)
    
    # 展平数据
    X1_flat = X[idx1].flatten(start_dim=1).numpy()
    X2_flat = X_new[idx2].flatten(start_dim=1).numpy()
    
    # PCA分析
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(np.vstack([X1_flat, X2_flat]))
    
    # 可视化
    plt.figure(figsize=(12, 6))
    plt.scatter(X_pca[:n_samples, 0], X_pca[:n_samples, 1], 
                alpha=0.3, label='第一批')
    plt.scatter(X_pca[n_samples:, 0], X_pca[n_samples:, 1], 
                alpha=0.3, label='第二批')
    plt.title("PCA特征空间分布")
    plt.xlabel("主成分1 (解释方差: {:.1f}%)".format(pca.explained_variance_ratio_[0]*100))
    plt.ylabel("主成分2 (解释方差: {:.1f}%)".format(pca.explained_variance_ratio_[1]*100))
    plt.legend()
    plt.savefig('pca_distribution.png')
    plt.close()
    
    # 计算分布距离
    w_dist = wasserstein_distance(X1_flat.mean(axis=0), X2_flat.mean(axis=0))
    print(f" Wasserstein: {w_dist:.4f}")
    print(f"   >0.3表示显著差异, >0.5表示严重差异")

spatial_distribution()

# 6. RGB图像质量检查
def rgb_inspection():
    os.makedirs('rgb_samples', exist_ok=True)
    
    # 随机选择3对样本
    pairs = []
    for _ in range(3):
        idx1 = np.random.randint(len(rgb_image_paths))
        idx2 = np.random.randint(len(rgb_image_paths_new))
        pairs.append((rgb_image_paths[idx1], rgb_image_paths_new[idx2]))
    
    # 可视化对比
    plt.figure(figsize=(15, 8))
    for i, (path1, path2) in enumerate(pairs):
        img1 = Image.open(path1)
        img2 = Image.open(path2)
        
        plt.subplot(3, 2, 2*i+1)
        plt.imshow(img1)
        plt.title(f"第一批\n{os.path.basename(path1)}")
        plt.axis('off')
        
        plt.subplot(3, 2, 2*i+2)
        plt.imshow(img2)
        plt.title(f"第二批\n{os.path.basename(path2)}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('rgb_comparison.png')
    plt.close()
    print("RGB检查完成 - 结果保存到 rgb_comparison.png")

rgb_inspection()

# 7. 自动问题检测
def auto_detect():
    issues = []
    
    # 检查数据范围
    range_diff = abs(X.mean() - X_new.mean()) / X.mean()
    if range_diff > 0.3:
        issues.append(f"数据范围差异过大(第一批均值{X.mean():.2f}, 第二批{X_new.mean():.2f})")
    
    # 检查标签分布
    if set(y.numpy()) != set(y_new.numpy()):
        issues.append(f"标签类别不一致(第一批{set(y.numpy())}, 第二批{set(y_new.numpy())})")
    
    # 检查RGB尺寸
    sample_img = Image.open(rgb_image_paths[0])
    sample_img_new = Image.open(rgb_image_paths_new[0])
    if sample_img.size != sample_img_new.size:
        issues.append(f"RGB尺寸不同(第一批{sample_img.size}, 第二批{sample_img_new.size})")
    
    if issues:
        print("\n⚠️ 检测到潜在问题:")
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")
    else:
        print("\n✅ 未检测到明显数据结构问题")

auto_detect()

# 8. 生成综合报告
print("\n=== 诊断报告总结 ===")
print("1. 光谱特征对比图: spectral_comparison.png")
print("2. 空间分布PCA图: pca_distribution.png")
print("3. RGB样本对比图: rgb_comparison.png")
print("\n建议下一步操作:")
print("- 如果PCA图中两批数据分离明显，需采用域适应技术")
print("- 如果光谱曲线形状差异大，检查传感器校准")
print("- 如果RGB图像存在色差，需做色彩校正")
