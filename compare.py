import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
from Data_Loader import Data_Loader, RGBLoader

# 1. 创建输出目录
output_dir = "comparison_results"
os.makedirs(output_dir, exist_ok=True)

def save_figure(fig, filename):
    """保存图形到文件并显示"""
    path = os.path.join(output_dir, filename)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.show()
    plt.close(fig)
    print(f"图形已保存至: {path}")

def select_bands_by_class_difference(X, y, classes=[0, 1, 2, 3], class_names=None, top_k=100, band_step=20):
    """
    根据类别间平均光谱差异选取最具区分度的波段
    X: [n, c, h, w] 或 [n, c]（HSI 数据）
    y: [n]（类别标签）
    classes: 要对比的类别列表
    class_names: 类别名称（可选）
    top_k: 返回区分度最高的前k个波段
    band_step: 横轴（波段）显示的步长
    """
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    
    if class_names is None:
        class_names = [f"Class {cls}" for cls in classes]

    # 计算每个类别的平均光谱 [n_classes, n_bands]
    class_means = []
    for cls in classes:
        cls_pixels = X[y == cls].reshape(-1, X.shape[-1])  # 展平成 [n_pixels, n_bands]
        cls_mean = cls_pixels.mean(axis=0)
        class_means.append(cls_mean)
    class_means = np.array(class_means)  # [n_classes, n_bands]

    # 计算所有类别两两之间的差异 [n_pairs, n_bands]
    n_classes = len(classes)
    pairwise_diff = []
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            diff = np.abs(class_means[i] - class_means[j])  # 差异绝对值
            pairwise_diff.append(diff)
    pairwise_diff = np.array(pairwise_diff)  # [n_pairs, n_bands]

    # 波段重要性评分（取所有类别对差异的最大值）
    band_scores = pairwise_diff.max(axis=0)  # [n_bands]
    top_bands = np.argsort(band_scores)[-top_k:][::-1]  # 降序排列

    # 打印关键波段
    print(f"Top {top_k} discriminative bands (sorted by importance):")
    for band in top_bands:
        print(f"Band {band}: Score = {band_scores[band]:.4f}")

    # 可视化波段差异曲线
    bands = np.arange(X.shape[-1])
    fig = plt.figure(figsize=(12, 6))
    plt.plot(bands, band_scores, 'b-', label='Class Difference Score')
    
    # 标注前top_k波段
    for band in top_bands:
        plt.axvline(x=band, color='red', linestyle='--', alpha=0.3)
        plt.text(band, band_scores[band] * 0.9, f"Band {band}", rotation=90, ha='center', va='top')

    plt.xlabel("Spectral Band")
    plt.ylabel("Max Class Difference Score")
    plt.title("Band Importance Based on Class Differences")
    plt.xticks(np.arange(0, len(bands), band_step))
    plt.legend()
    plt.grid(True)
    
    save_figure(fig, "band_selection_by_class_difference.png")
    return top_bands, band_scores

def compare_four_pixelwise_classes(X, y, classes=[0, 1, 2, 3], class_names=None, band_step=20, output_prefix=None):
    """
    逐像素地对比四个类别的平均光谱和标准差，并分别绘制均值图和标准差图
    X: torch.Tensor [n, c, h, w]
    y: torch.Tensor 或 np.ndarray [n]
    classes: 要对比的四个类别编号
    class_names: 类别名称列表（可选）
    band_step: 横轴（波段）显示的步长
    output_prefix: 输出文件名前缀（可选）
    """
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()

    if class_names is None:
        class_names = [f"Class {cls}" for cls in classes]

    # 颜色列表（确保四个类别颜色不同）
    colors = ['blue', 'red', 'green', 'purple']

    # 计算每个类别的像素级均值和标准差
    class_means = []
    class_stds = []

    for cls in classes:
        # 取出该类别的所有像素 [n_pixels, c]
        cls_imgs = X[y == cls]  # [n1, c, h, w]
        cls_pixels = cls_imgs.transpose(0, 2, 3, 1).reshape(-1, X.shape[1])
        
        cls_mean = cls_pixels.mean(axis=0)
        cls_std = cls_pixels.std(axis=0)
        
        class_means.append(cls_mean)
        class_stds.append(cls_std)

    bands = np.arange(X.shape[1])

    # ========== 1. 绘制均值对比图 ==========
    fig_mean = plt.figure(figsize=(12, 6))
    for i, cls in enumerate(classes):
        plt.plot(bands, class_means[i], label=f'{class_names[i]} Mean', color=colors[i], linestyle='-')

    plt.xlabel("Spectral Band")
    plt.ylabel("Reflectance")
    plt.title("run2_Pixel-wise Mean Spectral Comparison")
    plt.xticks(np.arange(0, len(bands), band_step))
    plt.legend()
    plt.grid(True)

    fname_mean = f"run2_{output_prefix}_mean_comparison.png" if output_prefix else "run2_four_classes_mean_comparison.png"
    save_figure(fig_mean, fname_mean)

    # ========== 2. 绘制标准差对比图 ==========
    fig_std = plt.figure(figsize=(12, 6))
    for i, cls in enumerate(classes):
        plt.plot(bands, class_stds[i], label=f'{class_names[i]} Std', color=colors[i], linestyle='--')

    plt.xlabel("Spectral Band")
    plt.ylabel("Standard Deviation")
    plt.title("run2_Pixel-wise Standard Deviation Comparison")
    plt.xticks(np.arange(0, len(bands), band_step))
    plt.legend()
    plt.grid(True)

    fname_std = f"run2_{output_prefix}_std_comparison.png" if output_prefix else "run2_four_classes_std_comparison.png"
    save_figure(fig_std, fname_std)
    
# ---------------------- HSI整体对比 ---------------------
def compare_hsi_batches(batch1, batch2, names=['Batch1', 'Batch2'], band_step=20):
    stats1 = {
        'mean': batch1.mean(dim=(0,2,3)),
        'std': batch1.std(dim=(0,2,3))
    }
    stats2 = {
        'mean': batch2.mean(dim=(0,2,3)),
        'std': batch2.std(dim=(0,2,3))
    }

    fig = plt.figure(figsize=(15,5))
    
    # 均值对比
    plt.subplot(121)
    plt.plot(stats1['mean'].numpy(), 'b-', label=f'{names[0]} Mean')
    plt.plot(stats2['mean'].numpy(), 'r--', label=f'{names[1]} Mean')
    plt.xlabel('Band Number')
    plt.ylabel('Reflectance')
    plt.xticks(np.arange(0, len(stats1['mean']), band_step))
    plt.legend()
    plt.title('Spectral Mean Comparison')

    # 标准差对比
    plt.subplot(122)
    plt.plot(stats1['std'].numpy(), 'b-', label=f'{names[0]} Std')
    plt.plot(stats2['std'].numpy(), 'r--', label=f'{names[1]} Std')
    plt.xlabel('Band Number')
    plt.ylabel('Standard Deviation')
    plt.xticks(np.arange(0, len(stats1['std']), band_step))
    plt.legend()
    plt.title('Spectral Std Comparison')

    plt.tight_layout()
    save_figure(fig, "hsi_spectral_comparison.png")
    return stats1, stats2

# ---------------------- 每类光谱均值±方差对比 ---------------------
def compare_classwise_spectra(X1, y1, X2, y2, class_names=None):
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(torch.unique(y1)))]

    num_classes = len(class_names)
    for cls in range(num_classes):
        idx1 = (y1 == cls)
        idx2 = (y2 == cls)
        data1 = X1[idx1]  # (N1, C, H, W)
        data2 = X2[idx2]

        mean1 = data1.mean(dim=(0, 2, 3))
        std1 = data1.std(dim=(0, 2, 3))
        mean2 = data2.mean(dim=(0, 2, 3))
        std2 = data2.std(dim=(0, 2, 3))

        fig, ax = plt.subplots(figsize=(10, 5))
        bands = np.arange(mean1.shape[0])
        ax.plot(bands, mean1, 'b-', label='Batch1 Mean')
        ax.fill_between(bands, mean1 - std1, mean1 + std1, color='blue', alpha=0.2)

        ax.plot(bands, mean2, 'r--', label='Batch2 Mean')
        ax.fill_between(bands, mean2 - std2, mean2 + std2, color='red', alpha=0.2)

        ax.set_title(f"Class {cls} - {class_names[cls]}")
        ax.set_xlabel("Band Number")
        ax.set_ylabel("Reflectance")
        ax.legend()
        ax.grid(True)

        save_figure(fig, f"class_{cls}_spectral_comparison.png")

# ---------------------- 可视化样本对比 ---------------------
def visualize_hsi_samples(batch1, batch2, sample_idx=0, bands=[20, 50, 100]):
    fig = plt.figure(figsize=(15,10))
    
    # RGB合成显示
    plt.subplot(2, 3, 1)
    plt.imshow(batch1[sample_idx, bands, :, :].permute(1,2,0))
    plt.title('Batch1 RGB Composite')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(batch2[sample_idx, bands, :, :].permute(1,2,0))
    plt.title('Batch2 RGB Composite')
    plt.axis('off')

    # 单波段对比
    band = bands[1]
    vmin = min(batch1[sample_idx, band].min(), batch2[sample_idx, band].min())
    vmax = max(batch1[sample_idx, band].max(), batch2[sample_idx, band].max())
    
    plt.subplot(2, 3, 2)
    plt.imshow(batch1[sample_idx, band], cmap='jet', vmin=vmin, vmax=vmax)
    plt.title(f'Batch1 Band {band}')
    plt.colorbar()
    
    plt.subplot(2, 3, 5)
    plt.imshow(batch2[sample_idx, band], cmap='jet', vmin=vmin, vmax=vmax)
    plt.title(f'Batch2 Band {band}')
    plt.colorbar()

    # 差异热力图
    plt.subplot(1, 3, 3)
    diff = (batch2[sample_idx] - batch1[sample_idx]).abs().mean(dim=0)
    plt.imshow(diff, cmap='hot')
    plt.title('Absolute Difference')
    plt.colorbar()
    
    plt.tight_layout()
    save_figure(fig, "hsi_spatial_comparison.png")

# ---------------------- 主执行流程 ---------------------
if __name__ == "__main__":
    print("加载数据中...")
    # 第一批数据
    data_loader = Data_Loader()
    data_loader.load_data()
    X, y = data_loader.get_data()
    X = torch.tensor(X, dtype=torch.float32).permute(0, 3, 1, 2)
    y = torch.tensor(y)
    print("标签分布：", np.unique(y, return_counts=True))

    #     # 第二批数据
    data_loader_new = Data_Loader()
    data_loader_new.load_data(hsi_root="run2")
    X_new, y_new = data_loader_new.get_data()
    X_new = torch.tensor(X_new, dtype=torch.float32).permute(0, 3, 1, 2)
    y_new = torch.tensor(y_new)
    print("标签分布：", np.unique(y_new, return_counts=True))

    print(f"Run2 num: {X_new.shape[0]}")
    # 假设 X 是 HSI 数据 [n, c, h, w]，y 是标签 [n]
    class_names = ["Healthy", "2dpi", "4dpi", "6dpi"]
    compare_four_pixelwise_classes(X_new, y_new, classes=[0, 1, 2, 3], class_names=class_names, band_step=20, output_prefix="four_classes")
#     top_bands, band_scores = select_bands_by_class_difference(
#     X, y, classes=[0, 1, 2, 3], class_names=class_names, top_k=100
# )
   


#     # 假设 X 是高光谱数据 (n_samples, n_bands), y 是标签 (n_samples,)
#     # 1. 标准化数据（必须）
#     X_scaled = StandardScaler().fit_transform(X)

#     # 2. 初步降维（可选但推荐）
#     pca = PCA(n_components=50)  # 降至50维以加速计算
#     X_pca = pca.fit_transform(X_scaled)

#     # 3. 运行t-SNE
#     tsne = TSNE(
#         n_components=2,      # 降为2D可视化
#         perplexity=30,       # 根据样本量调整（建议5-50）
#         n_iter=1000,         # 迭代次数（复杂数据需增加）
#         learning_rate=200,   # 学习率（高维数据可提高到500-1000）
#         random_state=42      # 固定随机种子保证可复现
#     )
#     X_tsne = tsne.fit_transform(X_pca)  # 如果跳过PCA，直接用X_scaled

#     # 4. 可视化（按真实标签着色）
#     plt.figure(figsize=(10, 8))
#     scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.6)
#     plt.colorbar(scatter, label='Class Label')
#     plt.title('t-SNE Visualization of Hyperspectral Data')
#     plt.xlabel('t-SNE Dimension 1')
#     plt.ylabel('t-SNE Dimension 2')
#     plt.show()
#     with open("selected_top_bands.txt", "w") as f:
#         f.write("\n".join(map(str, top_bands)))  # 每行一个波段编号

# # 加载时：
#     with open("selected_top_bands.txt", "r") as f:
#         loaded_top_bands = np.array([int(line.strip()) for line in f.readlines()])
#     print("Loaded top bands:", loaded_top_bands)

    
# #     # 第二批数据
#     data_loader_new = Data_Loader()
#     data_loader_new.load_data(hsi_root="run2")
#     X_new, y_new = data_loader_new.get_data()
#     X_new = torch.tensor(X_new, dtype=torch.float32).permute(0, 3, 1, 2)
#     y_new = torch.tensor(y_new)
#     compare_four_pixelwise_classes(X_new, y_new, classes=[0, 1, 2, 3], class_names=class_names, band_step=20, output_prefix="four_classes")
#     print(f"Run2 num: {X_new.shape[0]}")
#     print("开始生成对比分析报告...")

#     # 1. 整体 HSI 光谱分布对比
#     hsi_stats1, hsi_stats2 = compare_hsi_batches(X, X_new)

#     # 2. 可视化样本空间差异
#     visualize_hsi_samples(X, X_new)

#     # 3. 每类平均光谱 ± 标准差 对比图
#     class_names = ["Healthy", "Mild", "Moderate", "Severe"]
#     compare_classwise_spectra(X, y, X_new, y_new, class_names=class_names)

#     # 4. 报告总结
#     hsi_diff = (hsi_stats2['mean'] - hsi_stats1['mean']).abs().mean()
#     report = f"""=== 跨批次数据差异报告 ===
# HSI平均光谱差异: {hsi_diff:.4f}

# [关键建议]"""
#     if hsi_diff > 0.1:
#         report += "\n  * HSI批次间存在显著光谱偏移，建议进行辐射归一化"
#     else:
#         report += "\n  * HSI整体差异较小，重点检查各类别是否偏移"

#     # 保存报告
#     report_path = os.path.join(output_dir, "comparison_report.txt")
#     with open(report_path, 'w') as f:
#         f.write(report)

#     print("分析完成！所有图像与报告已保存至:", output_dir)
#     print("\n报告摘要:\n", report)
 