import torch
import numpy as np
import os 
from tqdm import tqdm

# --- 1. 加载并合并你的数据 ---
print("Loading and merging trainval and test data...")
trainval_pairs = torch.load("splits/trainval.pt")
test_pairs     = torch.load("splits/test.pt")

# 提取 trainval 数据
rgb_trainval = [p[0] for p in trainval_pairs]
X_trainval   = torch.stack([p[1] for p in trainval_pairs])
y_trainval   = torch.tensor([p[2] for p in trainval_pairs])

# 提取 test 数据
rgb_test = [p[0] for p in test_pairs]
X_test   = torch.stack([p[1] for p in test_pairs])
y_test   = torch.tensor([p[2] for p in test_pairs])

# 将 trainval 和 test 数据集合并
all_rgb_identifiers = rgb_trainval + rgb_test
all_X_hsi = torch.cat([X_trainval, X_test], dim=0)
all_y = torch.cat([y_trainval, y_test], dim=0)

print(f"Total samples to analyze: {len(all_X_hsi)}")
print(f"Data shape: {all_X_hsi.shape}")

# --- 2. 计算每个样本的统计量 ---
print("\nStep 1: Calculating per-sample statistics...")
# dim=[1,2,3] 表示在通道、高度、宽度维度上计算
sample_means = torch.mean(all_X_hsi, dim=[1, 2, 3]).numpy()
sample_variances = torch.var(all_X_hsi, dim=[1, 2, 3]).numpy()

print("Per-sample statistics calculation complete.")
print(f"Calculated means for {len(sample_means)} samples.")
print(f"Calculated variances for {len(sample_variances)} samples.")

# --- 3. 定义离群点检测函数 (基于3倍标准差) ---
def find_outliers_by_std(values, identifiers, stat_name, std_threshold=3):
    """
    使用标准差法找到离群点，并返回它们的详细信息。
    
    :param values: 一维Numpy数组 (e.g., sample_means)
    :param identifiers: 样本的标识符列表 (e.g., all_rgb_identifiers)
    :param stat_name: 统计量的名称 (e.g., 'Mean' or 'Variance')
    :param std_threshold: N倍标准差的阈值
    :return: 离群点的索引列表, 离群点的详细信息字典
    """
    if len(values) == 0:
        return np.array([]), {}
        
    mean_of_vals = np.mean(values)
    std_of_vals = np.std(values)
    
    print(f"\n--- Analyzing {stat_name} ---")
    print(f"Global Mean of {stat_name}s: {mean_of_vals:.4f}")
    print(f"Global Std Dev of {stat_name}s: {std_of_vals:.4f}")
    
    lower_bound = mean_of_vals - std_threshold * std_of_vals
    upper_bound = mean_of_vals + std_threshold * std_of_vals
    
    print(f"Identifying outliers outside the range [{lower_bound:.4f}, {upper_bound:.4f}]")
    
    # 计算每个点的Z-score
    z_scores = np.abs((values - mean_of_vals) / std_of_vals)
    
    # 找到Z-score超过阈值的点的索引
    outlier_indices = np.where(z_scores > std_threshold)[0]
    
    outlier_details = {}
    for idx in outlier_indices:
        outlier_details[idx] = {
            'identifier': identifiers[idx],
            'value': values[idx],
            'z_score': z_scores[idx]
        }
        
    return outlier_indices, outlier_details

# --- 4. 执行离群点检测 ---
STD_THRESHOLD = 3

# 4.1 基于样本均值（亮度）检测离群点
mean_outlier_indices, mean_outlier_details = find_outliers_by_std(
    sample_means, all_rgb_identifiers, 'Mean (Brightness)', STD_THRESHOLD
)

print(f"\nFound {len(mean_outlier_indices)} outliers based on MEAN (Z-score > {STD_THRESHOLD}):")
for idx, details in mean_outlier_details.items():
    print(f"  - Index: {idx}, Identifier: {details['identifier']}, Mean: {details['value']:.4f}, Z-score: {details['z_score']:.2f}")

# 4.2 基于样本方差（对比度）检测离群点
variance_outlier_indices, variance_outlier_details = find_outliers_by_std(
    sample_variances, all_rgb_identifiers, 'Variance (Contrast)', STD_THRESHOLD
)

print(f"\nFound {len(variance_outlier_indices)} outliers based on VARIANCE (Z-score > {STD_THRESHOLD}):")
for idx, details in variance_outlier_details.items():
    print(f"  - Index: {idx}, Identifier: {details['identifier']}, Variance: {details['value']:.4f}, Z-score: {details['z_score']:.2f}")

# --- 5. 合并离群点并创建干净的数据集 ---
# 使用 np.union1d 来获取所有唯一离群点的索引
all_outlier_indices = np.union1d(mean_outlier_indices, variance_outlier_indices)
print(f"\nTotal unique outlier indices found: {len(all_outlier_indices)}")
print("Indices to be removed:", sorted(list(all_outlier_indices)))

# 创建一个包含所有“好”数据点索引的掩码(mask)
clean_indices_mask = np.ones(len(all_X_hsi), dtype=bool)
clean_indices_mask[all_outlier_indices] = False
clean_indices = np.where(clean_indices_mask)[0]

# 创建不包含离群点的新数据集
X_hsi_clean = all_X_hsi[clean_indices]
y_clean = all_y[clean_indices]
rgb_identifiers_clean = [all_rgb_identifiers[i] for i in clean_indices]


# --- 6. 重新划分训练集和测试集 ---
print("\nStep 2: Re-splitting the clean data into trainval and test sets...")

# 从原始的、未合并的数据中找到干净样本的来源
trainval_size = len(X_trainval)
clean_trainval_indices_in_original = [i for i in clean_indices if i < trainval_size]
# 注意：测试集索引需要减去训练集的大小才是它在 X_test 中的原始索引
clean_test_indices_in_original = [i - trainval_size for i in clean_indices if i >= trainval_size]

print(f"Original trainval size: {len(X_trainval)}")
print(f"Clean trainval size: {len(clean_trainval_indices_in_original)}")
print(f"Original test size: {len(X_test)}")
print(f"Clean test size: {len(clean_test_indices_in_original)}")

# 创建新的、干净的 trainval 和 test pairs
clean_trainval_pairs = [trainval_pairs[i] for i in clean_trainval_indices_in_original]
clean_test_pairs = [test_pairs[i] for i in clean_test_indices_in_original]

# --- 7. 将干净的数据保存到新的文件中 ---
# 创建一个新的文件夹来存放清洗后的数据，避免覆盖原始文件
output_dir = "splits_clean"
os.makedirs(output_dir, exist_ok=True)

clean_trainval_path = os.path.join(output_dir, "trainval.pt")
clean_test_path = os.path.join(output_dir, "test.pt")

print(f"\nStep 3: Saving clean data splits...")
torch.save(clean_trainval_pairs, clean_trainval_path)
print(f"Clean trainval data saved to: {clean_trainval_path}")
torch.save(clean_test_pairs, clean_test_path)
print(f"Clean test data saved to: {clean_test_path}")

print("\nProcess complete. You can now point your main script to the 'splits_clean' directory.")