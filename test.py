# import torch
# import random
# import os
# from sklearn.model_selection import train_test_split
# from Data_Loader import *

# SEED = 42
# random.seed(SEED)
# torch.manual_seed(SEED)

# # 加载数据
# data_loader = Data_Loader()
# data_loader.load_data(hsi_root="run1")
# X = torch.tensor(data_loader.X, dtype=torch.float32).permute(0, 3, 1, 2)  # [N,C,H,W]
# y = torch.tensor(data_loader.y, dtype=torch.long)

# rgb_loader = RGBLoader(data_root="RGB")
# rgb_paths, rgb_labels = rgb_loader.load_paths()

# # 验证对齐
# assert len(X) == len(rgb_paths)
# assert y.tolist() == rgb_labels

# # 打包数据
# data_pairs = list(zip(rgb_paths, X, y))

# # 固定划分 test = 10%

# # 固定划分 test = 10%
# train_pairs, temp_pairs = train_test_split(data_pairs, test_size=0.2, stratify=y, random_state=SEED)
# val_pairs, test_pairs = train_test_split(temp_pairs, test_size=0.5, stratify=[p[2] for p in temp_pairs], random_state=SEED)
# trainval_pairs = train_pairs + val_pairs
# # 保存为文件（方便后续加载）
# os.makedirs("splits", exist_ok=True)
# torch.save(trainval_pairs, "splits/trainval.pt")
# torch.save(test_pairs, "splits/test.pt")
# print("✅ Saved fixed splits: trainval.pt and test.pt")

import torch

# 加载原始划分
trainval_pairs = torch.load("splits/trainval.pt")
test_pairs     = torch.load("splits/test.pt")

# 要排除的索引
exclude = {20, 23}

# 重新生成 test_pairs（去掉 20 和 23）
new_test_pairs = [pair for i, pair in enumerate(test_pairs) if i not in exclude]

# 保存新的 test 划分
torch.save(new_test_pairs, "splits/cleantest.pt")

print(f"原始测试集大小: {len(test_pairs)}")
print(f"新测试集大小: {len(new_test_pairs)} (已去掉 {exclude})")
