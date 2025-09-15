import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import random
from scipy.ndimage import rotate, shift
from scipy.signal import savgol_filter
from skimage.util import random_noise
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch 
# class SynchronizedNaturalLight(nn.Module):
#     def __init__(self, 
#                  hsi_channels=313,
#                  rgb_channel_indices=(115, 72, 34), # (R, G, B) ~ (640nm, 550nm, 470nm)
#                  brightness_range=(0.7, 1.3), 
#                  spectral_strength=0.05, 
#                  hsi_sensor_noise_std=0.01,
#                  rgb_sensor_noise_std=0.01,
#                  p=0.75): # 默认在75%的样本上应用
#         super().__init__()
    
#         self.hsi_channels = hsi_channels
#         self.rgb_indices = rgb_channel_indices
#         self.brightness_range = brightness_range
#         self.spectral_strength = spectral_strength
#         self.hsi_sensor_noise_std = hsi_sensor_noise_std
#         self.rgb_sensor_noise_std = rgb_sensor_noise_std
#         self.p = p

#     def forward(self, rgb_tensor, hsi_tensor):
#         if torch.rand(1).item() > self.p:
#             return rgb_tensor, hsi_tensor
#         device = hsi_tensor.device
#         brightness_factor = torch.rand(1, device=device) * (self.brightness_range[1] - self.brightness_range[0]) + self.brightness_range[0]
#         hsi_augmented = hsi_tensor * brightness_factor
#         rgb_augmented = rgb_tensor * brightness_factor
#         low_dim_noise = torch.randn(1, 1, 10, device=device)
#         spectral_noise_curve = F.interpolate(low_dim_noise, size=self.hsi_channels, mode='linear', align_corners=False).squeeze()
#         hsi_spectral_noise = spectral_noise_curve.view(-1, 1, 1) * self.spectral_strength
#         hsi_augmented = hsi_augmented + hsi_spectral_noise
#         r_noise, g_noise, b_noise = spectral_noise_curve[self.rgb_indices[0]], spectral_noise_curve[self.rgb_indices[1]], spectral_noise_curve[self.rgb_indices[2]]
#         rgb_spectral_noise = torch.tensor([r_noise, g_noise, b_noise], device=device).view(3, 1, 1) * self.spectral_strength
#         rgb_augmented = rgb_augmented + rgb_spectral_noise
#         hsi_sensor_noise = torch.randn_like(hsi_tensor) * self.hsi_sensor_noise_std
#         rgb_sensor_noise = torch.randn_like(rgb_tensor) * self.rgb_sensor_noise_std
#         hsi_augmented = hsi_augmented + hsi_sensor_noise
#         rgb_augmented = rgb_augmented + rgb_sensor_noise
#         hsi_augmented = torch.clamp(hsi_augmented, 0, 1)
#         rgb_augmented = torch.clamp(rgb_augmented, 0, 1)
#         return rgb_augmented, hsi_augmented

class MultiModalDataset(Dataset):
    def __init__(self, rgb_files, hsi_data, labels, transform_rgb=None, transform_hsi=None):
        self.rgb_files = rgb_files  # RGB 图像路径列表
        self.hsi_data = hsi_data  # HSI Tensor 数据
        self.labels = labels  # 标签
        self.transform_rgb = transform_rgb
        self.transform_hsi = transform_hsi

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 读取 RGB 图像
        rgb_img = Image.open(self.rgb_files[idx]).convert("RGB")
        if self.transform_rgb:
            rgb_img = self.transform_rgb(rgb_img)

        # 读取 HSI 数据
        hsi_img = self.hsi_data[idx]  # 直接从 HSI Tensor 获取
        if self.transform_hsi:
            hsi_img = self.transform_hsi(hsi_img)

        label = self.labels[idx]
        return rgb_img, hsi_img, label
    
class Data_Loader:
    def __init__(self, samples_per_directory=144, directories = ['2dpi', '4dpi', '6dpi', 'Healthy']):
        self.directories = directories
        self.samples_per_directory = samples_per_directory
        self.n = 1
        self.h = 132
        self.w = 135
        self.c = 313
        self.X = []
        self.y = []
        self.sample_count = 0
        self.spa_col = [ 59,  92,  97,  95,  49,  28, 182,  68,  44,   8, 111, 168, 305,
    3, 140, 300,  55, 288, 149,  82, 278, 307, 312, 283, 101, 158,
    0,  21, 266, 129, 226]
    
    
    def load_data(self, norm_method=None, hsi_root = "run1"):
        """支持多种归一化方法的数据加载函数
        Args:
            norm_method (str/None): 可选值：
                - None: 不归一化
                - "standard": 全局标准化（展平后标准化）
                - "minmax": 全局Min-Max归一化到[0,1]
                - "bandwise": 逐波段标准化（保持空间结构）
                - "bandminmax": 逐波段Min-Max归一化到[0,1]
        """
        # 清空已有数据
        self.X, self.y = [], []

        # 排
        custom_order = ['Healthy', '2dpi', '4dpi', '6dpi']
        order_dict = {name: i for i, name in enumerate(custom_order)}
        self.directories = [os.path.join(hsi_root, d) for d in os.listdir(hsi_root) if os.path.isdir(os.path.join(hsi_root, d))]
        self.directories.sort(key=lambda x: order_dict.get(os.path.basename(x), float('inf')))
        
        # 加载原始数据
        for label, directory in enumerate(self.directories):
            if not os.path.exists(directory):
                print(f"警告: 类别目录 {directory} 不存在，跳过...")
                continue

            npy_files = [f for f in os.listdir(directory) if f.endswith('.npy')]

            for npy_file in npy_files:
                data_path = os.path.join(directory, npy_file)
                data = np.load(data_path)
                self.X.append(data)
                self.y.append(label)

        # 转换为numpy数组
        self.X = np.array(self.X)
        self.X = self.X.reshape(-1, self.h, self.w, self.c)# 形状 (n, h, w, c)
        self.y = np.array(self.y)
        self.n = len(self.y)

        # 执行归一化
        if norm_method:
            original_shape = self.X.shape

            if norm_method == "standard":
                # 全局标准化（展平后处理）
                scaler = StandardScaler()
                self.X = scaler.fit_transform(self.X.reshape(self.n, -1))
                self.X = self.X.reshape(original_shape)

            elif norm_method == "minmax":
                # 全局Min-Max归一化
                min_val = np.min(self.X)
                max_val = np.max(self.X)
                self.X = (self.X - min_val) / (max_val - min_val + 1e-8)

            elif norm_method == "bandwise":
                # 逐波段标准化
                mean = np.mean(self.X, axis=(0, 1, 2))  # (c,)
                std = np.std(self.X, axis=(0, 1, 2))  # (c,)
                std[std < 1e-8] = 1e-8  # 防止除零
                self.X = (self.X - mean) / std

            elif norm_method == "bandminmax":
                # 逐波段Min-Max归一化
                min_vals = np.min(self.X, axis=(0, 1, 2))  # (c,)
                max_vals = np.max(self.X, axis=(0, 1, 2))  # (c,)
                range_vals = max_vals - min_vals
                range_vals[range_vals < 1e-8] = 1e-8  # 防止除零
                self.X = (self.X - min_vals) / range_vals

            else:
                raise ValueError(f"不支持的归一化方法: {norm_method}")

            print(f"归一化方法 [{norm_method}] 应用成功")

        # 确保形状一致性
        self.X = self.X.reshape(self.n, self.h, self.w, self.c)


    def get_data(self):
        return self.X.reshape(self.n,self.h,self.w,self.c), self.y

    def bandwise_normalize(train_data, test_data=None):
        """每个波段独立标准化"""
        # 计算训练集统计量
        mean = np.mean(train_data, axis=(0, 1))  # (B,)
        std = np.std(train_data, axis=(0, 1))  # (B)

        # 防止除以0
        eps = 1e-8
        std[std < eps] = eps

        # 归一化训练集
        norm_train = (train_data - mean) / std

        # 归一化测试集
        if test_data is not None:
            norm_test = (test_data - mean) / std
            return norm_train, norm_test

        return norm_train

    def global_normalize(train_data, test_data=None):
        """整个数据立方体统一标准化"""
        global_mean = np.mean(train_data)
        global_std = np.std(train_data)

        eps = 1e-8
        if global_std < eps:
            global_std = eps

        norm_train = (train_data - global_mean) / global_std

        if test_data is not None:
            norm_test = (test_data - global_mean) / global_std
            return norm_train, norm_test

        return norm_train

    def patch_normalize(data, patch_size=64):
        """对单个图像进行分块归一化"""
        H, W, B = data.shape
        normalized = np.zeros_like(data)

        for i in range(0, H, patch_size):
            for j in range(0, W, patch_size):
                # 获取当前块
                i_end = min(i + patch_size, H)
                j_end = min(j + patch_size, W)
                patch = data[i:i_end, j:j_end, :]

                # 计算块统计量
                patch_mean = np.mean(patch, axis=(0, 1))
                patch_std = np.std(patch, axis=(0, 1))

                # 归一化
                normalized[i:i_end, j:j_end, :] = (patch - patch_mean) / (patch_std + 1e-8)

        return normalized

    def spectral_normalize(data):
        """每个像素的光谱向量独立归一化"""
        pixel_mean = np.mean(data, axis=2, keepdims=True)  # (H, W, 1)
        pixel_std = np.std(data, axis=2, keepdims=True)  # (H, W, 1)
        return (data - pixel_mean) / (pixel_std + 1e-8)

    def denoising_normalize(data, noise_bands=None):
        """带噪声波段过滤的归一化"""
        # 示例噪声波段检测（需根据实际数据调整）
        if noise_bands is None:
            # 假设前5个和后5个波段噪声较高
            noise_bands = list(range(5)) + list(range(data.shape[2] - 5, data.shape[2]))

        # 保留有效波段
        valid_bands = [b for b in range(data.shape[2]) if b not in noise_bands]
        cleaned_data = data[:, :, valid_bands]

        # 执行逐波段标准化
        return bandwise_normalize(cleaned_data)



class RGBLoader:
    """RGB数据加载器，保持与HSI相同的排序"""
    def __init__(self, data_root="RGB"):
        self.data_root = data_root
        self.class_order = ['Healthy', '2dpi', '4dpi', '6dpi']  # 与HSI相同的顺序
        
    def load_paths(self):
        # 创建排序字典
        order_dict = {name: i for i, name in enumerate(self.class_order)}
        
        # 获取所有目录并排序
        directories = [os.path.join(self.data_root, d) 
                      for d in os.listdir(self.data_root) 
                      if os.path.isdir(os.path.join(self.data_root, d))]
        directories.sort(key=lambda x: order_dict.get(os.path.basename(x), float('inf')))

        image_paths = []
        labels = []
        
        # 遍历类别目录
        for label, directory in enumerate(directories):
            if not os.path.exists(directory):
                print(f"警告: 类别目录 {directory} 不存在，跳过...")
                continue

            # 获取当前目录下的所有 .jpg 文件
            jpg_files = [f for f in os.listdir(directory) if f.lower().endswith(".jpg")]

            # 遍历文件
            for jpg_file in jpg_files:
                image_paths.append(os.path.join(directory, jpg_file))
                labels.append(label)  # 使用目录顺序作为标签
                
        return image_paths, labels


    
def get_loader(random_state=42):
    """准备数据集，保持原划分逻辑：先分20%测试，再从剩余分50%验证"""
    # 1. 加载数据（自动保持排序）
    hsi_loader = HSILoader()
    rgb_loader = RGBLoader()
    
    X, y = hsi_loader.load_data()
    rgb_paths, y_rgb = rgb_loader.load_paths()
    n, h, w, c = X.shape

    X_tensor = torch.tensor(X, dtype=torch.float32)  # (n, h, w, c)
    X = X_tensor.permute(0, 3, 1, 2)  # 转换为 (n, c, h, w)
    y = torch.tensor(y, dtype=torch.long)  # 适用于分类任务
    # 2. 验证数据对齐
    assert torch.equal(y_hsi, y_rgb), "HSI和RGB标签不匹配"
    assert len(rgb_paths) == len(X_hsi), "HSI和RGB样本数量不匹配"
    data_pairs = list(zip(rgb_image_paths, X, y))  # [(RGB路径, HSI, 标签), ...]

    hsi_ids = [os.path.splitext(os.path.basename(path))[0] for path in rgb_image_paths]

    # 构建RGB路径字典：{id: path}
    rgb_path_dict = {os.path.splitext(os.path.basename(path))[0]: path for path in rgb_image_paths}

    # 验证并创建配对
    data_pairs = []
    for i, hsi_id in enumerate(hsi_ids):
        if hsi_id not in rgb_path_dict:
            raise ValueError(f"找不到HSI样本{hsi_id}对应的RGB图像")
        data_pairs.append((rgb_path_dict[hsi_id], X[i], y[i]))


    # 4️⃣ **一次性划分数据集，确保 RGB 和 HSI 顺序完全一致**

    train_pairs, temp_pairs = train_test_split(data_pairs, test_size=0.2, stratify=y, random_state=SEED)
    val_pairs, test_pairs = train_test_split(temp_pairs, test_size=0.5, stratify=[p[2] for p in temp_pairs], random_state=SEED)

    # **解包数据**
    rgb_train, X_train, y_train = zip(*train_pairs)
    rgb_val, X_val, y_val = zip(*val_pairs)
    rgb_test, X_test, y_test = zip(*test_pairs)

    # 转换为 Tensor
    X_train, y_train = torch.stack(X_train), torch.tensor(y_train)
    X_val, y_val = torch.stack(X_val), torch.tensor(y_val)
    X_test, y_test = torch.stack(X_test), torch.tensor(y_test)

    # 5️⃣ 归一化参数**
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
        # AddGaussianNoise(mean=0., std=0.05)
    ])
    transform_hsi_train = transforms.Compose([
        transforms.Normalize(mean=mean_hsi.tolist(), std=std_hsi.tolist()),
        # AddGaussianNoise(mean=0., std=0.1)
    ])
    transform_rgb_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0175, 0.0130, 0.0146], std=[0.0113, 0.0120, 0.0113]),
        # AddGaussianNoise(mean=0., std=0.1)
    ])
    train_dataset = MultiModalDataset(rgb_train, X_train, y_train, transform_rgb=transform_rgb, transform_hsi=transform_hsi)
    val_dataset = MultiModalDataset(rgb_val, X_val, y_val, transform_rgb=transform_rgb, transform_hsi=transform_hsi)
    test_dataset = MultiModalDataset(rgb_test, X_test, y_test, transform_rgb=transform_rgb, transform_hsi=transform_hsi)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    return train_loader,val_loader,test_loader
    
    

    
    return train_loader, val_loader, test_loader
# Example usage:
if __name__ == "__main__":

    dataloader = Data_Loader()
    dataloader.load_data()
    X, y = dataloader.get_data()

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # augmented_dataset , y = augment_and_increase_samples(X,y, target_ratio=2)
    # print("Original dataset shape:", X.shape)
    # print("Augmented dataset shape:", augmented_dataset.shape, y.shape)
    # np.save('augmented_images1.npy', augmented_dataset)
    # np.save('augmented_labels1.npy', y)
    # print(f"Shape of X: {X.shape}")
    # print(f"Mean of X: {np.mean(X)}")
    # print(f"Standard deviation of X: {np.std(X)}")
    # print(f"Min of X: {np.min(X)}")
    # print(f"Max of X: {np.max(X)}")
