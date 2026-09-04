import torch
import numpy as np
from sklearn.decomposition import PCA
import os
import pickle
from Data_Loader import Data_Loader, MultiModalDataset, RGBLoader

class HSIPCAProcessor:
    def __init__(self, n_components=10, cache_dir='pca_cache'):
        self.n_components = n_components
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def apply_pca_to_hsi(self, hsi_image):
        """应用PCA到单个HSI图像"""
        h, w, c = hsi_image.shape
        # 展平空间维度
        flattened = hsi_image.reshape(-1, c)
        # 执行PCA
        pca = PCA(n_components=self.n_components)
        transformed = pca.fit_transform(flattened)
        # 恢复空间维度
        return transformed.reshape(h, w, self.n_components), pca
    
    def process_and_save(self, X, y, prefix='dataset'):
        """处理并保存PCA结果"""
        n, c, h, w = X.shape
        pca_results = []
        pca_models = []
        
        for i in range(n):
            # 转换为(h, w, c)格式
            hsi_image = X[i].permute(1, 2, 0).numpy()
            # 应用PCA
            pca_result, pca_model = self.apply_pca_to_hsi(hsi_image)
            pca_results.append(pca_result)
            pca_models.append(pca_model)
        
        # 转换为(n, h, w, n_components)并保存
        pca_tensor = torch.tensor(np.array(pca_results), dtype=torch.float32)
        # 转换为PyTorch标准的(n, n_components, h, w)格式
        pca_tensor = pca_tensor.permute(0, 3, 1, 2)
        
        # 保存结果
        cache_path = os.path.join(self.cache_dir, f'{prefix}_pca.pt')
        torch.save({
            'X': pca_tensor,
            'y': y,
            'pca_models': pca_models
        }, cache_path)
        
        return pca_tensor
    
    @staticmethod
    def load_pca_data(prefix='dataset', cache_dir='pca_cache'):
        """加载已处理的PCA数据"""
        cache_path = os.path.join(cache_dir, f'{prefix}_pca.pt')
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"PCA缓存文件 {cache_path} 不存在")
        return torch.load(cache_path)

# 测试用例
if __name__ == "__main__":

    data_loader = Data_Loader()
    data_loader.load_data()
    X, y = data_loader.get_data()
    n, h, w, c = X.shape
    X_tensor = torch.tensor(X, dtype=torch.float32)  # (n, h, w, c)
    X = X_tensor.permute(0, 3, 1, 2)  # 转换为 (n, c, h, w)
    y = torch.tensor(y, dtype=torch.long)  # 适用于分类任务
    # 初始化处理器
    processor = HSIPCAProcessor(n_components=10)
    
    # 处理并保存数据
    pca_X = processor.process_and_save(X, y, prefix='test')
    print("PCA处理后形状:", pca_X.shape)  # 应为 (10, 10, 64, 64)
    
    # 加载数据
    loaded_data = HSIPCAProcessor.load_pca_data(prefix='test')
    print("加载的数据形状:", loaded_data['X'].shape)
    print("标签数量:", len(loaded_data['y']))
    print("PCA模型数量:", len(loaded_data['pca_models']))
