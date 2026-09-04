import numpy as np
import random
from scipy.ndimage import rotate, shift
from scipy.signal import savgol_filter
from skimage.util import random_noise
from torchvision.transforms import functional as F
import random
import torch

class HyperspectralAugmentation:
    def __init__(self, preprocessing_method="SG"):
        """
        初始化高光谱图像数据增强类。

        Args:
            preprocessing_method (str): 选择的预处理方法，包括 "S-G"（Savitzky-Golay）、"MSC" 和 "SNV"。
        """
        self.preprocessing_method = preprocessing_method


    def augment_image(self, image):
        if random.random() < 0.5:
            angle = random.choice([90, 180, 270])
            image = rotate(image, angle, axes=(0, 1), reshape=False, mode='reflect')

        if random.random() < 0.5:
            image = np.flip(image, axis=random.choice([0, 1]))

        if random.random() < 0.5:
            shift_x = random.randint(-5, 5)
            shift_y = random.randint(-5, 5)
            image = shift(image, shift=(shift_x, shift_y, 0), mode='reflect')

        if random.random() < 0.5:
            factor = random.uniform(0.8, 1.2)
            image = np.clip(image * factor, 0, 1)

        if random.random() < 0.5:
            image = random_noise(image, mode='gaussian', var=0.01)

        return image

    def augment_and_increase_samples(self, images, target_ratio=1.5):
        """
           执行数据增强并将数据集扩充到目标比例。

           Args:
               images (numpy.ndarray): 原始高光谱图像，形状为 (n, h, w, c)。
               target_ratio (float): 数据集扩充的目标比例。

           Returns:
               numpy.ndarray: 扩充后的数据集。
           """
        original_count = images.shape[0]
        target_count = int(original_count * target_ratio)
        additional_count = target_count - original_count

        augmented_images = []
        while len(augmented_images) < additional_count:
            for image in images:
                preprocessed = self.preprocess(image)
                augmented = self.augment_image(preprocessed)
                augmented_images.append(augmented)
                if len(augmented_images) >= additional_count:
                    break

        augmented_images = np.array(augmented_images)
        return np.concatenate([images, augmented_images], axis=0)


class HyperspectralRandAugment:
    """
    适用于高光谱数据的 RandAugment 实现。

    主要功能：
    - 随机选择 num_ops 个数据增强操作
    - 适用于 3D 高光谱数据 (C, H, W) 格式
    - 包含翻转、旋转、高斯噪声等增强方法

    参数：
    - num_ops: 每次增强时选择的操作数
    - magnitude: 变换强度（可以调整噪声幅度等）
    """

    def __init__(self, num_ops=2, magnitude=9):
        self.num_ops = num_ops  # 每次增强时选择的变换数量
        self.magnitude = magnitude  # 控制某些变换的强度

        # 定义可能的增强操作
        self.transforms = [
            lambda x: F.hflip(x) if random.random() > 0.5 else x,  # 50% 概率水平翻转
            lambda x: F.vflip(x) if random.random() > 0.5 else x,  # 50% 概率垂直翻转
            lambda x: F.rotate(x, angle=random.randint(-30, 30)),  # 随机旋转 [-30°, 30°]
            lambda x: x + (torch.randn_like(x) * 0.05),  # 添加高斯噪声（标准差 0.05）
            lambda x: F.gaussian_blur(x, kernel_size=3) if random.random() > 0.7 else x  # 30% 概率模糊
        ]

    def __call__(self, x):
        """
        对输入的高光谱图像 x 进行增强。

        参数：
        - x: PyTorch Tensor，形状 (C, H, W)

        返回：
        - 变换后的 Tensor
        """
        # 随机选择 num_ops 个操作
        ops = random.sample(self.transforms, self.num_ops)
        for op in ops:
            x = op(x)  # 依次应用这些变换
        return x

# 示例用法
if __name__ == "__main__":
    hyperspectral_images = np.random.rand(10, 132, 135, 313)  # 生成随机数据集
    augmenter = HyperspectralAugmentation(preprocessing_method="MSC")  # 初始化类，选择 MSC 预处理
    augmented_dataset = augmenter.augment_and_increase_samples(hyperspectral_images, target_ratio=1.5)

    print("原始数据集形状:", hyperspectral_images.shape)
    print("扩充后数据集形状:", augmented_dataset.shape)
