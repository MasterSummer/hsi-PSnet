import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

# 参数
input_path = "plant.jpeg"  # 你的图片路径
out_dir = "./results"
os.makedirs(out_dir, exist_ok=True)

# 读取 RGB
img = cv2.imread(input_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 转灰度
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# 选取强烈色差的 colormap（蓝、红、黄、绿等都有）
cmaps = ['jet', 'turbo', 'rainbow', 'plasma', 'hsv']

# 生成 5 张夸张色彩的伪波段
for i, cmap_name in enumerate(cmaps, start=1):
    cmap = plt.get_cmap(cmap_name)
    colored = (cmap(gray / 255.0)[:, :, :3] * 255).astype(np.uint8)

    out_path = os.path.join(out_dir, f"pseudo_band_{i}.png")
    cv2.imwrite(out_path, cv2.cvtColor(colored, cv2.COLOR_RGB2BGR))
    print(f"保存：{out_path}")
