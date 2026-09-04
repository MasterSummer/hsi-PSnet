from PIL import Image
import os

# 定义裁剪区域
x, y, w, h = 1205, 820, 160, 160
crop_area = (x, y, x + w, y + h)

# 图片所在的目录列表
src_dirs = [
    # 'Data for OneDrive/Run 1/2dpi',
    # 'Data for OneDrive/Run 1/4dpi',
    # 'Data for OneDrive/Run 1/6dpi',
    # 'Data for OneDrive/Run 1/Healthy/Day 2',
    # 'Data for OneDrive/Run 1/Healthy/Day 4',
    # 'Data for OneDrive/Run 1/Healthy/Day 6',
    'RGB2/2dpi',
    'RGB2/4dpi',
    'RGB2/6dpi',
    'RGB2/Healthy',
]


cnt = 0

for src_dir in src_dirs:
    if not os.path.exists(src_dir):
        print(f"Directory not found: {src_dir}")
        continue

    for filename in os.listdir(src_dir):
        file_path = os.path.join(src_dir, filename)

        # 删除已有的 .png 文件
        if filename.lower().endswith('.png'):
            os.remove(file_path)
            print(f"Deleted PNG: {file_path}")
            continue

        # 仅处理 .rgb888 文件
        if not filename.lower().endswith('.jpg'):
            continue

        try:
            # 提取编号：最后一个下划线后的部分
            base_name = os.path.splitext(filename)[0]
            number_str = base_name.split('_')[-1]
            number = int(number_str)
        except ValueError:
            print(f"Cannot extract number from: {filename}")
            continue

        # 跳过编号 <= 20 的图片
        if number <= 20:
            continue

        try:
            # 打开并裁剪
            img = Image.open(file_path)
            cropped_img = img.crop(crop_area)

            # 保存为 .jpg
            new_filename = f"{base_name}.jpg"
            new_path = os.path.join(src_dir, new_filename)
            cropped_img.save(new_path, 'JPEG')
            cnt += 1
            print(f"[{cnt}] Saved: {new_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
