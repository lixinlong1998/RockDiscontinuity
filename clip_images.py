import os
from PIL import Image


def crop_images_by_region(input_folder, output_folder, width_range, height_range):
    """
    批量裁剪图片的指定区域

    参数:
    input_folder: 输入文件夹路径
    output_folder: 输出文件夹路径
    width_range: 宽度区间，元组 (left, right)，单位像素
    height_range: 高度区间，元组 (bottom, top)，单位像素
    """
    # 支持的图片格式
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif')

    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 获取所有图片文件
    image_files = []
    for file in os.listdir(input_folder):
        if file.lower().endswith(supported_formats):
            image_files.append(file)

    if not image_files:
        print(f"在文件夹 {input_folder} 中未找到支持的图片文件")
        return

    print(f"找到 {len(image_files)} 张图片")

    processed_count = 0
    for filename in image_files:
        try:
            # 打开图片
            input_path = os.path.join(input_folder, filename)
            image = Image.open(input_path)

            # 转换为RGB模式（解决RGBA问题）
            if image.mode in ['RGBA', 'P', 'LA']:
                image = image.convert('RGB')

            # 获取原图尺寸
            original_width, original_height = image.size
            print(original_width, original_height)
            # 解析裁剪区间
            left, right = width_range
            bottom, top = height_range

            # 确保区间在图片范围内
            left = max(0, left)
            right = min(original_width, right)
            bottom = max(0, bottom)
            top = min(original_height, top)

            # 确保区间有效
            if left >= right or bottom >= top:
                print(f"跳过 {filename}: 裁剪区间无效")
                continue

            # 计算裁剪区域 (left, bottom, right, top)
            # 注意：PIL使用(左上X, 左上Y, 右下X, 右下Y)坐标系统
            # 我们的bottom是从底部开始，需要转换为从顶部开始
            top_y = original_height - top
            bottom_y = original_height - bottom

            # 裁剪图片
            cropped_image = image.crop((left, bottom, right, top))

            # 生成输出文件名
            name, ext = os.path.splitext(filename)
            output_filename = f"{name}_cropped{ext}"
            output_path = os.path.join(output_folder, output_filename)

            # 保存图片（保持原格式，确保是RGB模式）
            cropped_image.save(output_path)

            print(f"已处理: {filename}")
            processed_count += 1

        except Exception as e:
            print(f"处理 {filename} 时出错: {e}")

    print(f"\n处理完成！成功处理 {processed_count}/{len(image_files)} 张图片")


# 使用示例
if __name__ == "__main__":
    input_folder = r"E:\Projects\20240610_新疆天山-独库野外监测方案_姣姐\20251205_提交一些已有的材料\潜在崩塌区_谷歌地球影像"
    output_folder = r"E:\Projects\20240610_新疆天山-独库野外监测方案_姣姐\20251205_提交一些已有的材料\潜在崩塌区_谷歌地球影像_clip"

    # 设置裁剪区间
    # width_range: (left, right) - 宽度方向从left到right
    # height_range: (bottom, top) - 高度方向从底部bottom到顶部top
    # 例如：裁剪左下角 800x600 的区域
    width_range = (500, 2560)  # 从左边0像素到800像素
    height_range = (95, 1380)  # 从底部600像素到顶部0像素（即从底部往上600像素）

    # 或者，如果你想要从底部裁剪300像素高，整个宽度：
    # width_range = (0, 1920)    # 整个宽度
    # height_range = (300, 0)    # 从底部往上300像素

    # 执行裁剪
    crop_images_by_region(input_folder, output_folder, width_range, height_range)
