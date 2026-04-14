import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
import random


def build_pointcept_dataset(
        csv_path,
        output_dir,
        tile_size=5.0,
        overlap=0.1,
        min_points=20000,
        max_points=120000,
        min_fracture_ratio=0.02,
        split_ratios=(0.7, 0.15, 0.15)  # Train, Val, Test 比例
):
    """
    岩体结构面点云 PTv3 标准化预处理与拆分管线
    """
    print(f"========== 开始加载源数据 ==========")
    print(f"输入文件: {csv_path}")
    df = pd.read_csv(csv_path)
    out_root = Path(output_dir)

    # 创建目录结构
    for split in ['train', 'val', 'test']:
        (out_root / split).mkdir(parents=True, exist_ok=True)

    # 1. 提取全局基础特征
    coords_all = df[['X', 'Y', 'Z']].values.astype(np.float32)

    # 【核心修改 1】: 将 RGB 和 Normals 拼接，并直接定义为模型期望的 color 概念
    color_all = np.concatenate([
        df[['R', 'G', 'B']].values.astype(np.float32) / 255.0,
        df[['nx', 'ny', 'nz']].values.astype(np.float32)
    ], axis=1)

    # 提取实例与语义标签
    instance_labels = df['Discontinuity_id'].values.astype(np.int64)
    semantic_labels = (instance_labels != -1).astype(np.int64)

    # 2. 计算滑动窗口步长
    stride = tile_size * (1 - overlap)
    x_min, y_min, z_min = coords_all.min(axis=0)
    x_max, y_max, z_max = coords_all.max(axis=0)

    # 1. 第一阶段：仅扫描并收集所有符合条件的块信息，不急着保存
    valid_tiles = []
    print(f"========== 开始 3D 滑动窗口扫描 ==========")
    for x in np.arange(x_min, x_max, stride):
        for y in np.arange(y_min, y_max, stride):
            mask = (coords_all[:, 0] >= x) & (coords_all[:, 0] < x + tile_size) & \
                   (coords_all[:, 1] >= y) & (coords_all[:, 1] < y + tile_size)

            num_p = np.sum(mask)
            if num_p < 100: continue

            fracture_p = np.sum(semantic_labels[mask] == 1)
            ratio = fracture_p / num_p if num_p > 0 else 0
            mem_mb = (num_p * (3 * 4 + 6 * 4 + 8 + 8)) / (1024 * 1024)

            if min_points <= num_p <= max_points and ratio >= min_fracture_ratio:
                # 只把有效的信息存起来，此时不读写磁盘
                valid_tiles.append({
                    'mask': mask,
                    'num_p': num_p,
                    'ratio': ratio,
                    'mem_mb': mem_mb
                })

    total_valid = len(valid_tiles)
    print(f"扫描完毕，共发现 {total_valid} 个有效数据块。")
    if total_valid == 0:
        print("未找到符合条件的数据块，请检查阈值设置。")
        return

    # 2. 第二阶段：精确计算切分数量并打乱顺序
    random.shuffle(valid_tiles)  # 保证随机性，但总量固定

    train_end = int(total_valid * split_ratios[0])
    val_end = train_end + int(total_valid * split_ratios[1])
    # 剩下的自动全归 test

    # 3. 第三阶段：执行实际的磁盘写入与建档
    print(f"========== 开始按严格比例写入磁盘 ==========")
    metadata_records = []

    for i, tile_info in enumerate(valid_tiles):
        # 根据精确的索引分配归属
        if i < train_end:
            split = 'train'
        elif i < val_end:
            split = 'val'
        else:
            split = 'test'

        scene_name = f"rock_scene_{i:04d}"
        scene_dir = out_root / split / scene_name
        scene_dir.mkdir(parents=True, exist_ok=True)

        mask = tile_info['mask']
        # 局部坐标去中心化
        tile_coords = coords_all[mask]
        tile_coords_local = tile_coords - tile_coords.mean(axis=0)

        # 保存 .npy
        np.save(scene_dir / "coord.npy", tile_coords_local)
        np.save(scene_dir / "color.npy", color_all[mask])
        np.save(scene_dir / "segment.npy", semantic_labels[mask])
        np.save(scene_dir / "instance.npy", instance_labels[mask])

        # 记录元数据
        metadata_records.append({
            'Scene_ID': scene_name,
            'Split': split,
            'Num_Points': tile_info['num_p'],
            'Fracture_Ratio': round(tile_info['ratio'], 4),
            'Memory_MB': round(tile_info['mem_mb'], 2),
            'Absolute_Path': str(scene_dir.absolute())
        })

    # 4. 生成数据报表与统计图
    print(f"========== 生成数据统计台账 ==========")
    df_meta = pd.DataFrame(metadata_records)
    csv_out_path = out_root / "dataset_metadata_inventory.csv"
    df_meta.to_csv(csv_out_path, index=False)

    # 绘制直方图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].hist(df_meta['Num_Points'], bins=30, color='skyblue', edgecolor='black')
    axes[0].set_title('Point Counts per Tile')
    axes[1].hist(df_meta['Fracture_Ratio'], bins=30, color='salmon', edgecolor='black')
    axes[1].set_title('Fracture Ratio Distribution')

    # 统计饼图
    split_counts = df_meta['Split'].value_counts()
    axes[2].pie(split_counts, labels=split_counts.index, autopct='%1.1f%%', colors=['#ff9999', '#66b3ff', '#99ff99'])
    axes[2].set_title('Dataset Split Ratio')

    plt.savefig(out_root / "dataset_statistics.png")

    print(f"\n✅ 数据集重构完成！")
    print(f"总生成有效数据块: {len(df_meta)}")
    print(f"Train/Val/Test 数量: {split_counts.to_dict()}")
    print(f"统计台账表格已保存至: {csv_out_path}")
    print(f"数据可视化图表已保存至: {out_root / 'dataset_statistics.png'}")


# 执行脚本 (确保在此之前清空旧的 data/rock_fracture 文件夹)
build_pointcept_dataset(
    csv_path=r"D:\Research\20250313_RockFractureSeg\Code\RockDiscontinuity\data\rock_data\Rock_GLS4_part1_localize_0.02m_ManualSegments_merged_label.csv",
    output_dir=r"D:\Research\20250313_RockFractureSeg\Code\DockerVolume\PointTransformerV3\Workspace1\PointTransformerV3\data\rock_fracture",
    tile_size=5.0,
    overlap=0.1,
    min_points=20000,
    max_points=300000,
    min_fracture_ratio=0.02,
    split_ratios=(0.8, 0.2, 0)  # 自动按 80% 训练，20% 验证，0% 测试分配
)
