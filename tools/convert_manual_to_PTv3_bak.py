import pandas as pd
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from pathlib import Path


def preprocess_rock_csv(
        csv_path,
        output_dir,
        tile_size=5.0,
        overlap=0.1,
        min_points=10000,
        max_points=150000,
        min_fracture_ratio=0.05
):
    """
    针对岩体结构面的 PTv3 数据预处理脚本
    """
    print(f"开始加载数据: {csv_path}")
    df = pd.read_csv(csv_path)
    output_path = Path(output_dir)
    pth_dir = output_path / "pth_files"
    pth_dir.mkdir(parents=True, exist_ok=True)

    # 1. 提取原始数据
    coords_all = df[['X', 'Y', 'Z']].values.astype(np.float32)
    # 特征包含: RGB归一化 + 法向量
    feats_all = np.concatenate([
        df[['R', 'G', 'B']].values.astype(np.float32) / 255.0,
        df[['nx', 'ny', 'nz']].values.astype(np.float32)
    ], axis=1)

    # 标签处理
    instance_labels = df['Discontinuity_id'].values.astype(np.int64)
    semantic_labels = (instance_labels != -1).astype(np.int64)

    # 2. 计算分块步长
    stride = tile_size * (1 - overlap)
    x_min, y_min, z_min = coords_all.min(axis=0)
    x_max, y_max, z_max = coords_all.max(axis=0)

    stats = []
    valid_paths = []

    print("开始执行 3D 滑动窗口切块...")
    tile_idx = 0
    # 针对岩体，通常沿 X, Y 轴切块，Z 轴根据厚度决定是否需要切割（此处默认切 X,Y）
    for x in np.arange(x_min, x_max, stride):
        for y in np.arange(y_min, y_max, stride):
            mask = (coords_all[:, 0] >= x) & (coords_all[:, 0] < x + tile_size) & \
                   (coords_all[:, 1] >= y) & (coords_all[:, 1] < y + tile_size)

            num_p = np.sum(mask)
            if num_p < 100: continue  # 过滤极小碎片

            # 计算结构面占比
            fracture_p = np.sum(semantic_labels[mask] == 1)
            ratio = fracture_p / num_p if num_p > 0 else 0

            # 计算近似内存占用 (MB)
            # (coord: 3*4 + feat: 6*4 + sem: 8 + ins: 8) bytes per point
            mem_size = (num_p * (3 * 4 + 6 * 4 + 8 + 8)) / (1024 * 1024)

            # 存储统计信息
            stats.append({
                'id': tile_idx,
                'num_points': num_p,
                'fracture_ratio': ratio,
                'mem_mb': mem_size
            })

            # 满足阈值的块进行保存
            if min_points <= num_p <= max_points and ratio >= min_fracture_ratio:
                tile_coords = coords_all[mask]
                # 局部去中心化
                tile_coords_local = tile_coords - tile_coords.mean(axis=0)

                data_dict = {
                    "coord": torch.from_numpy(tile_coords_local),
                    "feat": torch.from_numpy(feats_all[mask]),
                    "segment": torch.from_numpy(semantic_labels[mask]),
                    "instance": torch.from_numpy(instance_labels[mask]),
                    "name": f"rock_tile_{tile_idx:04d}"
                }

                file_name = f"rock_tile_{tile_idx:04d}.pth"
                save_path = pth_dir / file_name
                torch.save(data_dict, save_path)
                valid_paths.append(str(save_path.absolute()))

            tile_idx += 1

    # 3. 生成统计直方图
    df_stats = pd.DataFrame(stats)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(df_stats['num_points'], bins=30, color='skyblue', edgecolor='black')
    axes[0].set_title('Distribution of Point Counts')

    axes[1].hist(df_stats['fracture_ratio'], bins=30, color='salmon', edgecolor='black')
    axes[1].set_title('Fracture Point Ratio')

    axes[2].hist(df_stats['mem_mb'], bins=30, color='lightgreen', edgecolor='black')
    axes[2].set_title('Estimated Memory Usage (MB)')

    plt.savefig(output_path / "data_statistics.png")
    print(f"统计直方图已保存至: {output_path / 'data_statistics.png'}")

    # 4. 保存路径列表
    with open(output_path / "valid_tiles_list.txt", "w") as f:
        for p in valid_paths:
            f.write(p + "\n")

    print(f"\n--- 处理汇总 ---")
    print(f"总切块数: {tile_idx}")
    print(f"符合筛选条件的块数: {len(valid_paths)}")
    print(f"路径清单保存在: {output_path / 'valid_tiles_list.txt'}")
    print(df_stats.describe())


# 执行转换 (请根据实际路径修改)
preprocess_rock_csv(
    csv_path=r"D:\Research\20250313_RockFractureSeg\Code\RockDiscontinuity\data\rock_data\Rock_GLS4_part1_localize_0.02m_ManualSegments_merged_label.csv",
    output_dir=r"D:\Research\20250313_RockFractureSeg\Code\DockerVolume\PointTransformerV3\Workspace1\PointTransformerV3\Pointcept\data\rock_fracture",
    tile_size=5.0,
    overlap=0.1,
    min_points=20000,  # 适配训练的最小点数
    max_points=300000,  # 16G 显存防溢出硬指标
    min_fracture_ratio=0.02  # 确保训练块里有结构面
)
