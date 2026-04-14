import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
import random


def save_point_cloud_to_ply(filename, coords, colors, normals, segment, instance):
    """轻量级无依赖的 PLY 导出器"""
    num_points = coords.shape[0]
    colors_u8 = np.clip(colors * 255.0, 0, 255).astype(np.uint8)

    with open(filename, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("property float nx\nproperty float ny\nproperty float nz\n")
        f.write("property int class\nproperty int inst\nend_header\n")

        data = np.zeros((num_points, 11))
        data[:, 0:3] = coords
        data[:, 3:6] = colors_u8
        data[:, 6:9] = normals
        data[:, 9] = segment
        data[:, 10] = instance
        fmt = '%.6f %.6f %.6f %d %d %d %.6f %.6f %.6f %d %d'
        np.savetxt(f, data, fmt=fmt)


def build_pointcept_dataset(
        csv_path,
        output_dir,
        tile_size=5.0,
        overlap=0.1,
        min_points=20000,
        max_points=120000,
        min_fracture_ratio=0.02,
        split_ratios=(0.8, 0.2, 0.0),  # 完美支持包含 0 的比例
        export_ply=True
):
    print(f"========== 开始加载源数据 ==========")
    df = pd.read_csv(csv_path)
    out_root = Path(output_dir)

    # 初始化三个目录，即便 test 为空也会建一个空文件夹保持结构完整
    for split in ['train', 'val', 'test']:
        (out_root / split).mkdir(parents=True, exist_ok=True)

    coords_all = df[['X', 'Y', 'Z']].values.astype(np.float32)
    rgb_all = df[['R', 'G', 'B']].values.astype(np.float32) / 255.0
    normals_all = df[['nx', 'ny', 'nz']].values.astype(np.float32)
    color_feat_all = np.concatenate([rgb_all, normals_all], axis=1)

    instance_labels = df['Discontinuity_id'].values.astype(np.int64)
    semantic_labels = (instance_labels != -1).astype(np.int64)

    stride = tile_size * (1 - overlap)
    x_min, y_min, z_min = coords_all.min(axis=0)
    x_max, y_max, z_max = coords_all.max(axis=0)

    # ================= 第一阶段：扫描并记录 =================
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
                valid_tiles.append({'mask': mask, 'num_p': num_p, 'ratio': ratio, 'mem_mb': mem_mb})

    total_valid = len(valid_tiles)
    print(f"扫描完毕，共发现 {total_valid} 个有效数据块。")
    if total_valid == 0: return

    # ================= 第二阶段：精准洗牌与余数补偿切分 =================
    random.shuffle(valid_tiles)

    train_count = int(total_valid * split_ratios[0])
    val_count = int(total_valid * split_ratios[1])
    test_count = int(total_valid * split_ratios[2])

    # 修复截断误差：将丢失的余数补给占比最大的数据集
    remainder = total_valid - (train_count + val_count + test_count)
    if remainder > 0:
        if split_ratios[0] >= split_ratios[1] and split_ratios[0] >= split_ratios[2]:
            train_count += remainder
        elif split_ratios[1] >= split_ratios[2]:
            val_count += remainder
        else:
            test_count += remainder

    train_end = train_count
    val_end = train_end + val_count

    # ================= 第三阶段：写入磁盘与建档 =================
    print(f"========== 开始按严格比例写入磁盘 (.npy 与 .ply) ==========")
    metadata_records = []

    for i, tile_info in enumerate(valid_tiles):
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
        tile_coords_local = coords_all[mask] - coords_all[mask].mean(axis=0)

        np.save(scene_dir / "coord.npy", tile_coords_local)
        np.save(scene_dir / "color.npy", color_feat_all[mask])
        np.save(scene_dir / "segment.npy", semantic_labels[mask])
        np.save(scene_dir / "instance.npy", instance_labels[mask])

        if export_ply:
            save_point_cloud_to_ply(
                scene_dir / f"{scene_name}.ply",
                tile_coords_local, rgb_all[mask], normals_all[mask],
                semantic_labels[mask], instance_labels[mask]
            )

        metadata_records.append({
            'Scene_ID': scene_name, 'Split': split, 'Num_Points': tile_info['num_p'],
            'Fracture_Ratio': round(tile_info['ratio'], 4), 'Memory_MB': round(tile_info['mem_mb'], 2),
            'Absolute_Path': str(scene_dir.absolute())
        })

    # ================= 第四阶段：生成报告 =================
    df_meta = pd.DataFrame(metadata_records)
    csv_out_path = out_root / "dataset_metadata_inventory.csv"
    df_meta.to_csv(csv_out_path, index=False)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].hist(df_meta['Num_Points'], bins=30, color='skyblue', edgecolor='black')
    axes[0].set_title('Point Counts per Tile')
    axes[1].hist(df_meta['Fracture_Ratio'], bins=30, color='salmon', edgecolor='black')
    axes[1].set_title('Fracture Ratio Distribution')

    split_counts = df_meta['Split'].value_counts()
    axes[2].pie(split_counts, labels=split_counts.index, autopct='%1.1f%%', colors=['#ff9999', '#66b3ff', '#99ff99'])
    axes[2].set_title('Dataset Split Ratio')
    plt.savefig(out_root / "dataset_statistics.png")

    print(f"\n✅ 数据集重构完成！")
    print(f"精确划分数量: {split_counts.to_dict()}")


# 请替换为您的真实 CSV 路径后运行
build_pointcept_dataset(
    csv_path=r"D:\Research\20250313_RockFractureSeg\Code\RockDiscontinuity\data\rock_data\Rock_GLS4_part1_localize_0.02m_ManualSegments_merged_label.csv",
    output_dir=r"D:\Research\20250313_RockFractureSeg\Code\DockerVolume\PointTransformerV3\Workspace1\PointTransformerV3\Pointcept\data\rock_fracture",
    tile_size=5.0,
    overlap=0.1,
    min_points=20000,  # 适配训练的最小点数
    max_points=300000,  # 16G 显存防溢出硬指标
    min_fracture_ratio=0.02,  # 确保训练块里有结构面
    split_ratios=(0.8, 0.2, 0.0),  # 测试 test=0 的情况
)
