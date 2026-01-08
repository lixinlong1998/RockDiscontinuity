import os
import sys
import csv
import json
import inspect
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 路径设置：参考 demo.py，把当前目录加入搜索路径
# ---------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from src.rock_discon_extract.io_pointcloud import PointCloudIO
from src.rock_discon_extract.results_exporter import ResultsExporter
from src.rock_discon_extract.logging_utils import LoggerManager, Timer


def ComputeVoxelIds(
        points: np.ndarray,
        voxel_size: float,
        origin: np.ndarray
) -> np.ndarray:
    """
    功能简介:
        计算每个点所属的体素整数坐标 (ix, iy, iz)。

    实现思路:
        使用 axis-aligned voxel grid：
        voxel_id = floor((p - origin) / voxel_size)

    输入:
        points: np.ndarray
            形状 (N, 3)，float，点坐标。
        voxel_size: float
            体素边长(>0)。
        origin: np.ndarray
            形状 (3,)，float，体素网格原点，通常取 points.min(axis=0)。

    输出:
        voxel_ids: np.ndarray
            形状 (N, 3)，int64，每个点对应的 (ix, iy, iz)。
    """
    if voxel_size <= 0:
        raise ValueError("voxel_size 必须 > 0")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points 必须是形状 (N,3) 的数组")

    voxel_ids = np.floor((points - origin[None, :]) / voxel_size).astype(np.int64)
    return voxel_ids


def GroupPointIndicesByVoxel(
        voxel_ids: np.ndarray
) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    """
    功能简介:
        将点按体素 id 分组，输出每个体素的点索引列表。

    实现思路:
        1) np.unique(voxel_ids, axis=0, return_inverse=True, return_counts=True)
        2) 对 inverse 排序后用 split 切分出每个体素的点索引集合

    输入:
        voxel_ids: np.ndarray
            形状 (N,3) 的 int64 体素 id。

    输出:
        unique_voxels: np.ndarray
            形状 (M,3)，每个体素的 (ix,iy,iz)。
        groups: List[np.ndarray]
            长度 M，每个元素是该体素内点的全局索引数组。
        counts: np.ndarray
            形状 (M,)，每个体素内点数。
    """
    if voxel_ids.ndim != 2 or voxel_ids.shape[1] != 3:
        raise ValueError("voxel_ids 必须是形状 (N,3) 的数组")

    unique_voxels, inverse, counts = np.unique(
        voxel_ids, axis=0, return_inverse=True, return_counts=True
    )

    # 对体素标签排序，便于线性切分分组
    order = np.argsort(inverse)
    inverse_sorted = inverse[order]

    # 找到每个体素分组的切分点
    split_pos = np.nonzero(np.diff(inverse_sorted))[0] + 1
    groups = np.split(order, split_pos)

    return unique_voxels, groups, counts


def _CallExportToMeshlabPly(
        out_ply_path: str,
        vertices: np.ndarray,
        colors: Optional[np.ndarray]
) -> None:
    """
    功能简介:
        调用 ResultsExporter 内部的 _ExportToMeshlabPly 写出点云 ply（xyz + 可选RGB）。

    实现思路:
        由于不同版本可能参数名略有差异，这里用 inspect.signature 自适应拼装参数。

    输入:
        out_ply_path: str
            输出 ply 文件路径。
        vertices: np.ndarray
            (N,3) float 顶点坐标。
        colors: Optional[np.ndarray]
            (N,3) uint8 颜色；若 None 则不写颜色。

    输出:
        无（落盘写文件）。
    """
    if not hasattr(ResultsExporter, "_ExportToMeshlabPly"):
        raise AttributeError("ResultsExporter 中未找到 _ExportToMeshlabPly，无法按要求导出 PLY。")

    export_func = getattr(ResultsExporter, "_ExportToMeshlabPly")
    sig = inspect.signature(export_func)
    param_names = list(sig.parameters.keys())

    # 保证类型
    vertices = np.asarray(vertices, dtype=np.float64)
    if colors is not None:
        colors = np.asarray(colors, dtype=np.uint8)

    kwargs = {}

    # 兼容常见命名：filename / file_path
    if "filename" in param_names:
        kwargs["filename"] = out_ply_path
    elif "file_path" in param_names:
        kwargs["file_path"] = out_ply_path
    else:
        # 若第一个参数不是关键字命名，则退化到位置参数
        pass

    # vertices 参数名一般就是 vertices
    if "vertices" in param_names:
        kwargs["vertices"] = vertices

    # colors 可选
    if "colors" in param_names:
        kwargs["colors"] = colors

    # edges/faces 如果存在参数就显式传 None
    if "edges" in param_names:
        kwargs["edges"] = None
    if "faces" in param_names:
        kwargs["faces"] = None

    # 优先 kwargs 调用；不满足时再位置参数兜底
    try:
        if kwargs:
            export_func(**kwargs)
        else:
            # 位置参数兜底（按最常见顺序：filename, vertices, edges, faces, colors）
            export_func(out_ply_path, vertices, None, None, colors)
    except TypeError:
        # 再兜底一次：filename, vertices, colors
        export_func(out_ply_path, vertices, colors)


def PlotVoxelCountsHistogram(
        counts: np.ndarray,
        out_png_path: str
) -> None:
    """
    功能简介:
        绘制并保存体素点数分布直方图。

    实现思路:
        matplotlib hist 直接绘制，保存 png。

    输入:
        counts: np.ndarray
            (M,) 每个体素点数。
        out_png_path: str
            输出图像路径。

    输出:
        无（保存 png）。
    """
    counts = np.asarray(counts).astype(np.int64)
    plt.figure()
    plt.hist(counts, bins=50)
    plt.xlabel("points per voxel")
    plt.ylabel("voxel count")
    plt.title("Voxel point-count distribution")
    plt.tight_layout()
    plt.savefig(out_png_path, dpi=200)
    plt.close()


if __name__ == "__main__":
    # =========================
    # 用户参数：直接改这里
    # =========================
    point_path = r"D:\Research\20250313_RockFractureSeg\Code\RockDiscontinuity\data\rock_data\Rock_GLS4_part1_localize_0.02m.ply"  # 输入点云路径
    output_dir = r"D:\Research\20250313_RockFractureSeg\Code\RockDiscontinuity\result\cc_pycc_export\VoxelSplit"  # 指定输出文件夹（若冲突且不允许覆盖，会自动加时间戳）
    voxel_size = 1.0  # 体素边长（单位与点云坐标一致）
    min_points_per_voxel = 1  # 小于该点数的体素不导出
    overwrite_output_dir = False  # False：若目录已存在则新建带时间戳目录
    enable_plot = True  # 是否输出 voxel 点数直方图

    logger = LoggerManager.GetLogger("VoxelSplitExport")

    # -------------------------
    # 输出目录处理
    # -------------------------
    output_dir = os.path.abspath(output_dir)
    if os.path.exists(output_dir) and (not overwrite_output_dir):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{output_dir}_{ts}"
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(point_path))[0]
    voxels_dir = os.path.join(output_dir, f"{base_name}_voxels")
    os.makedirs(voxels_dir, exist_ok=True)

    index_csv_path = os.path.join(output_dir, f"{base_name}_voxel_index.csv")
    meta_json_path = os.path.join(output_dir, f"{base_name}_voxel_meta.json")
    hist_png_path = os.path.join(output_dir, f"{base_name}_voxel_counts_hist.png")

    # -------------------------
    # 1) 读取点云
    # -------------------------
    with Timer(f"ReadPointCloudArrays({os.path.basename(point_path)})", logger):
        points, normals, colors, extra_attrs = PointCloudIO.ReadPointCloudArrays(point_path)

    if points.size == 0:
        raise RuntimeError("输入点云为空，终止。")

    # -------------------------
    # 2) 体素编号与分组
    # -------------------------
    with Timer("ComputeVoxelIds + Grouping", logger):
        origin = points.min(axis=0)
        voxel_ids = ComputeVoxelIds(points, voxel_size, origin)
        unique_voxels, groups, counts = GroupPointIndicesByVoxel(voxel_ids)

    logger.info(f"总点数: {points.shape[0]}")
    logger.info(f"体素数量(未过滤): {unique_voxels.shape[0]}")
    logger.info(f"点数统计: min={counts.min()}, max={counts.max()}, mean={counts.mean():.2f}")

    # -------------------------
    # 3) 写 meta
    # -------------------------
    meta = {
        "point_path": os.path.abspath(point_path),
        "base_name": base_name,
        "voxel_size": float(voxel_size),
        "origin_xyz": origin.tolist(),
        "num_points": int(points.shape[0]),
        "num_voxels_raw": int(unique_voxels.shape[0]),
        "min_points_per_voxel": int(min_points_per_voxel),
        "has_normals": normals is not None,
        "has_colors": colors is not None,
        "extra_attrs_keys": list(extra_attrs.keys()) if isinstance(extra_attrs, dict) else [],
        "note": "PLY 导出使用 ResultsExporter._ExportToMeshlabPly：仅写 xyz + 可选RGB；不会写 normals/extra_attrs。",
    }
    with open(meta_json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # -------------------------
    # 4) 导出每个体素子点云（PLY）+ 索引表
    # -------------------------
    with Timer("ExportVoxelPLYs", logger):
        with open(index_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "voxel_ix", "voxel_iy", "voxel_iz",
                "num_points",
                "ply_path",
                "min_x", "min_y", "min_z",
                "max_x", "max_y", "max_z",
            ])

            exported = 0
            for k, idxs in enumerate(groups):
                n_k = int(idxs.shape[0])
                if n_k < min_points_per_voxel:
                    continue

                ix, iy, iz = unique_voxels[k].tolist()

                pts_k = points[idxs]
                cols_k = colors[idxs] if (colors is not None) else None

                # 子集包围盒
                bb_min = pts_k.min(axis=0)
                bb_max = pts_k.max(axis=0)

                out_ply_name = f"{base_name}_vx{ix}_vy{iy}_vz{iz}_n{n_k}.ply"
                out_ply_path = os.path.join(voxels_dir, out_ply_name)

                # 使用 ResultsExporter 内部 PLY writer 导出
                _CallExportToMeshlabPly(out_ply_path, pts_k, cols_k)

                writer.writerow([
                    ix, iy, iz,
                    n_k,
                    out_ply_path,
                    float(bb_min[0]), float(bb_min[1]), float(bb_min[2]),
                    float(bb_max[0]), float(bb_max[1]), float(bb_max[2]),
                ])
                exported += 1

    logger.info(f"体素子点云导出完成: {exported} 个 ply")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"索引表: {index_csv_path}")
    logger.info(f"元信息: {meta_json_path}")

    # -------------------------
    # 5) 可视化：点数直方图
    # -------------------------
    if enable_plot:
        with Timer("PlotVoxelCountsHistogram", logger):
            PlotVoxelCountsHistogram(counts, hist_png_path)
        logger.info(f"直方图: {hist_png_path}")
