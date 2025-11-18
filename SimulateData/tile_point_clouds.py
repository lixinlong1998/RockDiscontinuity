# -*- coding: utf-8 -*-
"""
批量将岩体点云按设定 cube_size 裁切成小块 PLY 的工具脚本
依赖: open3d, numpy, pandas, matplotlib, scipy

作者: (Your Name)
日期: 2025-11-12
"""

import os
import math
import time
import logging
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree


# =========================
# 全局常量与默认参数
# =========================
INPUT_DIR = r"E:\Database\_RockPoints\TSDK_Rockfall_IrregularClip"
OUTPUT_ROOT = r"E:\Database\_RockPoints\TSDK_Rockfall_IrregularClip\_tiles_output"  # 自动创建
CUBE_SIZE = 5         # 立方体边长(单位与点云一致)
MIN_POINTS = 100         # 每块最少点数
OVERLAP = 0.0            # 重叠长度(0 表示无重叠，>0 启用滑窗)
WRITE_BINARY = True      # 写出二进制PLY(更快更小)
SAVE_VIS = True          # 保存可视化图
RANDOM_SEED = 42         # 可视化采样时使用


# =========================
# 日志与计时
# =========================
class StepTimer:
    """用于记录某一步骤耗时的简单计时器"""
    def __init__(self, step_name: str):
        self.step_name = step_name
        self.start = None
        self.elapsed = None

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.elapsed = time.perf_counter() - self.start
        logging.info(f"[Time] {self.step_name}: {self.elapsed:.3f}s")


def InitLogger(output_dir: str) -> None:
    """
    功能: 初始化日志系统, 同时输出到控制台与文件
    实现: logging.basicConfig + FileHandler
    输入:
        output_dir(str): 输出目录, 日志文件写入其中
    输出: 无
    """
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "tiling.log")
    fmt = "%(asctime)s | %(levelname)s | %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(fmt))
    logging.getLogger().addHandler(file_handler)
    logging.info("===== Point Cloud Tiling Started =====")


# =========================
# 数据类
# =========================
@dataclass
class TileInfo:
    """单个块的元数据记录"""
    source_file: str
    tile_id: str
    i: int
    j: int
    k: int
    origin_x: float
    origin_y: float
    origin_z: float
    n_points: int
    bbox_min_x: float
    bbox_min_y: float
    bbox_min_z: float
    bbox_max_x: float
    bbox_max_y: float
    bbox_max_z: float
    save_path: str


# =========================
# 核心类
# =========================
class PointCloudTiler:
    """
    类功能简介:
        将输入目录中的多个PLY点云文件, 批量按设定的 cube_size 进行轴对齐立方体裁切,
        输出为多个小PLY与索引CSV, 并生成简单的可视化图表。

    实现思路详细描述:
        1) 逐文件读取点云, 保留原有的颜色与法向(若存在)；
        2) 若 overlap == 0: 使用体素索引法
           - 以点云整体最小坐标为原点 grid_min, 计算每个点的体素编号 ijk = floor((xyz - grid_min)/cube_size)
           - 获取唯一体素键, 再用“按组编号排序+切片”的方式快速得到每个体素内的点索引
           - 过滤点数不足的体素, 按 i,j,k 命名并保存PLY
        3) 若 overlap > 0: 使用滑动窗口
           - 步长 stride = cube_size - overlap
           - 构建 cKDTree, 对每个窗口原点执行盒选(先半径初筛再精确 AABB 判断), 保存符合阈值的窗口
        4) 记录全部块的元数据到 DataFrame, 写出 CSV 与可视化图

    输入变量:
        input_dir (str): 输入目录, 包含 .ply 文件
        output_root (str): 输出根目录, 会在其中为每个源文件创建子目录
        cube_size (float): 立方体边长
        min_points (int): 每块最少点数(过滤)
        overlap (float): 重叠长度, 0 表示无重叠
        write_binary (bool): 写出二进制PLY
        save_vis (bool): 是否保存统计图

    输出变量:
        无直接返回; 产生以下外部成果:
            - 小块PLY文件若干
            - index.csv 索引表
            - 统计图 (points_per_tile直方图, tile_centers_xy散点)
            - tiling.log 日志文件

    备注:
        - 为保证效率, 无重叠模式采用纯向量化+分组；重叠模式虽然通用, 但计算量较大。
        - 坐标单位请与 cube_size 保持一致。
    """

    def __init__(self,
                 input_dir: str,
                 output_root: str,
                 cube_size: float,
                 min_points: int,
                 overlap: float = 0.0,
                 write_binary: bool = True,
                 save_vis: bool = True):
        self.input_dir = input_dir
        self.output_root = output_root
        self.cube_size = float(cube_size)
        self.min_points = int(min_points)
        self.overlap = float(overlap)
        self.write_binary = bool(write_binary)
        self.save_vis = bool(save_vis)
        np.random.seed(RANDOM_SEED)

        if self.cube_size <= 0:
            raise ValueError("cube_size 必须为正数")
        if self.overlap < 0 or self.overlap >= self.cube_size:
            raise ValueError("overlap 必须满足 0 <= overlap < cube_size")

    # ---------- 工具函数 ----------

    def LoadPointCloud(self, ply_path: str) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        功能简介:
            读取PLY点云, 返回坐标与可选的颜色、法向数组。

        实现思路:
            使用 open3d.io.read_point_cloud 读取后, 将 points/colors/normals 转为 numpy。
            若缺失某字段则返回 None。

        输入:
            ply_path (str): 点云文件路径

        输出:
            xyz (np.ndarray, shape [N,3], dtype=float64)
            rgb (np.ndarray or None, shape [N,3], [0,1] 浮点; 若无颜色则为 None)
            nrm (np.ndarray or None, shape [N,3]; 若无法向则为 None)
        """
        with StepTimer(f"ReadPointCloud: {os.path.basename(ply_path)}"):
            pcd = o3d.io.read_point_cloud(ply_path)
        xyz = np.asarray(pcd.points, dtype=np.float64)
        rgb = np.asarray(pcd.colors, dtype=np.float64) if pcd.has_colors() else None
        nrm = np.asarray(pcd.normals, dtype=np.float64) if pcd.has_normals() else None
        logging.info(f"Loaded {ply_path} | points: {xyz.shape[0]} | has_color={rgb is not None} | has_normal={nrm is not None}")
        return xyz, rgb, nrm

    def SaveTile(self,
                 save_dir: str,
                 base_name: str,
                 tile_key: Tuple[int, int, int],
                 xyz_tile: np.ndarray,
                 rgb_tile: Optional[np.ndarray],
                 nrm_tile: Optional[np.ndarray],
                 write_binary: bool) -> str:
        """
        功能简介:
            将单个块写出为 PLY 文件。

        实现思路:
            将 numpy 数组装载到 open3d.geometry.PointCloud 并调用 write_point_cloud。
            写名包含 i,j,k 与点数。

        输入:
            save_dir (str): 输出子目录
            base_name (str): 源文件基名(不含扩展名)
            tile_key (Tuple[int,int,int]): (i,j,k) 或滑窗编号
            xyz_tile (np.ndarray [M,3]): 本块点坐标
            rgb_tile (np.ndarray/None [M,3]): 本块颜色
            nrm_tile (np.ndarray/None [M,3]): 本块法向
            write_binary (bool): 是否二进制写出

        输出:
            save_path (str): 写出的文件完整路径
        """
        i, j, k = tile_key
        m = xyz_tile.shape[0]
        pcd_tile = o3d.geometry.PointCloud()
        pcd_tile.points = o3d.utility.Vector3dVector(xyz_tile)
        if rgb_tile is not None:
            pcd_tile.colors = o3d.utility.Vector3dVector(rgb_tile)
        if nrm_tile is not None:
            pcd_tile.normals = o3d.utility.Vector3dVector(nrm_tile)

        file_name = f"{base_name}_tile_{i}_{j}_{k}_{m}.ply"
        save_path = os.path.join(save_dir, file_name)
        with StepTimer(f"WriteTile: {file_name} ({m} pts)"):
            o3d.io.write_point_cloud(save_path, pcd_tile,
                                     write_ascii=not write_binary,
                                     compressed=False,
                                     print_progress=False)
        return save_path

    def _GroupByVoxel_NoOverlap(self,
                                xyz: np.ndarray,
                                cube_size: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        功能简介:
            无重叠模式下, 根据 cube_size 将点云映射到体素, 返回分组信息。

        实现思路:
            - 计算 grid_min, 体素索引 ijk = floor((xyz - grid_min)/cube_size)
            - 使用 np.unique(return_inverse=True, return_counts=True) 得唯一体素键、每点所属组编号与每组元素数
            - 将“按组编号排序”得到每组的连续切片范围 (start_idx, end_idx)

        输入:
            xyz (np.ndarray [N,3]): 点云坐标
            cube_size (float): 体素边长

        输出:
            unique_ijk (np.ndarray [G,3], int64): 唯一体素键
            group_slices (np.ndarray [G,2], int64): 每组在 sorted 索引中的起止 [start,end) 下标
            sorted_point_idx (np.ndarray [N], int64): 按组编号排序后的原始点索引
            grid_min (np.ndarray [3], float64): 网格原点(最小坐标)
        """
        grid_min = xyz.min(axis=0, keepdims=False)
        # 计算体素索引
        ijk = np.floor((xyz - grid_min) / cube_size).astype(np.int64)  # [N,3]
        # 唯一化
        unique_ijk, inverse, counts = np.unique(ijk, axis=0, return_inverse=True, return_counts=True)
        # 按组编号排序
        sorted_idx = np.argsort(inverse, kind="mergesort")
        inv_sorted = inverse[sorted_idx]
        # 计算每个组的切片边界
        change = np.nonzero(np.diff(inv_sorted))[0] + 1
        starts = np.concatenate(([0], change))
        ends = np.concatenate((change, [inv_sorted.size]))
        group_slices = np.stack([starts, ends], axis=1)  # [G,2]
        logging.info(f"Voxel groups (no-overlap): {unique_ijk.shape[0]} groups")
        return unique_ijk, group_slices, sorted_idx, grid_min

    def _Windows_WithOverlap(self,
                             xyz: np.ndarray,
                             cube_size: float,
                             overlap: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        功能简介:
            构建重叠滑动窗口的原点列表, 并返回每个窗口的 AABB [min,max)。

        实现思路:
            - 计算整体 [min,max)
            - 步长 stride = cube_size - overlap
            - 三轴上用 np.arange 生成原点坐标, 组合成网格
            - 返回窗口原点列表与对应的 AABB

        输入:
            xyz (np.ndarray [N,3]): 点云坐标
            cube_size (float): 立方体边长
            overlap (float): 重叠长度

        输出:
            origins (np.ndarray [W,3]): 窗口原点
            aabbs (np.ndarray [W,2,3]): 每个窗口的 [min,max)
        """
        xyz_min = xyz.min(axis=0)
        xyz_max = xyz.max(axis=0)
        stride = cube_size - overlap
        # 为保证包含边界, 上限加一丝余量后取 np.arange
        xs = np.arange(xyz_min[0], xyz_max[0] + 1e-9, stride)
        ys = np.arange(xyz_min[1], xyz_max[1] + 1e-9, stride)
        zs = np.arange(xyz_min[2], xyz_max[2] + 1e-9, stride)
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        origins = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        aabbs_min = origins
        aabbs_max = origins + cube_size
        aabbs = np.stack([aabbs_min, aabbs_max], axis=1)  # [W,2,3]
        logging.info(f"Sliding windows (overlap): {origins.shape[0]} windows, stride={stride}")
        return origins, aabbs

    # ---------- 主流程 ----------

    def TileOneCloud(self, ply_path: str) -> List[TileInfo]:
        """
        功能简介:
            对单个 PLY 文件执行分块并保存结果, 返回块元数据列表。

        实现思路:
            - 读取点云与可选属性
            - 根据 overlap 选择无重叠体素分组或滑窗圈选
            - 过滤点数不足者, 写出PLY, 记录元数据
            - 返回 TileInfo 列表

        输入:
            ply_path (str): 点云文件路径

        输出:
            tiles (List[TileInfo]): 本文件产生的所有块的元数据
        """
        xyz, rgb, nrm = self.LoadPointCloud(ply_path)
        base_name = pathlib.Path(ply_path).stem
        save_dir = os.path.join(self.output_root, base_name)
        os.makedirs(save_dir, exist_ok=True)
        tiles: List[TileInfo] = []

        if self.overlap == 0.0:
            # 无重叠: 体素分组最高效
            with StepTimer("GroupByVoxel_NoOverlap"):
                unique_ijk, group_slices, sorted_idx, grid_min = self._GroupByVoxel_NoOverlap(xyz, self.cube_size)

            # 遍历每个体素组
            for g_idx, (s, e) in enumerate(group_slices):
                idxs = sorted_idx[s:e]
                if idxs.size < self.min_points:
                    continue
                ijk = unique_ijk[g_idx]
                i, j, k = int(ijk[0]), int(ijk[1]), int(ijk[2])

                xyz_tile = xyz[idxs]
                rgb_tile = rgb[idxs] if rgb is not None else None
                nrm_tile = nrm[idxs] if nrm is not None else None

                # 立方体原点
                origin = grid_min + ijk * self.cube_size
                bbox_min = origin
                bbox_max = origin + self.cube_size

                save_path = self.SaveTile(save_dir, base_name, (i, j, k), xyz_tile, rgb_tile, nrm_tile, self.write_binary)

                ti = TileInfo(
                    source_file=os.path.basename(ply_path),
                    tile_id=f"{base_name}_{i}_{j}_{k}",
                    i=i, j=j, k=k,
                    origin_x=float(origin[0]),
                    origin_y=float(origin[1]),
                    origin_z=float(origin[2]),
                    n_points=int(xyz_tile.shape[0]),
                    bbox_min_x=float(bbox_min[0]),
                    bbox_min_y=float(bbox_min[1]),
                    bbox_min_z=float(bbox_min[2]),
                    bbox_max_x=float(bbox_max[0]),
                    bbox_max_y=float(bbox_max[1]),
                    bbox_max_z=float(bbox_max[2]),
                    save_path=save_path
                )
                tiles.append(ti)
        else:
            # 有重叠: 滑窗 + KDTree 圈选
            origins, aabbs = self._Windows_WithOverlap(xyz, self.cube_size, self.overlap)
            with StepTimer("BuildKDTree"):
                kdt = cKDTree(xyz)

            # 半径初筛(对角线一半) + AABB精确过滤
            rad = math.sqrt(3.0) * (self.cube_size / 2.0)
            for w_idx, (o, ab) in enumerate(zip(origins, aabbs)):
                mid = o + self.cube_size * 0.5
                cand = kdt.query_ball_point(mid, r=rad)
                if len(cand) < self.min_points:
                    # 粗筛不足, 直接跳过
                    continue
                cand = np.asarray(cand, dtype=np.int64)
                xyz_cand = xyz[cand]
                mask = np.logical_and.reduce([
                    xyz_cand[:, 0] >= ab[0, 0], xyz_cand[:, 0] < ab[1, 0],
                    xyz_cand[:, 1] >= ab[0, 1], xyz_cand[:, 1] < ab[1, 1],
                    xyz_cand[:, 2] >= ab[0, 2], xyz_cand[:, 2] < ab[1, 2],
                ])
                idxs = cand[mask]
                if idxs.size < self.min_points:
                    continue

                i = int(round((o[0] - origins[0, 0]) / (self.cube_size - self.overlap)))
                j = int(round((o[1] - origins[0, 1]) / (self.cube_size - self.overlap)))
                k = int(round((o[2] - origins[0, 2]) / (self.cube_size - self.overlap)))

                xyz_tile = xyz[idxs]
                rgb_tile = rgb[idxs] if rgb is not None else None
                nrm_tile = nrm[idxs] if nrm is not None else None

                save_path = self.SaveTile(save_dir, base_name, (i, j, k), xyz_tile, rgb_tile, nrm_tile, self.write_binary)

                bbox_min = ab[0]
                bbox_max = ab[1]
                ti = TileInfo(
                    source_file=os.path.basename(ply_path),
                    tile_id=f"{base_name}_{i}_{j}_{k}",
                    i=i, j=j, k=k,
                    origin_x=float(o[0]),
                    origin_y=float(o[1]),
                    origin_z=float(o[2]),
                    n_points=int(xyz_tile.shape[0]),
                    bbox_min_x=float(bbox_min[0]),
                    bbox_min_y=float(bbox_min[1]),
                    bbox_min_z=float(bbox_min[2]),
                    bbox_max_x=float(bbox_max[0]),
                    bbox_max_y=float(bbox_max[1]),
                    bbox_max_z=float(bbox_max[2]),
                    save_path=save_path
                )
                tiles.append(ti)

        logging.info(f"File done: {os.path.basename(ply_path)} | tiles saved: {len(tiles)}")
        return tiles

    def VisualizeTileStats(self, df: pd.DataFrame, out_dir: str) -> None:
        """
        功能简介:
            输出统计图: (1) 每块点数直方图 (2) 块中心XY散点图。

        实现思路:
            使用 matplotlib 生成单图并保存 PNG，不设置特定颜色。

        输入:
            df (pd.DataFrame): 索引表
            out_dir (str): 输出目录

        输出:
            写出两张PNG图
        """
        if df.empty:
            logging.warning("索引表为空，跳过可视化")
            return
        os.makedirs(out_dir, exist_ok=True)
        # 1) 点数直方图
        plt.figure(figsize=(8, 5))
        plt.hist(df["n_points"], bins=50)
        plt.xlabel("Points per Tile")
        plt.ylabel("Count")
        plt.title("Histogram of Points per Tile")
        plt.tight_layout()
        fig1 = os.path.join(out_dir, "points_per_tile_hist.png")
        plt.savefig(fig1, dpi=150)
        plt.close()
        logging.info(f"Saved: {fig1}")

        # 2) 块中心 XY 散点
        cx = (df["bbox_min_x"].values + df["bbox_max_x"].values) * 0.5
        cy = (df["bbox_min_y"].values + df["bbox_max_y"].values) * 0.5
        # 若点数很多, 随机抽样显示
        n = cx.shape[0]
        max_show = 50000
        if n > max_show:
            sel = np.random.choice(n, max_show, replace=False)
            cx, cy = cx[sel], cy[sel]
        plt.figure(figsize=(6, 6))
        plt.scatter(cx, cy, s=1)
        plt.xlabel("Center X")
        plt.ylabel("Center Y")
        plt.title("Tile Centers (XY)")
        plt.axis("equal")
        plt.tight_layout()
        fig2 = os.path.join(out_dir, "tile_centers_xy.png")
        plt.savefig(fig2, dpi=150)
        plt.close()
        logging.info(f"Saved: {fig2}")

    def RunBatch(self) -> None:
        """
        功能简介:
            扫描输入目录中的 PLY 文件, 逐一裁切, 汇总写出 index.csv 与统计图。

        实现思路:
            - 列出输入目录下的 .ply 文件
            - 对每个文件调用 TileOneCloud
            - 将所有 TileInfo 汇总为 DataFrame 写出
            - 生成可视化图

        输入: 无(使用初始化参数)
        输出: 无(产生外部文件)
        """
        os.makedirs(self.output_root, exist_ok=True)
        InitLogger(self.output_root)

        # 枚举PLY文件
        ply_files = []
        for name in os.listdir(self.input_dir):
            if name.lower().endswith(".ply"):
                ply_files.append(os.path.join(self.input_dir, name))
            break
        ply_files.sort()
        logging.info(f"Found {len(ply_files)} ply files")

        all_tiles: List[TileInfo] = []
        for i, ply_path in enumerate(ply_files):
            logging.info(f"({i+1}/{len(ply_files)}) Processing: {os.path.basename(ply_path)}")
            try:
                with StepTimer("TileOneCloud"):
                    tiles = self.TileOneCloud(ply_path)
                all_tiles.extend(tiles)
            except Exception as e:
                logging.exception(f"Error processing {ply_path}: {e}")

        # 汇总写出索引
        rows = [t.__dict__ for t in all_tiles]
        df = pd.DataFrame(rows)
        csv_path = os.path.join(self.output_root, "index.csv")
        if not df.empty:
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            logging.info(f"Saved index CSV: {csv_path} (rows={len(df)})")
        else:
            logging.warning("No tiles generated; index.csv not written (empty).")

        # 可视化
        if self.save_vis:
            with StepTimer("VisualizeTileStats"):
                self.VisualizeTileStats(df, self.output_root)

        logging.info("===== All Done =====")


# =========================
# 入口
# =========================
if __name__ == "__main__":
    # 这里直接设定参数（按需修改）
    input_dir = INPUT_DIR
    output_root = os.path.join(OUTPUT_ROOT, f"cube_{CUBE_SIZE:g}_overlap_{OVERLAP:g}_minpts_{MIN_POINTS}")
    cube_size = CUBE_SIZE
    min_points = MIN_POINTS
    overlap = OVERLAP
    write_binary = WRITE_BINARY
    save_vis = SAVE_VIS

    tiler = PointCloudTiler(
        input_dir=input_dir,
        output_root=output_root,
        cube_size=cube_size,
        min_points=min_points,
        overlap=overlap,
        write_binary=write_binary,
        save_vis=save_vis
    )
    tiler.RunBatch()
