import os
import time
import logging
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np

# 读写点云 + 导出上色 ply
import open3d as o3d

# DBSCAN
from sklearn.cluster import DBSCAN

# 可视化
import matplotlib.pyplot as plt


@dataclass
class DbscanConfig:
    """
    功能简介:
        基于距离的 DBSCAN 点云聚类参数配置。

    实现思路:
        - mode='xyz'：以点间欧氏距离聚类
        - mode='range'：以点到参考点(默认原点)的量程距离聚类（1D DBSCAN）

    输入变量:
        mode (str): 'xyz' 或 'range'
        eps (float): DBSCAN 邻域半径
        min_samples (int): DBSCAN 最小邻域点数
        voxel_size (Optional[float]): 体素降采样尺寸(米)，None 表示不降采样
        range_origin (Tuple[float, float, float]): mode='range'时的参考点
        random_seed (int): 上色用随机种子
        out_dir (str): 输出目录

    输出变量:
        无（由 pipeline 的 Run() 返回 labels 与统计信息）
    """
    mode: str = "xyz"  # 'xyz' or 'range'
    eps: float = 0.15
    min_samples: int = 20
    voxel_size: Optional[float] = None
    range_origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    random_seed: int = 7
    out_dir: str = "./dbscan_out"


class StepTimer:
    """简单计时器：用于统计每步耗时。"""
    def __init__(self, logger: logging.Logger, step_name: str):
        self.logger = logger
        self.step_name = step_name
        self.t0 = None

    def __enter__(self):
        self.t0 = time.perf_counter()
        self.logger.info(f"[开始] {self.step_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        dt = time.perf_counter() - self.t0
        self.logger.info(f"[结束] {self.step_name} | 耗时: {dt:.3f} s")


class PointCloudDbscanPipeline:
    """
    功能简介:
        对点云按距离进行 DBSCAN 聚类，并输出统计、可视化与上色点云文件。

    实现思路(详细):
        1) 读取点云 -> numpy (N,3)
        2) 可选体素降采样
        3) 构造聚类特征:
           - xyz: (N,3)
           - range: (N,1) 其中 r = ||p - origin||
        4) sklearn DBSCAN 聚类 -> labels (N,)
        5) 统计每个簇的点数、质心、AABB
        6) 可视化簇规模分布；导出 colored.ply

    输入变量及类型/结构:
        point_path (str): 点云路径，支持 ply/pcd/xyz/csv
        config (DbscanConfig): 聚类配置

    输出变量及类型/结构:
        labels (np.ndarray): (N,) 每点簇标签，噪声为 -1
        stats (Dict[int, Dict]): 每个簇的统计信息（不含 -1）
    """
    def __init__(self, point_path: str, config: DbscanConfig):
        self.point_path = point_path
        self.cfg = config

        os.makedirs(self.cfg.out_dir, exist_ok=True)

        self.logger = logging.getLogger("PointCloudDbscan")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            ch.setFormatter(fmt)
            self.logger.addHandler(ch)

    def LoadPoints(self) -> np.ndarray:
        """读取点云为 numpy (N,3)。"""
        suffix = os.path.splitext(self.point_path)[1].lower()

        if suffix in [".ply", ".pcd", ".xyz", ".xyzn", ".xyzrgb"]:
            pcd = o3d.io.read_point_cloud(self.point_path)
            xyz = np.asarray(pcd.points, dtype=np.float64)
            return xyz

        if suffix == ".csv":
            # 为了最小依赖，这里用 numpy 读取；要求表头包含 X,Y,Z（大小写均可）
            data = np.genfromtxt(self.point_path, delimiter=",", names=True, encoding="utf-8")
            colnames = [c.lower() for c in data.dtype.names]
            def pick(name: str) -> str:
                idx = colnames.index(name)
                return data.dtype.names[idx]

            x = data[pick("x")]
            y = data[pick("y")]
            z = data[pick("z")]
            xyz = np.vstack([x, y, z]).T.astype(np.float64)
            return xyz

        raise ValueError(f"不支持的点云格式: {suffix}")

    def VoxelDownSample(self, xyz: np.ndarray, voxel_size: float) -> np.ndarray:
        """体素降采样。"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd_ds = pcd.voxel_down_sample(voxel_size=voxel_size)
        return np.asarray(pcd_ds.points, dtype=np.float64)

    def BuildFeatures(self, xyz: np.ndarray) -> np.ndarray:
        """构造 DBSCAN 输入特征矩阵。"""
        if self.cfg.mode == "xyz":
            return xyz

        if self.cfg.mode == "range":
            origin = np.asarray(self.cfg.range_origin, dtype=np.float64).reshape(1, 3)
            r = np.linalg.norm(xyz - origin, axis=1, keepdims=True)  # (N,1)
            return r

        raise ValueError("cfg.mode 必须是 'xyz' 或 'range'")

    def RunDbscan(self, feat: np.ndarray) -> np.ndarray:
        """执行 DBSCAN，返回 labels。"""
        db = DBSCAN(eps=self.cfg.eps, min_samples=self.cfg.min_samples, metric="euclidean", n_jobs=-1)
        labels = db.fit_predict(feat)
        return labels.astype(np.int32)

    def ComputeStats(self, xyz: np.ndarray, labels: np.ndarray) -> Dict[int, Dict]:
        """统计每个簇的信息（忽略噪声 -1）。"""
        stats: Dict[int, Dict] = {}
        valid_ids = [cid for cid in np.unique(labels) if cid != -1]

        for cid in valid_ids:
            idx = np.where(labels == cid)[0]
            pts = xyz[idx]
            center = pts.mean(axis=0)
            aabb_min = pts.min(axis=0)
            aabb_max = pts.max(axis=0)
            stats[int(cid)] = {
                "count": int(idx.size),
                "center_xyz": center,
                "aabb_min": aabb_min,
                "aabb_max": aabb_max,
            }
        return stats

    def SaveColoredPointCloud(self, xyz: np.ndarray, labels: np.ndarray, out_ply: str) -> None:
        """导出按簇上色的点云 ply（噪声为灰色）。"""
        rng = np.random.default_rng(self.cfg.random_seed)
        unique_labels = np.unique(labels)

        # 为每个簇分配随机颜色
        color_map: Dict[int, np.ndarray] = {}
        for cid in unique_labels:
            if cid == -1:
                color_map[int(cid)] = np.array([0.6, 0.6, 0.6], dtype=np.float64)  # 噪声灰色
            else:
                color_map[int(cid)] = rng.random(3)  # 0-1

        colors = np.zeros((xyz.shape[0], 3), dtype=np.float64)
        for cid, col in color_map.items():
            colors[labels == cid] = col

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(out_ply, pcd)

    def PlotClusterSizeHistogram(self, stats: Dict[int, Dict], out_png: str) -> None:
        """绘制簇规模直方图。"""
        sizes = [v["count"] for v in stats.values()]
        if len(sizes) == 0:
            self.logger.warning("没有有效簇（除噪声外）。跳过直方图绘制。")
            return

        plt.figure()
        plt.hist(sizes, bins=30)
        plt.xlabel("cluster size (#points)")
        plt.ylabel("frequency")
        plt.title("DBSCAN cluster size histogram")
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()

    def SaveStatsTxt(self, stats: Dict[int, Dict], out_txt: str) -> None:
        """保存簇统计到 txt。"""
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(f"mode={self.cfg.mode}, eps={self.cfg.eps}, min_samples={self.cfg.min_samples}\n")
            f.write(f"num_clusters={len(stats)}\n\n")
            for cid, info in sorted(stats.items(), key=lambda x: -x[1]["count"]):
                c = info["center_xyz"]
                mn = info["aabb_min"]
                mx = info["aabb_max"]
                f.write(
                    f"cluster {cid:04d} | count={info['count']}\n"
                    f"  center_xyz=({c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f})\n"
                    f"  aabb_min =({mn[0]:.4f}, {mn[1]:.4f}, {mn[2]:.4f})\n"
                    f"  aabb_max =({mx[0]:.4f}, {mx[1]:.4f}, {mx[2]:.4f})\n\n"
                )

    def Run(self) -> Tuple[np.ndarray, Dict[int, Dict]]:
        with StepTimer(self.logger, "Step 1: 读取点云"):
            xyz = self.LoadPoints()
            self.logger.info(f"点数: {xyz.shape[0]}")

        if self.cfg.voxel_size is not None and self.cfg.voxel_size > 0:
            with StepTimer(self.logger, "Step 2: 体素降采样"):
                xyz = self.VoxelDownSample(xyz, self.cfg.voxel_size)
                self.logger.info(f"降采样后点数: {xyz.shape[0]}")

        with StepTimer(self.logger, "Step 3: 构造聚类特征"):
            feat = self.BuildFeatures(xyz)
            self.logger.info(f"特征维度: {feat.shape}")

        with StepTimer(self.logger, "Step 4: DBSCAN 聚类"):
            labels = self.RunDbscan(feat)
            n_noise = int(np.sum(labels == -1))
            n_clusters = int(len(np.unique(labels)) - (1 if -1 in labels else 0))
            self.logger.info(f"簇数(不含噪声): {n_clusters} | 噪声点数: {n_noise}")

        with StepTimer(self.logger, "Step 5: 统计簇信息"):
            stats = self.ComputeStats(xyz, labels)

        with StepTimer(self.logger, "Step 6: 输出结果"):
            out_ply = os.path.join(self.cfg.out_dir, "clusters_colored.ply")
            out_png = os.path.join(self.cfg.out_dir, "cluster_size_hist.png")
            out_txt = os.path.join(self.cfg.out_dir, "cluster_stats.txt")

            self.SaveColoredPointCloud(xyz, labels, out_ply)
            self.PlotClusterSizeHistogram(stats, out_png)
            self.SaveStatsTxt(stats, out_txt)

            self.logger.info(f"已输出: {out_ply}")
            self.logger.info(f"已输出: {out_png}")
            self.logger.info(f"已输出: {out_txt}")

        return labels, stats


if __name__ == "__main__":
    # ========== 你只需要改这里的参数 ==========
    point_path = r"D:\Research\20250313_RockFractureSeg\Code\RockDiscontinuity\data\rock_data\Rock_GLS4_part1_localize_0.05m.ply"  # 或 .pcd/.xyz/.csv（csv需含X,Y,Z表头）

    cfg = DbscanConfig(
        mode="xyz",          # 'xyz'：点间距离聚类；'range'：到参考点距离聚类
        eps=0.16,            # 邻域半径（米）
        min_samples=20,      # 最小邻域点数
        voxel_size=0.05,     # 例如 0.05；None 表示不降采样
        range_origin=(0, 0, 0),
        out_dir=r"D:\Research\20250313_RockFractureSeg\Code\RockDiscontinuity\result/dbscan_out"
    )
    # ======================================

    pipeline = PointCloudDbscanPipeline(point_path=point_path, config=cfg)
    labels, stats = pipeline.Run()
