from typing import List

import numpy as np
from scipy.spatial import cKDTree

from .logging_utils import LoggerManager, Timer
from .geometry import Point, BoundingBox, Discontinuity


class PointCloud:
    """
    功能简介:
        作为整个框架的基础数据容器, 表示一个完整的点云数据集.

    实现思路:
        - 内部以 List[Point] 形式存储所有点;
        - 初始化时自动计算轴对齐包围盒(BoundingBox);
        - 预留空间索引(KDTree等)和法向估计、下采样等操作接口。
    """

    def __init__(self, points: List[Point]):
        self.logger = LoggerManager.GetLogger(self.__class__.__name__)
        self.points = points
        self.bounding_box = self._ComputeBoundingBox()
        self.spatial_index = None  # 预留 KDTree 等

    def _ComputeBoundingBox(self) -> BoundingBox:
        """
        功能简介:
            计算当前点云的轴对齐包围盒(AABB)。

        实现思路:
            遍历所有点, 分别在 x/y/z 三个坐标维度上取最小值和最大值。
        """
        min_x = min(p.x for p in self.points)
        min_y = min(p.y for p in self.points)
        min_z = min(p.z for p in self.points)
        max_x = max(p.x for p in self.points)
        max_y = max(p.y for p in self.points)
        max_z = max(p.z for p in self.points)
        return BoundingBox(min_x, min_y, min_z, max_x, max_y, max_z)

    def EstimateNormals(self, k_neighbor: int = 20, est_normals: bool = True, est_curvature: bool = True) -> None:
        """
        【未经验证】功能简介:
            使用 k 近邻方法为点云中每个点估计法向量, 并可选地计算基于
            PCA 特征值的局部曲率(即最小特征值占比).

        【未经验证】实现思路:
            1) 将所有点坐标整理为 (N,3) 的 numpy 数组 coords;
            2) 构建 cKDTree, 支持快速 k 近邻查询;
            3) 对每个点 i:
               - 查询其 k_neighbor 个邻近点索引 idxs;
               - 取邻域点坐标 pts, 计算质心 centroid;
               - 构造去中心化矩阵 pts_centered = pts - centroid;
               - 协方差矩阵 cov = (pts_centered^T @ pts_centered) / max(M,1);
               - 对 cov 做特征分解, 得到特征值和特征向量:
                   * 若 est_normals=True, 取最小特征值对应的特征向量为法向 normal,
                     并单位化后写入 Point.normal;
                   * 若 est_curvature=True, 用最小特征值与特征值和计算
                     曲率: curvature = lambda_min / (lambda1 + lambda2 + lambda3),
                     写入 Point.curvature;
               - 若某一开关为 False, 则对应属性不写入/不覆盖。
            4) 若 est_normals=False 且 est_curvature=False, 则直接返回, 不做任何计算。

        输入:
            k_neighbor: int
                每个点用于估计法向的邻近点数量, 建议 20~50 之间根据点云密度调整。
            est_normals: bool
                是否计算并写入法向量 Point.normal。
            est_curvature: bool
                是否计算并写入局部曲率 Point.curvature。

        输出:
            无(结果直接写入 self.points[i].normal 和/或 self.points[i].curvature)
        """
        # 若两个开关都关闭, 则不做任何计算, 直接返回
        if not est_normals and not est_curvature:
            self.logger.info(
                "EstimateNormals: est_normals=False 且 est_curvature=False, "
                "不执行任何法向/曲率估计, 直接返回."
            )
            return

        num_points = len(self.points)
        if num_points == 0:
            self.logger.warning("EstimateNormals: 点云为空, 跳过法向/曲率估计.")
            return

        if k_neighbor < 3:
            self.logger.warning(
                f"EstimateNormals: k_neighbor={k_neighbor} < 3, "
                f"自动调整为 3."
            )
            k_neighbor = 3

        # 将所有点坐标整理为 numpy 数组
        coords = np.array([[p.x, p.y, p.z] for p in self.points], dtype=float)

        with Timer(
                f"PointCloud.EstimateNormals(N={num_points}, k={k_neighbor}, "
                f"est_normals={est_normals}, est_curvature={est_curvature})",
                self.logger,
        ):
            # 构建 KDTree
            tree = cKDTree(coords)

            # 若点数少于 k_neighbor, 统一调整 k
            effective_k = min(k_neighbor, num_points)

            for i in range(num_points):
                # 查询当前点的 k 近邻(含自身)
                dists, idxs = tree.query(coords[i], k=effective_k)
                if np.isscalar(idxs):
                    # 只有一个点的退化情况
                    if est_normals:
                        self.points[i].normal = None
                    if est_curvature:
                        self.points[i].curvature = 0.0
                    continue

                pts = coords[idxs]
                m = pts.shape[0]
                if m < 3:
                    # 邻域点数不足以拟合平面
                    if est_normals:
                        self.points[i].normal = None
                    if est_curvature:
                        self.points[i].curvature = 0.0
                    continue

                # 计算质心与协方差矩阵
                centroid = pts.mean(axis=0)
                pts_centered = pts - centroid
                cov = (pts_centered.T @ pts_centered) / float(max(m, 1))

                # 特征分解
                try:
                    eigen_values, eigen_vectors = np.linalg.eigh(cov)
                except np.linalg.LinAlgError:
                    if est_normals:
                        self.points[i].normal = None
                    if est_curvature:
                        self.points[i].curvature = 0.0
                    continue

                # 取最小特征值对应的特征向量作为法向
                min_index = int(np.argmin(eigen_values))
                normal_vec = eigen_vectors[:, min_index]
                norm_len = np.linalg.norm(normal_vec)
                if norm_len < 1e-12:
                    if est_normals:
                        self.points[i].normal = None
                    if est_curvature:
                        self.points[i].curvature = 0.0
                    continue
                normal_vec = normal_vec / norm_len

                # 曲率: 最小特征值占比
                lambda_min = float(eigen_values[min_index])
                lambda_sum = float(np.sum(eigen_values)) + 1e-12
                curvature_val = lambda_min / lambda_sum

                # 根据开关写入属性
                if est_normals:
                    self.points[i].normal = (
                        float(normal_vec[0]),
                        float(normal_vec[1]),
                        float(normal_vec[2]),
                    )
                if est_curvature:
                    self.points[i].curvature = float(curvature_val)

            self.logger.info(
                f"EstimateNormals 完成: N={num_points}, k={effective_k}, "
                f"est_normals={est_normals}, est_curvature={est_curvature}."
            )

    def VoxelDownsample(self, voxel_size: float) -> "PointCloud":
        """
        功能简介:
            使用体素栅格下采样点云(当前为占位实现).

        实现思路:
            当前版本仅输出日志并返回自身, 后续可在此处实现真正的
            体素聚合逻辑(如按 voxel_size 将点云划分到 3D 网格,
            并在每个体素中选取代表点或计算质心)。

        输入:
            voxel_size: float
                体素边长, 单位与点云坐标单位一致。

        输出:
            downsampled_cloud: PointCloud
                下采样后的点云对象, 当前占位返回 self。
        """
        self.logger.info(
            f"VoxelDownsample(voxel_size={voxel_size}) is not implemented yet, "
            f"当前返回原始 PointCloud."
        )
        # TODO: 实现体素下采样, 当前先返回自身占位
        return self


class Chunk:
    """
    功能简介:
        表示一个大点云中的子区域(分块), 用于分块处理和并行计算.

    实现思路:
        - 当前版本只简单持有一个 PointCloud 引用和 chunk_id;
        - 后续可在此拓展: 记录分块内结构面、块体等结果, 支持并行处理。
    """

    def __init__(self, chunk_id: int, point_cloud: PointCloud):
        self.chunk_id = chunk_id
        self.point_cloud = point_cloud
        self.bounding_box = point_cloud.bounding_box
        self.discontinuities: List[Discontinuity] = []
