from typing import List, Tuple

import math
import numpy as np

try:
    from sklearn.neighbors import KDTree
except ImportError:
    KDTree = None

from .base import PlaneDetectionAlgorithm
from ..geometry import Discontinuity, Plane, Segment
from ..pointcloud import PointCloud
from ..logging_utils import Timer


class RegionGrowingDetector(PlaneDetectionAlgorithm):
    """
    功能简介:

        基于传统区域生长(RG)的平面检测算法骨架类.

    实现思路(当前版本):

        本版本最初仅提供类定义和接口占位, 方便在整体框架中被导入和调用,
        具体算法尚未实现, DetectPlanes 调用时会返回空列表。

        后续完整实现时, 典型流程可为:
            1) 预先估计点云法向量与曲率等特征;
            2) 选择种子点(如曲率较小、法向稳定的点);
            3) 对每个种子点, 基于法向差和点到当前平面的距离阈值进行区域生长,
               不断将满足条件的邻近点加入区域;
            4) 区域生长结束后, 对区域内点做一次最小二乘平面拟合,
               得到平面参数和均方根误差(RMSE);
            5) 过滤掉点数过少或质量过差的区域, 将合格区域封装为 Discontinuity.

        当前文件已在上述思路基础上补全了一个实际可运行的区域生长实现:
            - 仍然遵循上述 1)~5) 的整体流程;
            - 采用点级区域生长 + KDTree 邻域搜索(若可用), 并在结束后统一拟合平面;
            - 使用 min_region_size 过滤小区域, 并将 RMSE 作为 roughness 的占位指标之一。

    输入(构造函数参数):

        normal_angle_threshold: float

            生长过程中法向差的容许阈值(度).

        distance_threshold: float

            点到当前平面的最大容许距离.

        min_region_size: int

            被接受为结构面候选区域的最小点数.

    输出:

        DetectPlanes(point_cloud: PointCloud) -> List[Discontinuity]

            当前实现将返回检测到的结构面列表; 早期文档中“占位实现返回空列表”的描述
            在此仅保留为历史说明.
    """

    def __init__(
        self,
        normal_angle_threshold: float,
        distance_threshold: float,
        min_region_size: int
    ):
        super().__init__(name="RegionGrowing")

        self.normal_angle_threshold = normal_angle_threshold
        self.distance_threshold = distance_threshold
        self.min_region_size = min_region_size

        # 预计算法向角阈值的 cos, 以便快速判断法向差
        self._cos_normal_angle_threshold = math.cos(
            math.radians(self.normal_angle_threshold)
        )

        # 区域生长邻域搜索的半径比例因子: r = factor * distance_threshold
        # 该参数不对外暴露, 保持接口简洁
        self._neighbor_radius_factor = 3.0

    def _DetectImpl(self, point_cloud: PointCloud) -> List[Discontinuity]:
        """
        功能简介:
            在给定点云上执行基于区域生长的平面/结构面检测.

        实现思路(概要):
            1) 将点云转换为 numpy 坐标与法向数组, 并标记哪些点具有有效法向;
            2) 使用 KDTree(若可用)或简单距离计算构建邻域搜索结构;
            3) 在所有未访问且具有有效法向的点上依次启动区域生长:
               - 使用 BFS(队列)在空间邻域内扩展;
               - 同时满足: 法向夹角 < normal_angle_threshold 且
                         点到当前平面(初始为种子平面)距离 < distance_threshold
                 的邻近点被加入当前区域;
               - 为降低计算开销, 当前平面参数只在区域生长结束后统一拟合, 并在日志中记录 RMSE;
            4) 对每个区域的点做最小二乘平面拟合(_FitPlaneFromPoints);
            5) 过滤掉点数 < min_region_size 的区域;
            6) 为每个保留的区域构造 Segment 与 Discontinuity, 计算倾角/倾向,
               并将 RMSE 作为 roughness 占位指标.

        输入:
            point_cloud: PointCloud
                输入点云对象, 要求各点包含可用的 normal 属性(三维向量), 否则无法进行法向约束.

        输出:
            discontinuities: List[Discontinuity]
                检测到的结构面列表.
        """
        points = point_cloud.points
        num_points = len(points)

        if num_points == 0:
            self.logger.warning("RegionGrowing: input point cloud is empty, return [].")
            return []

        # 1) 构造坐标与法向数组
        coords, normals, normals_valid = self._BuildCoordsAndNormals(point_cloud)

        if not normals_valid.any():
            self.logger.warning(
                "RegionGrowing: no valid normals found in point cloud, cannot run region growing."
            )
            return []

        # 2) 构建邻域搜索结构
        with Timer("RegionGrowing.BuildNeighborSearcher", self.logger):
            neighbor_searcher = self._BuildNeighborSearcher(coords)

        # 3) 区域生长主循环
        visited = np.zeros(num_points, dtype=bool)
        discontinuities: List[Discontinuity] = []
        region_index = 0

        # 邻域搜索半径
        search_radius = self._neighbor_radius_factor * self.distance_threshold

        with Timer("RegionGrowing.RegionGrowAll", self.logger):
            for seed_idx in range(num_points):
                # 跳过已访问或无法向的点
                if visited[seed_idx]:
                    continue
                if not normals_valid[seed_idx]:
                    continue

                region_indices = self._GrowRegion(
                    seed_idx=seed_idx,
                    coords=coords,
                    normals=normals,
                    normals_valid=normals_valid,
                    neighbor_searcher=neighbor_searcher,
                    visited=visited,
                    search_radius=search_radius,
                    distance_threshold=self.distance_threshold,
                    cos_angle_thresh=self._cos_normal_angle_threshold,
                )

                if len(region_indices) < self.min_region_size:
                    # 区域过小, 丢弃
                    continue

                inlier_indices = np.array(region_indices, dtype=int)

                # 4) 对区域内点拟合平面
                plane = self._FitPlaneFromPoints(coords, inlier_indices)
                dip, dip_direction = self._ComputeDipAndDipDirection(plane)

                # 5) 封装为 Segment 与 Discontinuity
                segment = Segment(
                    plane=plane,
                    point_indices=region_indices,
                    trace_length=0.0,
                )

                discontinuity = Discontinuity(
                    segments=[segment],
                    dip=dip,
                    dip_direction=dip_direction,
                    roughness=plane.rmse,
                    algorithm_name=self.name,
                )

                discontinuities.append(discontinuity)

                self.logger.info(
                    "[RegionGrowing] region %d: points=%d, dip=%.2f, "
                    "dip_direction=%.2f, rmse=%.4f",
                    region_index,
                    len(region_indices),
                    dip,
                    dip_direction,
                    plane.rmse,
                )

                region_index += 1

        self.logger.info(
            "RegionGrowing finished: total %d discontinuities.", len(discontinuities)
        )

        return discontinuities

    def _BuildCoordsAndNormals(
        self,
        point_cloud: PointCloud,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        功能简介:
            将 PointCloud 中的点坐标和法向量转换为 numpy 数组形式,
            同时标记哪些点具有有效法向.

        实现思路:
            1) 为所有点构造 coords 数组, 形状 (N, 3);
            2) 对于 normal 不为 None 且长度非零的点, 进行单位化后写入 normals 数组,
               并将 normals_valid 对应位置设为 True;
            3) 其它点 normals 默认为 (0, 0, 0), normals_valid 为 False.

        输入:
            point_cloud: PointCloud
                输入点云对象.

        输出:
            coords: np.ndarray, shape (N, 3)
                所有点坐标数组.
            normals: np.ndarray, shape (N, 3)
                所有点法向数组(单位向量, 无效法向为 0).
            normals_valid: np.ndarray, shape (N,)
                布尔数组, 表示对应点是否具有有效法向.
        """
        points = point_cloud.points
        num_points = len(points)

        coords = np.zeros((num_points, 3), dtype=float)
        normals = np.zeros((num_points, 3), dtype=float)
        normals_valid = np.zeros(num_points, dtype=bool)

        for idx, p in enumerate(points):
            coords[idx, 0] = p.x
            coords[idx, 1] = p.y
            coords[idx, 2] = p.z

            if p.normal is None:
                continue

            nx, ny, nz = p.normal
            norm_len = math.sqrt(nx * nx + ny * ny + nz * nz)
            if norm_len < 1e-8:
                continue

            nx /= norm_len
            ny /= norm_len
            nz /= norm_len

            normals[idx, 0] = nx
            normals[idx, 1] = ny
            normals[idx, 2] = nz
            normals_valid[idx] = True

        self.logger.info(
            "RegionGrowing: built coords/normals for %d points, valid normals: %d.",
            num_points,
            int(normals_valid.sum()),
        )

        return coords, normals, normals_valid

    def _BuildNeighborSearcher(self, coords: np.ndarray):
        """
        功能简介:
            为区域生长构建邻域搜索结构.

        实现思路:
            1) 若 sklearn.neighbors.KDTree 可用, 使用 KDTree 以获得较高查询效率;
            2) 若不可用, 直接返回 coords, 后续在区域生长中使用暴力距离计算,
               但会在日志中提示可能的性能问题.

        输入:
            coords: np.ndarray, shape (N, 3)
                所有点的坐标数组.

        输出:
            neighbor_searcher:
                若 KDTree 可用, 返回 KDTree 对象;
                否则返回 coords 自身, 作为后续暴力搜索的输入.
        """
        if KDTree is not None:
            self.logger.info(
                "RegionGrowing: using sklearn.neighbors.KDTree for neighbor search."
            )
            return KDTree(coords, leaf_size=40)
        else:
            self.logger.warning(
                "RegionGrowing: scikit-learn is not available, "
                "falling back to brute-force neighbor search (may be slow)."
            )
            return coords

    def _QueryNeighbors(
        self,
        neighbor_searcher,
        coords: np.ndarray,
        center_index: int,
        search_radius: float,
    ) -> np.ndarray:
        """
        功能简介:
            查询给定中心点在指定半径内的邻近点索引.

        实现思路:
            1) 若 neighbor_searcher 具有 query_radius 方法(即 KDTree),
               直接调用其 query_radius 接口;
            2) 否则, 视 neighbor_searcher 为 coords 数组本身,
               通过逐点距离计算进行暴力查询。

        输入:
            neighbor_searcher:
                KDTree 对象或 coords 数组.
            coords: np.ndarray, shape (N, 3)
                所有点坐标数组.
            center_index: int
                查询中心点在 coords 中的索引.
            search_radius: float
                邻域搜索半径.

        输出:
            neighbor_indices: np.ndarray, shape (M,)
                满足距离约束的邻近点索引数组(包含 center_index 本身).
        """
        if hasattr(neighbor_searcher, "query_radius"):
            # KDTree 路径
            center = coords[center_index : center_index + 1]
            indices = neighbor_searcher.query_radius(center, r=search_radius)[0]
            return indices.astype(int)

        # 暴力路径
        center = coords[center_index]
        diff = coords - center
        dist = np.linalg.norm(diff, axis=1)
        indices = np.nonzero(dist <= search_radius)[0]
        return indices.astype(int)

    def _GrowRegion(
        self,
        seed_idx: int,
        coords: np.ndarray,
        normals: np.ndarray,
        normals_valid: np.ndarray,
        neighbor_searcher,
        visited: np.ndarray,
        search_radius: float,
        distance_threshold: float,
        cos_angle_thresh: float,
    ) -> List[int]:
        """
        功能简介:
            从给定种子点出发, 在空间与法向/距离约束下执行区域生长,
            返回该区域内所有点在全局点云中的索引.

        实现思路:
            1) 使用 BFS 维护待扩展队列 queue, 初始仅包含种子点;
            2) 当前平面初始以种子点的法向和位置定义:
               n = normals[seed_idx], d = -n·p_seed;
            3) 每次从队列取出一个点 idx, 在 search_radius 内查询其邻域点;
            4) 对每个候选点 j:
               - 若已访问或无效法向, 跳过;
               - 若 |n·normal_j| < cos_angle_thresh, 跳过(法向差过大);
               - 若 点到当前平面的距离 > distance_threshold, 跳过;
               - 否则接受该点: 标记 visited, 加入 region 与队列;
            5) 当前实现中, 平面参数在整个区域生长过程中保持为种子平面,
               以控制复杂度; 生长完成后再对整域点统一拟合高精度平面。

        输入:
            seed_idx: int
                区域生长的种子点索引.
            coords: np.ndarray, shape (N, 3)
                所有点的坐标数组.
            normals: np.ndarray, shape (N, 3)
                所有点的法向数组(单位向量).
            normals_valid: np.ndarray, shape (N,)
                各点法向是否有效的布尔数组.
            neighbor_searcher:
                KDTree 对象或 coords 数组.
            visited: np.ndarray, shape (N,)
                全局访问标记数组.
            search_radius: float
                邻域搜索半径.
            distance_threshold: float
                点到平面的最大容许距离.
            cos_angle_thresh: float
                法向夹角阈值的 cos 值(即 cos(max_angle)).

        输出:
            region_indices: List[int]
                区域内所有点在全局点云中的索引列表.
        """
        region_indices: List[int] = []
        queue: List[int] = []

        # 初始平面采用种子点的法向和位置
        n = normals[seed_idx]
        p0 = coords[seed_idx]
        d = -float(np.dot(n, p0))

        visited[seed_idx] = True
        region_indices.append(seed_idx)
        queue.append(seed_idx)

        while queue:
            current = queue.pop(0)

            neighbor_indices = self._QueryNeighbors(
                neighbor_searcher=neighbor_searcher,
                coords=coords,
                center_index=current,
                search_radius=search_radius,
            )

            for j in neighbor_indices:
                if visited[j]:
                    continue
                if not normals_valid[j]:
                    continue

                # 法向一致性约束
                dot_n = float(np.dot(n, normals[j]))
                if abs(dot_n) < cos_angle_thresh:
                    continue

                # 距离平面约束
                distance = abs(float(np.dot(n, coords[j]) + d))
                if distance > distance_threshold:
                    continue

                visited[j] = True
                region_indices.append(j)
                queue.append(j)

        return region_indices

    def _FitPlaneFromPoints(
        self,
        coords: np.ndarray,
        inlier_indices: np.ndarray,
    ) -> Plane:
        """
        功能简介:
            对给定内点集合进行最小二乘平面拟合, 构造 Plane 对象.

        实现思路:
            1) 根据 inlier_indices 取出点坐标子集 pts;
            2) 计算质心 centroid;
            3) 计算去中心化后的协方差矩阵, 做特征分解;
            4) 取最小特征值对应的特征向量作为法向 normal;
            5) 计算平面截距 d = -n·centroid;
            6) 计算所有内点到平面的有符号距离, 得到 RMSE.

        输入:
            coords: numpy.ndarray, shape (N, 3)
                所有点的坐标数组.
            inlier_indices: numpy.ndarray, shape (M,)
                属于该平面的内点在 coords 中的索引.

        输出:
            plane: Plane
                拟合得到的平面对象, 包含 normal, d, centroid, inlier_indices, rmse.
        """
        pts = coords[inlier_indices]
        centroid = pts.mean(axis=0)
        pts_centered = pts - centroid
        cov = (pts_centered.T @ pts_centered) / max(pts_centered.shape[0], 1)

        eigen_values, eigen_vectors = np.linalg.eigh(cov)
        min_index = int(np.argmin(eigen_values))
        normal = eigen_vectors[:, min_index]
        normal = normal / (np.linalg.norm(normal) + 1e-12)

        d = -float(np.dot(normal, centroid))

        distances = pts @ normal + d
        rmse = float(np.sqrt(np.mean(distances ** 2)))

        plane = Plane(
            normal=(float(normal[0]), float(normal[1]), float(normal[2])),
            d=d,
            centroid=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
            inlier_indices=inlier_indices.tolist(),
            rmse=rmse,
        )

        return plane

    def _ComputeDipAndDipDirection(self, plane: Plane) -> Tuple[float, float]:
        """
        【未经验证】功能简介:
            在假定坐标系为 x=东(East), y=北(North), z=上(Up) 的情况下,
            由平面法向估计结构面的倾角(dip)和倾向(dip_direction).

        【未经验证】实现思路:
            1) 取法向 n = (nx, ny, nz), 并单位化;
            2) 若 nz < 0, 则将法向翻转为 (-nx, -ny, -nz), 统一到上半球;
            3) 水平分量长度 h = sqrt(nx^2 + ny^2);
            4) 倾角 dip = atan2(h, |nz|) (弧度转度数);
            5) 倾向 dip_direction = atan2(nx, ny), 范围 (-180, 180],
               再平移到 [0, 360) 区间.

        输入:
            plane: Plane

        输出:
            dip: float
                倾角(度).
            dip_direction: float
                倾向(度, [0, 360)).
        """
        nx, ny, nz = plane.normal

        norm_len = math.sqrt(nx * nx + ny * ny + nz * nz) + 1e-12
        nx /= norm_len
        ny /= norm_len
        nz /= norm_len

        if nz < 0.0:
            nx = -nx
            ny = -ny
            nz = -nz

        horizontal = math.sqrt(nx * nx + ny * ny)
        dip = math.degrees(math.atan2(horizontal, abs(nz)))

        azimuth = math.degrees(math.atan2(nx, ny))
        if azimuth < 0.0:
            azimuth += 360.0

        return dip, azimuth
