from typing import List, Tuple
import math
import numpy as np

from .base import PlaneDetectionAlgorithm
from ..geometry import Discontinuity, Plane, Segment
from ..pointcloud import PointCloud
from ..logging_utils import Timer

# 顶部尝试导入, 失败时设为 None, 后面统一判断
try:
    import open3d as o3d
except ImportError:
    o3d = None

try:
    from sklearn.linear_model import RANSACRegressor, LinearRegression
except ImportError:
    RANSACRegressor = None
    LinearRegression = None


class RansacDetector(PlaneDetectionAlgorithm):
    """
    功能简介:
        基于 RANSAC 的平面检测算法, 支持多种实现路径(手动实现 / Open3D / scikit-learn).

    实现思路(概要):
        - 统一对外接口: DetectPlanes(point_cloud) -> List[Discontinuity]
        - 通过 impl_mode 控制内部具体实现:
            1) "manual": 使用 numpy 手写多平面 RANSAC (原始实现迁移到 _DetectManual)
            2) "open3d": 调用 open3d.geometry.PointCloud.segment_plane, 迭代提取多平面
            3) "sklearn": 调用 sklearn.linear_model.RANSACRegressor, 以 z = f(x, y) 形式做 RANSAC

    输入(构造函数参数):
        distance_threshold: float
            判定点是否为某平面内点的距离阈值(单位与坐标一致).
        angle_threshold: float
            目前版本暂未使用, 预留用于后续避免重复/近似平行平面.
        min_inliers: int
            接受一个平面的最小内点数量.
        max_iterations: int
            每一轮 RANSAC 搜索的最大迭代次数(传递给各实现).
        impl_mode: str
            指定具体实现路径:
                - "manual": 手写 numpy RANSAC
                - "open3d": 使用 Open3D 的 segment_plane
                - "sklearn": 使用 scikit-learn 的 RANSACRegressor

    输出:
        DetectPlanes 返回 List[Discontinuity], 每个 Discontinuity 包含至少一个 Segment 和一个 Plane.
    """

    def __init__(
            self,
            distance_threshold: float,
            angle_threshold: float,
            min_inliers: int,
            max_iterations: int = 1000,
            impl_mode: str = "manual"
    ):
        super().__init__(name="RANSAC")
        self.distance_threshold = distance_threshold
        self.angle_threshold = angle_threshold
        self.min_inliers = min_inliers
        self.max_iterations = max_iterations
        self.impl_mode = impl_mode.lower()  # 规范为小写字符串

    def _DetectImpl(self, point_cloud: PointCloud) -> List[Discontinuity]:
        """
        功能简介:
            根据 impl_mode 选择具体实现路径, 并调用对应内部方法.

        输入:
            point_cloud: PointCloud
                输入点云对象.

        输出:
            discontinuities: List[Discontinuity]
                检测到的结构面列表.
        """
        if self.impl_mode == "manual":
            self.logger.info("Use manual numpy RANSAC implementation.")
            return self._DetectManual(point_cloud)
        elif self.impl_mode == "open3d":
            self.logger.info("Use Open3D RANSAC implementation.")
            return self._DetectOpen3d(point_cloud)
        elif self.impl_mode == "sklearn":
            self.logger.info("Use scikit-learn RANSACRegressor implementation.")
            return self._DetectSklearn(point_cloud)
        else:
            self.logger.warning(
                f"Unknown impl_mode='{self.impl_mode}', fallback to manual implementation."
            )
            return self._DetectManual(point_cloud)

    # =========================
    # 实现路径 1: 手写 numpy RANSAC (原有实现迁移至此)
    # =========================

    def _DetectManual(self, point_cloud: PointCloud) -> List[Discontinuity]:
        """
        功能简介:
            使用 numpy 手写多平面 RANSAC 实现.

        实现思路(与之前版本一致):
            1) 将点云坐标整理为 N×3 的 numpy 数组 coords;
            2) 使用 remaining_indices 维护当前尚未属于任何平面的点索引;
            3) 外层 while 循环: 每次从 remaining_indices 中寻找一个最佳平面:
               - 在 [0, len(remaining_indices)) 范围内随机选择 3 个局部索引,
                 映射到全局索引后获取 3 个点, 拟合候选平面;
               - 若 3 点近似共线则跳过;
               - 计算 remaining_indices 上所有点到该平面的距离, 取绝对值;
               - 距离 <= distance_threshold 的点构成内点集合, 记录数量最多的一次;
            4) 若本轮没有找到满足 min_inliers 的平面, 则整体 RANSAC 结束;
            5) 对最佳内点集合调用 _FitPlaneFromPoints 进行最小二乘平面拟合;
            6) 根据平面法向计算倾角和倾向, 生成 Plane / Segment / Discontinuity 对象并加入结果列表;
            7) 将这些内点从 remaining_indices 中移除, 进入下一轮搜索.

        输入:
            point_cloud: PointCloud

        输出:
            discontinuities: List[Discontinuity]
        """
        self.name = self.name + "_manual"
        # 将点云坐标转换为 numpy 数组, 形状为 (N, 3)
        coords = np.array(
            [[p.x, p.y, p.z] for p in point_cloud.points],
            dtype=float
        )
        num_points = coords.shape[0]

        if num_points < self.min_inliers:
            self.logger.info(
                f"Point count ({num_points}) < min_inliers ({self.min_inliers}), "
                f"skip manual RANSAC."
            )
            return []

        remaining_indices = np.arange(num_points, dtype=int)
        discontinuities: List[Discontinuity] = []

        rng = np.random.default_rng()
        plane_index = 0

        while remaining_indices.shape[0] >= self.min_inliers:
            with Timer(f"RANSAC.Manual.FindPlane_{plane_index}", self.logger):
                best_inlier_indices = None  # type: ignore[assignment]

                for _ in range(self.max_iterations):
                    if remaining_indices.shape[0] < 3:
                        break

                    sample_local = rng.choice(
                        remaining_indices.shape[0],
                        size=3,
                        replace=False
                    )
                    sample_indices = remaining_indices[sample_local]

                    p0, p1, p2 = coords[sample_indices]

                    v1 = p1 - p0
                    v2 = p2 - p0
                    normal = np.cross(v1, v2)
                    norm_len = np.linalg.norm(normal)
                    if norm_len < 1e-8:
                        continue
                    normal = normal / norm_len

                    d = -float(np.dot(normal, p0))

                    candidate_points = coords[remaining_indices]
                    distances = np.abs(candidate_points @ normal + d)

                    inlier_mask = distances <= self.distance_threshold
                    inlier_indices = remaining_indices[inlier_mask]

                    if inlier_indices.shape[0] < self.min_inliers:
                        continue

                    if (best_inlier_indices is None or
                            inlier_indices.shape[0] > best_inlier_indices.shape[0]):
                        best_inlier_indices = inlier_indices

                if best_inlier_indices is None or \
                        best_inlier_indices.shape[0] < self.min_inliers:
                    self.logger.info(
                        "No more planes with enough inliers (manual), stop RANSAC loop."
                    )
                    break

                plane = self._FitPlaneFromPoints(coords, best_inlier_indices)
                dip, dip_direction = self._ComputeDipAndDipDirection(plane)

                segment = Segment(
                    plane=plane,
                    point_indices=best_inlier_indices.tolist(),
                    trace_length=0.0
                )

                discontinuity = Discontinuity(
                    segments=[segment],
                    dip=dip,
                    dip_direction=dip_direction,
                    roughness=plane.rmse,
                    algorithm_name=self.name
                )
                discontinuities.append(discontinuity)

                self.logger.info(
                    f"[manual] Detected plane {plane_index}: "
                    f"inliers={len(best_inlier_indices)}, "
                    f"dip={dip:.2f}, dip_direction={dip_direction:.2f}"
                )

                mask_keep = ~np.isin(remaining_indices, best_inlier_indices)
                remaining_indices = remaining_indices[mask_keep]

                plane_index += 1

        return discontinuities

    # =========================
    # 实现路径 2: Open3D RANSAC
    # =========================

    def _DetectOpen3d(self, point_cloud: PointCloud) -> List[Discontinuity]:
        """
        功能简介:
            使用 Open3D 提供的 segment_plane 实现多平面 RANSAC.

        实现思路:
            1) 将点云坐标整理为 numpy 数组 coords;
            2) 使用 remaining_mask 维护当前未解释点集合;
            3) 在循环中:
               - 构造一个只包含剩余点的 open3d.geometry.PointCloud;
               - 调用 segment_plane(distance_threshold, ransac_n=3, num_iterations=max_iterations)
                 得到平面模型 [a, b, c, d] 及局部 inlier 索引;
               - 将局部索引映射回全局索引, 若内点数 >= min_inliers, 则:
                   * 调用 _FitPlaneFromPoints 精细拟合平面;
                   * 构造 Segment / Discontinuity;
                   * 从 remaining_mask 中移除这些内点;
               - 否则终止循环.

        输入:
            point_cloud: PointCloud

        输出:
            discontinuities: List[Discontinuity]
        """
        if o3d is None:
            self.logger.warning("Open3D not available, fallback to manual.")
            return self._DetectManual(point_cloud)

        self.name = self.name + "_open3d"
        coords = np.array(
            [[p.x, p.y, p.z] for p in point_cloud.points],
            dtype=float
        )
        num_points = coords.shape[0]

        if num_points < self.min_inliers:
            self.logger.info(
                f"Point count ({num_points}) < min_inliers ({self.min_inliers}), "
                f"skip Open3D RANSAC."
            )
            return []

        remaining_mask = np.ones(num_points, dtype=bool)
        discontinuities: List[Discontinuity] = []
        plane_index = 0

        while remaining_mask.sum() >= self.min_inliers:
            with Timer(f"RANSAC.Open3D.FindPlane_{plane_index}", self.logger):
                remaining_indices = np.nonzero(remaining_mask)[0]
                pc_o3d = o3d.geometry.PointCloud()
                pc_o3d.points = o3d.utility.Vector3dVector(coords[remaining_indices])

                # Open3D RANSAC: 返回平面模型和局部内点索引
                plane_model, inlier_local = pc_o3d.segment_plane(
                    distance_threshold=self.distance_threshold,
                    ransac_n=3,
                    num_iterations=self.max_iterations
                )
                inlier_local = np.asarray(inlier_local, dtype=int)

                if inlier_local.size < self.min_inliers:
                    self.logger.info(
                        "No more planes with enough inliers (Open3D), stop loop."
                    )
                    break

                # 映射回全局索引
                inlier_indices = remaining_indices[inlier_local]

                # 用我们统一的最小二乘拟合, 保持与手写实现一致
                plane = self._FitPlaneFromPoints(coords, inlier_indices)
                dip, dip_direction = self._ComputeDipAndDipDirection(plane)

                segment = Segment(
                    plane=plane,
                    point_indices=inlier_indices.tolist(),
                    trace_length=0.0
                )
                discontinuity = Discontinuity(
                    segments=[segment],
                    dip=dip,
                    dip_direction=dip_direction,
                    roughness=plane.rmse,
                    algorithm_name=self.name
                )
                discontinuities.append(discontinuity)

                self.logger.info(
                    f"[open3d] Detected plane {plane_index}: "
                    f"inliers={len(inlier_indices)}, "
                    f"dip={dip:.2f}, dip_direction={dip_direction:.2f}"
                )

                # 更新剩余掩码
                remaining_mask[inlier_indices] = False
                plane_index += 1

        return discontinuities

    # =========================
    # 实现路径 3: scikit-learn RANSACRegressor
    # =========================

    def _DetectSklearn(self, point_cloud: PointCloud) -> List[Discontinuity]:
        """
        功能简介:
            使用 scikit-learn 的 RANSACRegressor 实现多平面 RANSAC.

        实现思路(说明):
            - 将平面近似为 z = a * x + b * y + c (即把 z 视为 x, y 的线性回归),
              用 RANSACRegressor 做鲁棒回归, 提取主要平面内点集合;
            - 对于每一轮:
                * 在剩余点上拟合 RANSACRegressor;
                * 通过 inlier_mask 得到内点索引;
                * 若内点数 >= min_inliers, 则用 _FitPlaneFromPoints 精细拟合平面,
                  并构造 Discontinuity;
                * 从剩余集合中移除这些内点;
            - 注意: 此实现假定平面可以较好地用 z = f(x, y) 表达, 对于接近垂直的平面
              适用性较差, 建议在坡面整体近似为 z(x, y) 的场景中使用.

        输入:
            point_cloud: PointCloud

        输出:
            discontinuities: List[Discontinuity]
        """
        if RANSACRegressor is None and LinearRegression is None:
            self.logger.warning("scikit-learn is not available, fallback to manual RANSAC.")
            return self._DetectManual(point_cloud)

        self.name = self.name + "_sklearn"
        coords = np.array(
            [[p.x, p.y, p.z] for p in point_cloud.points],
            dtype=float
        )
        num_points = coords.shape[0]

        if num_points < self.min_inliers:
            self.logger.info(
                f"Point count ({num_points}) < min_inliers ({self.min_inliers}), "
                f"skip sklearn RANSAC."
            )
            return []

        remaining_indices = np.arange(num_points, dtype=int)
        discontinuities: List[Discontinuity] = []
        plane_index = 0

        while remaining_indices.shape[0] >= self.min_inliers:
            with Timer(f"RANSAC.Sklearn.FindPlane_{plane_index}", self.logger):
                # 构造当前剩余点的特征与目标: X = [x, y], y = z
                X = coords[remaining_indices, 0:2]
                y = coords[remaining_indices, 2]

                # 构造 RANSAC 回归器
                base_estimator = LinearRegression()
                ransac = RANSACRegressor(
                    estimator=base_estimator,
                    min_samples=3,
                    residual_threshold=self.distance_threshold,
                    max_trials=self.max_iterations
                )

                try:
                    ransac.fit(X, y)
                except Exception as e:
                    self.logger.warning(
                        f"RANSACRegressor.fit failed with error: {e}, stop loop."
                    )
                    break

                inlier_mask = ransac.inlier_mask_
                if inlier_mask is None:
                    self.logger.info(
                        "No inlier_mask from sklearn RANSAC, stop loop."
                    )
                    break

                inlier_local = np.nonzero(inlier_mask)[0]
                if inlier_local.size < self.min_inliers:
                    self.logger.info(
                        "No more planes with enough inliers (sklearn), stop loop."
                    )
                    break

                # 映射回全局索引
                inlier_indices = remaining_indices[inlier_local]

                # 用统一方法拟合平面
                plane = self._FitPlaneFromPoints(coords, inlier_indices)
                dip, dip_direction = self._ComputeDipAndDipDirection(plane)

                segment = Segment(
                    plane=plane,
                    point_indices=inlier_indices.tolist(),
                    trace_length=0.0
                )
                discontinuity = Discontinuity(
                    segments=[segment],
                    dip=dip,
                    dip_direction=dip_direction,
                    roughness=plane.rmse,
                    algorithm_name=self.name
                )
                discontinuities.append(discontinuity)

                self.logger.info(
                    f"[sklearn] Detected plane {plane_index}: "
                    f"inliers={len(inlier_indices)}, "
                    f"dip={dip:.2f}, dip_direction={dip_direction:.2f}"
                )

                # 从剩余集合中移除内点
                mask_keep = ~np.isin(remaining_indices, inlier_indices)
                remaining_indices = remaining_indices[mask_keep]
                plane_index += 1

        return discontinuities

    # =========================
    # 公共辅助函数
    # =========================

    def _FitPlaneFromPoints(
            self,
            coords: np.ndarray,
            inlier_indices: np.ndarray
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
            rmse=rmse
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
