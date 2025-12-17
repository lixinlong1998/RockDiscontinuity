from typing import Dict, List, Tuple, Optional

import math
import numpy as np

from .base import PlaneDetectionAlgorithm
from ..geometry import Discontinuity, Plane, Segment
from ..pointcloud import PointCloud
from ..logging_utils import LoggerManager, Timer


class MoeDetector(PlaneDetectionAlgorithm):
    """
    功能简介:
        基于 MOE(Major Orientation Estimation) 的岩体点云结构面提取算法实现类。

    实现思路(与论文对应的三个阶段):
        1) 基于两级体素的近似共面体素聚类:
           - 使用规则体素划分点云;
           - 对每个“大体素”做 PCA, 结合特征值比与 MSE 判定是否近似共面;
           - 对“不共面但点数较多”的体素再细分为 2x2x2 子体素, 重复共面判定;
           - 得到一组高质量的共面体素, 作为后续生长单元。
        2) 基于高斯核的主方向估计(MOE):
           - 对每个共面体素, 由 Σ_xyz 通过一阶误差传播得到 Σ_{φ,θ};
           - 将体素法向映射至半球面, 在 (φ,θ) 空间使用二维高斯核做加权投票;
           - 在累加器上进行局部极大值检测, 得到若干主方向向量。
        3) 基于共面体素的区域生长提取结构面:
           - 以每个主方向为目标方向, 在共面体素集合中选取种子并进行 26 邻接的 BFS 生长;
           - 对每个连通区域内的点重新拟合平面, 计算倾角/倾向, 封装为 Discontinuity。

    备注:
        - 部分参数默认值需要根据具体工程数据做调参;
        - 算法名称固定为 "MOE", 便于与其他算法区分。
    """

    def __init__(
            self,
            voxel_size: float = 0.5,
            min_big_voxel_points: int = 50,
            min_small_voxel_points: int = 20,
            coplanar_lambda_ratio: float = 5.0,
            coplanar_mse_threshold: float = 0.01,
            accumulator_phi_bins: int = 90,
            accumulator_theta_bins: int = 180,
            kernel_sigma_factor: float = 2.0,
            weight_size_ratio: float = 0.5,
            seed_angle_threshold_deg: float = 10.0,
            grow_angle_threshold_deg: float = 12.0,
            grow_distance_threshold: float = 0.2,
            min_region_points: int = 200,
            normal_estimate_k: int = 30,
    ):
        super().__init__(name="MOE")

        self.logger = LoggerManager.GetLogger(self.__class__.__name__)

        # 体素与共面判据参数
        self.voxel_size = float(voxel_size)
        self.min_big_voxel_points = int(min_big_voxel_points)
        self.min_small_voxel_points = int(min_small_voxel_points)
        self.coplanar_lambda_ratio = float(coplanar_lambda_ratio)
        self.coplanar_mse_threshold = float(coplanar_mse_threshold)

        # 主方向估计参数
        self.accumulator_phi_bins = int(accumulator_phi_bins)
        self.accumulator_theta_bins = int(accumulator_theta_bins)
        self.kernel_sigma_factor = float(kernel_sigma_factor)
        self.weight_size_ratio = float(weight_size_ratio)

        # 区域生长参数
        self.seed_angle_threshold_deg = float(seed_angle_threshold_deg)
        self.grow_angle_threshold_deg = float(grow_angle_threshold_deg)
        self.grow_distance_threshold = float(grow_distance_threshold)
        self.min_region_points = int(min_region_points)

        # 法向估计参数
        self.normal_estimate_k = int(normal_estimate_k)

        # 预计算角度阈值对应 cos 值
        self._cos_seed_angle_threshold = math.cos(
            math.radians(self.seed_angle_threshold_deg)
        )
        self._cos_grow_angle_threshold = math.cos(
            math.radians(self.grow_angle_threshold_deg)
        )

        # 用于权重归一化的全局尺度
        self._mesh_size_for_weight: float = 1.0
        self._total_points_for_weight: int = 1

    # ------------------------------------------------------------------
    # 对外主入口: 基类 DetectPlanes 会调用此内部实现
    # ------------------------------------------------------------------
    def _DetectImpl(self, point_cloud: PointCloud) -> List[Discontinuity]:
        """
        功能简介:
            在给定点云上执行 MOE 主流程, 返回结构面列表。
        """
        if not point_cloud.points:
            self.logger.warning("MOE: input point cloud is empty.")
            return []

        # 1) 确保已有可靠法向
        with Timer("MOE.EnsureNormals", self.logger):
            coords, normals = self._EnsureNormals(point_cloud)

        # 2) 两级体素划分与共面体素聚类
        with Timer("MOE.BuildVoxelAndCoplanarClusters", self.logger):
            (
                voxel_centers,
                voxel_point_indices,
                voxel_covariances,
                voxel_normals,
            ) = self._ClusterCoplanarVoxels(coords)

        if len(voxel_centers) == 0:
            self.logger.warning("MOE: no coplanar voxels found, abort.")
            return []

        # 3) 主方向估计
        with Timer("MOE.MajorOrientationEstimation", self.logger):
            major_orientations = self._EstimateMajorOrientations(
                voxel_centers, voxel_covariances, voxel_normals
            )

        if len(major_orientations) == 0:
            self.logger.warning("MOE: no major orientations detected, abort.")
            return []

        # 4) 区域生长提取结构面
        with Timer("MOE.RegionGrowing", self.logger):
            discontinuities = self._RegionGrowOnVoxels(
                coords,
                voxel_centers,
                voxel_point_indices,
                voxel_normals,
                major_orientations,
            )

        return discontinuities

    # ------------------------------------------------------------------
    # 法向与基础数据准备
    # ------------------------------------------------------------------
    def _EnsureNormals(
            self, point_cloud: PointCloud
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        功能简介:
            从 PointCloud 中提取坐标与法向数组, 若法向缺失则调用
            PointCloud.EstimateNormals 进行估计。

        输出:
            coords: (N,3) float64
            normals: (N,3) float64, 已单位化, 若个别点无法估计则用 NaN 填充。
        """
        points = point_cloud.points
        num_points = len(points)

        coords = np.zeros((num_points, 3), dtype=np.float64)
        normals = np.full((num_points, 3), np.nan, dtype=np.float64)

        valid_normal_count = 0
        for i, p in enumerate(points):
            coords[i, 0] = p.x
            coords[i, 1] = p.y
            coords[i, 2] = p.z
            if p.normal is not None:
                nx, ny, nz = p.normal
                n = np.array([nx, ny, nz], dtype=np.float64)
                norm = np.linalg.norm(n)
                if norm > 1e-8:
                    normals[i, :] = n / norm
                    valid_normal_count += 1

        if valid_normal_count < int(0.8 * num_points):
            self.logger.info(
                f"MOE: only {valid_normal_count}/{num_points} points have normals, "
                f"re-estimating normals with k={self.normal_estimate_k}."
            )
            point_cloud.EstimateNormals(
                k_neighbor=self.normal_estimate_k,
                est_normals=True,
                est_curvature=False,
            )
            for i, p in enumerate(point_cloud.points):
                if p.normal is not None:
                    n = np.asarray(p.normal, dtype=np.float64)
                    norm = np.linalg.norm(n)
                    if norm > 1e-8:
                        normals[i, :] = n / norm
                else:
                    normals[i, :] = np.nan

        return coords, normals

    # ------------------------------------------------------------------
    # 体素聚类阶段: 两级体素 + 共面检测
    # ------------------------------------------------------------------
    def _ClusterCoplanarVoxels(
            self, coords: np.ndarray
    ) -> Tuple[
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
    ]:
        """
        功能简介:
            对点云进行两级体素划分, 并根据特征值比与 MSE 判定近似共面体素。

        输出:
            voxel_centers: List[(3,)] 体素中心坐标
            voxel_point_indices: List[np.ndarray] 各体素内点的全局索引
            voxel_covariances: List[(3,3)] 协方差矩阵
            voxel_normals: List[(3,)] 局部平面法向
        """
        num_points = coords.shape[0]
        if num_points == 0:
            return [], [], [], []

        # 整体包围盒与尺度
        min_xyz = coords.min(axis=0)
        max_xyz = coords.max(axis=0)
        mesh_size = float(np.max(max_xyz - min_xyz))

        voxel_map: Dict[Tuple[int, int, int], List[int]] = {}
        inv_size = 1.0 / self.voxel_size

        for idx in range(num_points):
            x, y, z = coords[idx]
            i = int(math.floor((x - min_xyz[0]) * inv_size))
            j = int(math.floor((y - min_xyz[1]) * inv_size))
            k = int(math.floor((z - min_xyz[2]) * inv_size))
            key = (i, j, k)
            voxel_map.setdefault(key, []).append(idx)

        coplanar_centers: List[np.ndarray] = []
        coplanar_point_indices: List[np.ndarray] = []
        coplanar_covariances: List[np.ndarray] = []
        coplanar_normals: List[np.ndarray] = []

        # 工具函数: 判定一个点集是否共面
        def _test_coplanarity(
                point_indices: List[int],
        ) -> Optional[
            Tuple[np.ndarray, np.ndarray, np.ndarray, float]
        ]:
            if len(point_indices) < 3:
                return None
            pts = coords[point_indices]
            centroid = pts.mean(axis=0)
            pts_centered = pts - centroid
            cov = (pts_centered.T @ pts_centered) / float(
                max(len(point_indices), 1)
            )
            try:
                eigen_vals, eigen_vecs = np.linalg.eigh(cov)
            except np.linalg.LinAlgError:
                return None
            order = np.argsort(eigen_vals)
            eigen_vals = eigen_vals[order]
            eigen_vecs = eigen_vecs[:, order]
            lambda1, lambda2, lambda3 = eigen_vals

            # 特征值比约束: λ2 > Sα * λ1
            if lambda1 <= 0.0 or lambda2 <= self.coplanar_lambda_ratio * lambda1:
                return None

            normal = eigen_vecs[:, 0]
            normal_norm = np.linalg.norm(normal)
            if normal_norm < 1e-8:
                return None
            normal = normal / normal_norm

            d = -float(normal.dot(centroid))
            distances = pts @ normal + d
            mse = float(np.mean(distances * distances))
            if mse > self.coplanar_mse_threshold:
                return None

            return centroid, cov, normal, mse

        # 一级体素
        for key, idx_list in voxel_map.items():
            if len(idx_list) < self.min_big_voxel_points:
                continue

            test_result = _test_coplanarity(idx_list)
            if test_result is not None:
                centroid, cov, normal, _ = test_result
                coplanar_centers.append(centroid)
                coplanar_point_indices.append(np.asarray(idx_list, dtype=int))
                coplanar_covariances.append(cov)
                coplanar_normals.append(normal)
            else:
                # 不共面但点数仍较多 -> 二级体素再检测
                if len(idx_list) < 2 * self.min_small_voxel_points:
                    continue

                sub_map: Dict[Tuple[int, int, int], List[int]] = {}

                base_center = np.array(
                    [
                        min_xyz[0] + (key[0] + 0.5) * self.voxel_size,
                        min_xyz[1] + (key[1] + 0.5) * self.voxel_size,
                        min_xyz[2] + (key[2] + 0.5) * self.voxel_size,
                    ],
                    dtype=np.float64,
                )

                for idx in idx_list:
                    x, y, z = coords[idx]
                    dx = x - base_center[0]
                    dy = y - base_center[1]
                    dz = z - base_center[2]
                    si = 0 if dx < 0 else 1
                    sj = 0 if dy < 0 else 1
                    sk = 0 if dz < 0 else 1
                    sub_key = (si, sj, sk)
                    sub_map.setdefault(sub_key, []).append(idx)

                for sub_indices in sub_map.values():
                    if len(sub_indices) < self.min_small_voxel_points:
                        continue
                    test_result = _test_coplanarity(sub_indices)
                    if test_result is None:
                        continue
                    centroid, cov, normal, _ = test_result
                    coplanar_centers.append(centroid)
                    coplanar_point_indices.append(
                        np.asarray(sub_indices, dtype=int)
                    )
                    coplanar_covariances.append(cov)
                    coplanar_normals.append(normal)

        if len(coplanar_centers) == 0:
            return [], [], [], []

        self.logger.info(
            f"MOE: found {len(coplanar_centers)} coplanar voxels "
            f"from {len(voxel_map)} primary voxels."
        )

        self._mesh_size_for_weight = mesh_size
        self._total_points_for_weight = num_points

        return (
            coplanar_centers,
            coplanar_point_indices,
            coplanar_covariances,
            coplanar_normals,
        )

    # ------------------------------------------------------------------
    # 主方向估计阶段: 高斯核 + 半球累加器
    # ------------------------------------------------------------------
    def _EstimateMajorOrientations(
            self,
            voxel_centers: List[np.ndarray],
            voxel_covariances: List[np.ndarray],
            voxel_normals: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        功能简介:
            在 (φ,θ) 半球空间上使用二维高斯核投票, 提取若干主方向。
        """
        num_voxels = len(voxel_normals)
        if num_voxels == 0:
            return []

        phi_bins = self.accumulator_phi_bins
        theta_bins = self.accumulator_theta_bins
        accumulator = np.zeros((phi_bins, theta_bins), dtype=np.float64)

        d_phi = math.pi / float(phi_bins)  # φ ∈ [0, π]
        d_theta = 2.0 * math.pi / float(theta_bins)  # θ ∈ [0, 2π)

        # 权重参数
        wa = float(self.weight_size_ratio)
        wd = 1.0 - wa
        voxel_volume = self.voxel_size ** 3
        mesh_volume = (
            self._mesh_size_for_weight ** 3
            if self._mesh_size_for_weight > 0
            else 1.0
        )

        for idx in range(num_voxels):
            n = np.asarray(voxel_normals[idx], dtype=np.float64)
            if n[2] < 0.0:
                n = -n
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-8:
                continue
            n /= n_norm

            # 法向 -> 球坐标
            phi = math.acos(max(min(n[2], 1.0), -1.0))
            theta = math.atan2(n[1], n[0])
            if theta < 0.0:
                theta += 2.0 * math.pi

            phi_idx = int(phi / d_phi)
            theta_idx = int(theta / d_theta)
            phi_idx = max(0, min(phi_bins - 1, phi_idx))
            theta_idx = max(0, min(theta_bins - 1, theta_idx))

            cov_xyz = voxel_covariances[idx]

            # 由 Σ_xyz 推导 Σ_{φ,θ}
            px, py, pz = n[0], n[1], n[2]
            rho = 1.0
            w = max(px * px + py * py, 1e-12)

            J = np.array(
                [
                    [
                        px * pz / math.sqrt(w * rho * rho),
                        py * pz / math.sqrt(w * rho * rho),
                        -math.sqrt(w) / rho,
                    ],
                    [-py / w, px / w, 0.0],
                ],
                dtype=np.float64,
            )

            cov_phitheta = J @ cov_xyz @ J.T
            cov_phitheta += np.eye(2) * 1e-12

            try:
                eig_vals, eig_vecs = np.linalg.eigh(cov_phitheta)
            except np.linalg.LinAlgError:
                continue

            min_idx = int(np.argmin(eig_vals))
            lambda_min = float(eig_vals[min_idx])
            std_dev = math.sqrt(max(lambda_min, 1e-12))
            v_min = eig_vecs[:, min_idx]

            # 阈值: 在主轴 2σ 位置的高斯值
            delta = 2.0 * std_dev * v_min
            threshold = self._Gaussian2D(delta, cov_phitheta)

            sigma_max = math.sqrt(float(np.max(eig_vals)))
            win_phi = max(1, int(self.kernel_sigma_factor * sigma_max / d_phi))
            win_theta = max(1, int(self.kernel_sigma_factor * sigma_max / d_theta))

            # 体素权重(尺寸 + 点数占比的简化形式)
            w_size = voxel_volume / mesh_volume
            w_count = 1.0
            w_k = wa * w_size + wd * w_count

            cov_inv = np.linalg.inv(cov_phitheta)
            det_cov = float(np.linalg.det(cov_phitheta))
            norm_factor = 1.0 / (
                    2.0 * math.pi * math.sqrt(max(det_cov, 1e-18))
            )

            for d_i in range(-win_phi, win_phi + 1):
                cur_phi_idx = phi_idx + d_i
                if cur_phi_idx < 0 or cur_phi_idx >= phi_bins:
                    continue
                cur_phi = (cur_phi_idx + 0.5) * d_phi

                for d_j in range(-win_theta, win_theta + 1):
                    cur_theta_idx = (theta_idx + d_j) % theta_bins
                    cur_theta = (cur_theta_idx + 0.5) * d_theta

                    delta_vec = np.array(
                        [cur_phi - phi, cur_theta - theta], dtype=np.float64
                    )
                    g_val = self._Gaussian2D(
                        delta_vec,
                        cov_phitheta,
                        cov_inv=cov_inv,
                        norm_factor=norm_factor,
                    )
                    if g_val < threshold:
                        continue
                    accumulator[cur_phi_idx, cur_theta_idx] += w_k * g_val

        # 峰值检测
        major_orientations: List[np.ndarray] = []
        window_size = 5
        half_w = window_size // 2

        global_max = float(accumulator.max()) if accumulator.size > 0 else 0.0
        if global_max <= 0.0:
            return []

        min_peak_value = 0.1 * global_max

        for i in range(phi_bins):
            for j in range(theta_bins):
                center_val = accumulator[i, j]
                if center_val < min_peak_value:
                    continue

                i_min = max(0, i - half_w)
                i_max = min(phi_bins, i + half_w + 1)

                # θ 方向视作环状, 简化处理
                local_block = []
                for ii in range(i_min, i_max):
                    for jj in range(j - half_w, j + half_w + 1):
                        local_block.append(
                            accumulator[ii, jj % theta_bins]
                        )
                local_max = max(local_block)

                if center_val < local_max:
                    continue

                phi = (i + 0.5) * d_phi
                theta = (j + 0.5) * d_theta
                nx = math.sin(phi) * math.cos(theta)
                ny = math.sin(phi) * math.sin(theta)
                nz = math.cos(phi)
                n_vec = np.array([nx, ny, nz], dtype=np.float64)
                major_orientations.append(n_vec)

        self.logger.info(
            f"MOE: detected {len(major_orientations)} major orientations."
        )
        return major_orientations

    def _Gaussian2D(
            self,
            delta: np.ndarray,
            cov: np.ndarray,
            cov_inv: Optional[np.ndarray] = None,
            norm_factor: Optional[float] = None,
    ) -> float:
        """
        功能简介:
            计算二维高斯核值。
        """
        if cov_inv is None:
            cov_inv = np.linalg.inv(cov)
        if norm_factor is None:
            det_cov = float(np.linalg.det(cov))
            norm_factor = 1.0 / (
                    2.0 * math.pi * math.sqrt(max(det_cov, 1e-18))
            )
        tmp = float(delta.T @ cov_inv @ delta)
        return float(norm_factor * math.exp(-0.5 * tmp))

    # ------------------------------------------------------------------
    # 区域生长阶段: 基于共面体素和主方向的 26 邻接生长
    # ------------------------------------------------------------------
    def _RegionGrowOnVoxels(
            self,
            coords: np.ndarray,
            voxel_centers: List[np.ndarray],
            voxel_point_indices: List[np.ndarray],
            voxel_normals: List[np.ndarray],
            major_orientations: List[np.ndarray],
    ) -> List[Discontinuity]:
        """
        功能简介:
            以共面体素为基本单元, 在每个主方向下做区域生长, 输出结构面列表。
        """
        num_voxels = len(voxel_centers)
        if num_voxels == 0:
            return []

        # 根据体素中心构造 3D 网格索引
        all_centers = np.vstack(voxel_centers)
        min_xyz = all_centers.min(axis=0)
        inv_size = 1.0 / self.voxel_size

        grid_index_to_voxel_indices: Dict[Tuple[int, int, int], List[int]] = {}
        voxel_index_to_grid: List[Tuple[int, int, int]] = []

        for v_idx, center in enumerate(voxel_centers):
            x, y, z = center
            i = int(math.floor((x - min_xyz[0]) * inv_size))
            j = int(math.floor((y - min_xyz[1]) * inv_size))
            k = int(math.floor((z - min_xyz[2]) * inv_size))
            g_key = (i, j, k)
            voxel_index_to_grid.append(g_key)
            grid_index_to_voxel_indices.setdefault(g_key, []).append(v_idx)

        # 预构造每个体素的 26 邻接列表
        voxel_neighbors: List[List[int]] = [[] for _ in range(num_voxels)]
        neighbor_offsets = [
            (dx, dy, dz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
            if not (dx == 0 and dy == 0 and dz == 0)
        ]

        for v_idx, g_idx in enumerate(voxel_index_to_grid):
            gi, gj, gk = g_idx
            neighbors: List[int] = []
            for dx, dy, dz in neighbor_offsets:
                key = (gi + dx, gj + dy, gk + dz)
                if key not in grid_index_to_voxel_indices:
                    continue
                neighbors.extend(grid_index_to_voxel_indices[key])
            voxel_neighbors[v_idx] = neighbors

        discontinuities: List[Discontinuity] = []

        # 对每个主方向执行一次区域生长
        for ori_idx, ori in enumerate(major_orientations):
            self.logger.info(
                f"MOE: start region growing for orientation {ori_idx}."
            )
            ori_vec = ori / max(np.linalg.norm(ori), 1e-8)

            used = np.zeros(num_voxels, dtype=bool)

            for v_idx in range(num_voxels):
                if used[v_idx]:
                    continue
                v_normal = voxel_normals[v_idx]
                if not self._CheckAngleWithOrientation(
                        v_normal, ori_vec, self._cos_seed_angle_threshold
                ):
                    continue

                # BFS 生长
                region_voxel_indices: List[int] = []
                queue: List[int] = [v_idx]
                used[v_idx] = True

                while queue:
                    cur_v = queue.pop(0)
                    region_voxel_indices.append(cur_v)

                    for nb_v in voxel_neighbors[cur_v]:
                        if used[nb_v]:
                            continue
                        nb_normal = voxel_normals[nb_v]
                        if not self._CheckAngleWithOrientation(
                                nb_normal,
                                ori_vec,
                                self._cos_grow_angle_threshold,
                        ):
                            continue
                        used[nb_v] = True
                        queue.append(nb_v)

                if not region_voxel_indices:
                    continue

                # 汇总区域内所有点
                region_point_indices: List[int] = []
                for rv in region_voxel_indices:
                    region_point_indices.extend(
                        voxel_point_indices[rv].tolist()
                    )

                if len(region_point_indices) < self.min_region_points:
                    continue

                inlier_indices = np.asarray(region_point_indices, dtype=int)
                pts = coords[inlier_indices]
                centroid = pts.mean(axis=0)
                pts_centered = pts - centroid
                cov = (pts_centered.T @ pts_centered) / float(
                    max(len(inlier_indices), 1)
                )
                try:
                    eigen_vals, eigen_vecs = np.linalg.eigh(cov)
                except np.linalg.LinAlgError:
                    continue
                order = np.argsort(eigen_vals)
                normal = eigen_vecs[:, order[0]]
                normal /= max(np.linalg.norm(normal), 1e-8)
                d = -float(normal.dot(centroid))
                distances = pts @ normal + d
                rmse = float(math.sqrt(np.mean(distances * distances)))

                plane = Plane(
                    normal=(
                        float(normal[0]),
                        float(normal[1]),
                        float(normal[2]),
                    ),
                    d=float(d),
                    centroid=(
                        float(centroid[0]),
                        float(centroid[1]),
                        float(centroid[2]),
                    ),
                    inlier_indices=inlier_indices.tolist(),
                    rmse=rmse,
                )

                dip, dip_direction = self._ComputeDipAndDipDirection(plane)

                segment = Segment(
                    plane=plane,
                    point_indices=inlier_indices.tolist(),
                    trace_length=0.0,
                )

                disc = Discontinuity(
                    dip=dip,
                    dip_direction=dip_direction,
                    segments=[segment],
                )
                discontinuities.append(disc)

        self.logger.info(
            f"MOE: finished region growing, got {len(discontinuities)} "
            f"discontinuities from {len(major_orientations)} orientations."
        )
        return discontinuities

    def _CheckAngleWithOrientation(
            self,
            normal_vec: np.ndarray,
            ori_vec: np.ndarray,
            cos_threshold: float,
    ) -> bool:
        """
        功能简介:
            判断 normal_vec 与 ori_vec 的夹角是否小于给定阈值。
        """
        n = np.asarray(normal_vec, dtype=np.float64)
        if n[2] < 0.0:
            n = -n
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-8:
            return False
        n /= n_norm

        o = np.asarray(ori_vec, dtype=np.float64)
        o_norm = np.linalg.norm(o)
        if o_norm < 1e-8:
            return False
        o /= o_norm

        cos_val = float(np.dot(n, o))
        return cos_val >= cos_threshold

    def _ComputeDipAndDipDirection(self, plane: Plane) -> Tuple[float, float]:
        """
        功能简介:
            由平面法向计算倾角(dip)与倾向(dip_direction, 以正北为起点顺时针)。
        """
        nx, ny, nz = plane.normal
        if nz < 0.0:
            nx, ny, nz = -nx, -ny, -nz

        dip = math.degrees(math.acos(max(min(nz, 1.0), -1.0)))

        # 走向向量: n 与竖直 z 轴叉乘
        sx = ny
        sy = -nx
        dip_dir_rad = math.atan2(sx, sy)
        dip_direction = (math.degrees(dip_dir_rad) + 360.0) % 360.0

        return dip, dip_direction
