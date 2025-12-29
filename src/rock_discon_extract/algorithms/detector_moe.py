from typing import List, Tuple, Dict
import math
import numpy as np

from .base import PlaneDetectionAlgorithm
from ..geometry import Discontinuity, Plane, Segment
from ..pointcloud import PointCloud
from ..logging_utils import Timer


class MoeDetector(PlaneDetectionAlgorithm):
    """
    功能简介:
        基于 3D 点云主方向估计的自动岩面提取算法 (Major Orientation Estimation, 简称 MOE)。该算法参考文献
        《Major Orientation Estimation-Based Rock Surface Extraction for 3D Rock-Mass Point Clouds》，
        通过体素聚类、主方向估计 (高斯核投票) 以及基于体素的区域生长, 快速提取岩石不连续面。

    实现思路(概要):
        - 阶段1: 体素划分与聚类。使用两级空间网格将点云体素化。首先按 `voxel_size` 划分大体素, 若点数超过
          `min_points_big_voxel` 则判断其近似共面性; 共面则加入共面体素集合, 否则再细分为 8 个小体素,
          再根据点数阈值 `min_points_sub_voxel` 及共面约束将其归类为共面体素、非共面体素或稀疏体素。
        - 阶段2: 主方向估计。对所有共面体素的法向量在半球空间上进行高斯核投票累加, 得到法向量分布累积图
          (累加器球)。采用二维高斯分布作为核函数进行投票, 每个共面体素的投票影响范围由其法向估计
          不确定性决定, 再使用滑动窗口在累加器上检测主方向峰值。
        - 阶段3: 岩面提取 (区域生长)。对于每个主方向, 从未分配的共面体素中选取法向与该主方向夹角在
          `t1` 阈值内的体素作为种子, 以体素为单位进行区域生长。在生长过程中, 邻近体素若法向与簇法向夹角
          在 `t2` 阈值范围且距离在 `max_distance` 内则整体并入簇; 对于邻接的非共面/稀疏体素, 则检查
          其中点的法向与主方向的夹角在 `t3` 内且距离在 `max_distance` 内的点, 将这些点并入当前簇。
          重复生长直至无新体素加入, 得到一个面簇; 然后继续寻找下一个种子, 提取同一主方向下其他平行子平面簇。
        - 结果输出: 对每个提取的面簇, 进行平面最小二乘拟合, 计算倾角和倾向等属性, 构造 `Discontinuity` 对象输出。

    输入(构造函数参数):
        voxel_size: float
            大体素的边长长度。用于初步空间划分点云, 通常取值与特征面尺寸同量级。
        num_major_orientations: int, default 0
            限定提取的主方向个数; 若为 0 则自适应提取所有峰值方向。
        Sa: float, default 30.0
            共面判据的特征值比阈值。要求次小特征值 λ2 > Sa * λ1 (最小特征值) 才认为体素内点近似共面。
        epsilon: float, default 0.05
            共面判据的均方误差 (MSE) 阈值。单位与坐标一致的平方, 代表点到拟合平面的平均平方距离上限。
        min_points_big_voxel: int, default 300
            大体素内最少点数阈值。大体素若少于此点数则直接归为稀疏体素, 不再细分。
        min_points_sub_voxel: int, default 150
            小体素(细分后)内最少点数阈值。小体素若少于此点数则标记为稀疏体素。
        t1: float, default 10.0
            种子体素选择的最大法向夹角阈值 (度)。主方向与共面体素法向之间的允许最大夹角。值越小,
            种子法向需与峰值更接近。
        t2: float, default 15.0
            区域生长的法向最大夹角阈值 (度)。决定生长过程中接受邻近共面体素的法向偏离程度。
        t3: float, default None
            点法向与面法向的最大夹角阈值 (度)。用于判断非共面/稀疏体素内的点是否可并入当前平面簇。
            默认等于 t2。
        max_distance: float, default 0.3
            平面距离阈值。用于判断相邻平面是否连续 (平行平面间的距离) 以及点到面距离是否在可接受范围。
            单位与坐标一致。

    输出:
        DetectPlanes 返回 List[Discontinuity] 列表, 每个 Discontinuity 表示一个提取出的不连续岩面。
    """

    def __init__(self,
                 voxel_size: float,
                 num_major_orientations: int = 0,
                 Sa: float = 30.0,
                 epsilon: float = 0.05,
                 min_points_big_voxel: int = 300,
                 min_points_sub_voxel: int = 150,
                 t1: float = 10.0,
                 t2: float = 15.0,
                 t3: float = None,
                 max_distance: float = 0.3):
        super().__init__(name="MOE")
        # 参数初始化
        self.voxel_size = float(voxel_size)
        self.num_major_orientations = int(num_major_orientations)
        self.Sa = float(Sa)
        self.epsilon = float(epsilon)
        self.min_points_big_voxel = int(min_points_big_voxel)
        self.min_points_sub_voxel = int(min_points_sub_voxel)
        self.t1 = float(t1)
        self.t2 = float(t2)
        self.t3 = float(t3) if t3 is not None else float(t2)
        self.max_distance = float(max_distance)

    def _DetectImpl(self, point_cloud: PointCloud) -> List[Discontinuity]:
        """
        功能简介:
            实现 MOE 算法的主流程, 对输入点云进行主方向估计和基于体素的区域生长, 输出不连续面集合。

        输入:
            point_cloud: PointCloud
                包含坐标、法向(可选)、曲率(可选)等信息的岩体点云。

        输出:
            discontinuities: List[Discontinuity]
                检测得到的不连续面列表。
        """
        # 若点云为空则直接返回
        if point_cloud is None or len(point_cloud.points) == 0:
            return []

        # 准备数据: 坐标和法向
        coords = np.array([[p.x, p.y, p.z] for p in point_cloud.points], dtype=float)
        normals_pt = np.array([[p.nx, p.ny, p.nz] for p in point_cloud.points], dtype=float) \
            if hasattr(point_cloud.points[0], "nx") else None
        total_points = coords.shape[0]

        # 阈值预计算（角度转余弦）
        cos_t1 = math.cos(math.radians(self.t1))
        cos_t2_min = math.cos(math.radians(self.t2 * 0.5))  # 默认 MinScale=0.5
        cos_t2_max = math.cos(math.radians(self.t2 * 1.0))  # 默认 MaxScale=1.0
        cos_t3 = math.cos(math.radians(self.t3))

        # ========================
        # 阶段1: 基于体素的点云聚类
        # ========================
        # 计算点云包围盒范围
        min_x = float(np.min(coords[:, 0]))
        max_x = float(np.max(coords[:, 0]))
        min_y = float(np.min(coords[:, 1]))
        max_y = float(np.max(coords[:, 1]))
        min_z = float(np.min(coords[:, 2]))
        max_z = float(np.max(coords[:, 2]))

        # 划分大体素
        big_voxel_index: Dict[Tuple[int, int, int], List[int]] = {}
        inv_size = 1.0 / self.voxel_size
        # 遍历所有点, 分配到所在大体素
        for idx, p in enumerate(coords):
            ix = int(math.floor((p[0] - min_x) * inv_size))
            iy = int(math.floor((p[1] - min_y) * inv_size))
            iz = int(math.floor((p[2] - min_z) * inv_size))
            big_voxel_index.setdefault((ix, iy, iz), []).append(idx)

        # 结果体素列表
        class Voxel:
            """
            体素结构:
                points: List[int]  - 点索引列表
                center: (float, float, float) - 体素质心
                normal: (float, float, float) or None - 拟合平面法向
                size: float - 体素边长
                vtype: str - 'cop'/'non'/'spa' 三种类型
                plane_d: float or None - 拟合平面截距 d, 满足 n·x + d = 0
                range: (int,int,int,int,int,int) or None - 全局网格索引范围 (用于邻接判断)
            """
            def __init__(self,
                         points,
                         center,
                         normal,
                         size,
                         vtype,
                         plane_d):
                self.points = points
                self.center = center
                self.normal = normal
                self.size = size
                self.vtype = vtype
                self.plane_d = plane_d
                self.range = None

        voxels: List[Voxel] = []      # 所有体素对象列表
        coplanar_indices: List[int] = []  # 共面体素在 voxels 列表中的索引集合 (用于后续快速访问)

        # 遍历每个大体素, 根据算法进行分类
        for big_idx, point_idx_list in big_voxel_index.items():
            count = len(point_idx_list)
            if count == 0:
                continue  # 无点则跳过
            if count > self.min_points_big_voxel:
                # 计算协方差矩阵与 PCA 特征值/向量
                pts = coords[point_idx_list]
                centroid = pts.mean(axis=0)
                pts_centered = pts - centroid
                cov = (pts_centered.T @ pts_centered) / max(pts_centered.shape[0], 1)
                eigen_vals, eigen_vecs = np.linalg.eigh(cov)
                # 排序特征值从小到大
                order = np.argsort(eigen_vals)
                eigen_vals = eigen_vals[order]
                eigen_vecs = eigen_vecs[:, order]
                # 平面法向为最小特征值对应特征向量 (单位化)
                normal_vec = eigen_vecs[:, 0]
                normal_vec = normal_vec / (np.linalg.norm(normal_vec) + 1e-12)
                # 计算平面截距 d = -n·centroid
                d_val = -float(np.dot(normal_vec, centroid))
                # 计算 MSE
                dist = pts_centered @ normal_vec
                mse_val = float(np.mean(dist ** 2))
                # 判断是否近似共面
                if eigen_vals[1] > self.Sa * eigen_vals[0] and mse_val < self.epsilon:
                    # 大体素共面, 直接作为共面体素
                    voxels.append(
                        Voxel(points=point_idx_list,
                              center=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
                              normal=(float(normal_vec[0]), float(normal_vec[1]), float(normal_vec[2])),
                              size=self.voxel_size,
                              vtype="cop",
                              plane_d=d_val)
                    )
                    coplanar_indices.append(len(voxels) - 1)
                else:
                    # 大体素非共面, 进一步划分为 8 个子体素
                    half = self.voxel_size / 2.0
                    # 计算大体素坐标范围
                    base_min = np.array([big_idx[0] * self.voxel_size + min_x,
                                         big_idx[1] * self.voxel_size + min_y,
                                         big_idx[2] * self.voxel_size + min_z], dtype=float)
                    # 子体素按二分 (0/1) 组合偏移
                    sub_buckets: Dict[Tuple[int, int, int], List[int]] = {
                        (a, b, c): [] for a in (0, 1) for b in (0, 1) for c in (0, 1)
                    }
                    # 将该大体素内点根据中心划分到 8 个子区域
                    mid = base_min + half  # 阈值坐标 (mid_x, mid_y, mid_z)
                    for pid in point_idx_list:
                        px, py, pz = coords[pid]
                        xi = 0 if px < mid[0] else 1
                        yi = 0 if py < mid[1] else 1
                        zi = 0 if pz < mid[2] else 1
                        sub_buckets[(xi, yi, zi)].append(pid)
                    # 遍历每个子体素
                    for (a, b, c), sub_points in sub_buckets.items():
                        if len(sub_points) == 0:
                            continue
                        if len(sub_points) > self.min_points_sub_voxel:
                            # 计算子体素共面性
                            sub_pts = coords[sub_points]
                            sub_centroid = sub_pts.mean(axis=0)
                            sub_pts_centered = sub_pts - sub_centroid
                            cov_sub = (sub_pts_centered.T @ sub_pts_centered) / max(sub_pts_centered.shape[0], 1)
                            e_vals, e_vecs = np.linalg.eigh(cov_sub)
                            order2 = np.argsort(e_vals)
                            e_vals = e_vals[order2]
                            e_vecs = e_vecs[:, order2]
                            n_vec = e_vecs[:, 0]
                            n_vec = n_vec / (np.linalg.norm(n_vec) + 1e-12)
                            d_val2 = -float(np.dot(n_vec, sub_centroid))
                            dist2 = sub_pts_centered @ n_vec
                            mse_sub = float(np.mean(dist2 ** 2))
                            if e_vals[1] > self.Sa * e_vals[0] and mse_sub < self.epsilon:
                                voxels.append(
                                    Voxel(points=sub_points,
                                          center=(float(sub_centroid[0]), float(sub_centroid[1]), float(sub_centroid[2])),
                                          normal=(float(n_vec[0]), float(n_vec[1]), float(n_vec[2])),
                                          size=half,
                                          vtype="cop",
                                          plane_d=d_val2)
                                )
                                coplanar_indices.append(len(voxels) - 1)
                            else:
                                voxels.append(
                                    Voxel(points=sub_points,
                                          center=(float(sub_centroid[0]), float(sub_centroid[1]), float(sub_centroid[2])),
                                          normal=None,
                                          size=half,
                                          vtype="non",
                                          plane_d=None)
                                )
                        else:
                            # 子体素稀疏
                            sub_pts = coords[sub_points]
                            sub_centroid = sub_pts.mean(axis=0)
                            voxels.append(
                                Voxel(points=sub_points,
                                      center=(float(sub_centroid[0]), float(sub_centroid[1]), float(sub_centroid[2])),
                                      normal=None,
                                      size=half,
                                      vtype="spa",
                                      plane_d=None)
                            )
            else:
                # 大体素点数不足, 作为稀疏体素
                pts = coords[point_idx_list]
                centroid = pts.mean(axis=0)
                voxels.append(
                    Voxel(points=point_idx_list,
                          center=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
                          normal=None,
                          size=self.voxel_size,
                          vtype="spa",
                          plane_d=None)
                )

        # ========================
        # 阶段2: 主方向估计 (高斯半球投票)
        # ========================
        # 半球累加器 (θ: 0-360, φ: 0-90)
        phi_bins = 91   # φ 0~90 度 (间隔 1 度)
        theta_bins = 360  # θ 0~359 度 (间隔 1 度)
        accumulator = np.zeros((phi_bins, theta_bins), dtype=float)

        # 计算 bounding box 边长 (最长边) 以便计算权重
        meshsize = max(max_x - min_x, max_y - min_y, max_z - min_z)
        meshsize = float(meshsize) if meshsize > 0.0 else 1.0

        # 对每个共面体素法向进行投票
        for vi in coplanar_indices:
            voxel = voxels[vi]
            n = np.array(voxel.normal, dtype=float)
            # 确保法向在上半球 (若 z < 0 则翻转)
            if n[2] < 0.0:
                n = -n
            # 计算法向在球坐标下的角度
            nx, ny, nz = n
            # θ: 方位角 [0, 360), 从 x 轴起顺时针
            if abs(nx) < 1e-12 and abs(ny) < 1e-12:
                theta = 0.0
            else:
                theta = math.degrees(math.atan2(ny, nx))
            if theta < 0.0:
                theta += 360.0
            # φ: 极角, 从正 z 轴向下
            phi = math.degrees(math.acos(max(min(nz, 1.0), -1.0)))
            if phi > 90.0:
                phi = 180.0 - phi
            # 取得对应累加器索引 (取整)
            theta_idx = int(theta) % theta_bins
            phi_idx = int(round(phi))
            if phi_idx >= phi_bins:
                phi_idx = phi_bins - 1

            # 根据特征值比估计角度标准差 sigma (粗略)
            evals = None
            if voxel.normal is not None:
                pts_idx = voxel.points
                if len(pts_idx) >= 3:
                    pts = coords[pts_idx]
                    pts_centered = pts - np.array(voxel.center)
                    cov_mat = (pts_centered.T @ pts_centered) / max(pts_centered.shape[0], 1)
                    evals = np.linalg.eigvalsh(cov_mat)
                    evals.sort()
            if evals is not None and evals[0] > 0.0:
                ratio = float(evals[1] / evals[0])  # λ2/λ1
            else:
                ratio = self.Sa  # 若无法计算, 用阈值代替

            sigma_base = 5.0  # 基准 5°
            sigma_angle = sigma_base * (self.Sa / ratio)
            if sigma_angle < 0.5:
                sigma_angle = 0.5
            sigma_phi = float(sigma_angle)
            sigma_theta = float(sigma_angle)

            # 投票范围 (约 1 个标准差)
            phi_range = max(1, int(round(sigma_phi)))
            theta_range = max(1, int(round(sigma_theta)))

            # 权重系数 w_k
            w_a = 0.5
            w_d = 0.5
            weight = w_a * (voxel.size / meshsize) + w_d * (len(voxel.points) / total_points)

            # 在 (φ_idx ± phi_range, θ_idx ± theta_range) 范围内累加高斯值
            for d_phi in range(-phi_range, phi_range + 1):
                pi = phi_idx + d_phi
                if pi < 0 or pi >= phi_bins:
                    continue
                for d_theta in range(-theta_range, theta_range + 1):
                    tj = theta_idx + d_theta
                    if tj < 0:
                        tj += theta_bins
                    elif tj >= theta_bins:
                        tj -= theta_bins
                    # 计算二维高斯 (省略归一化因子)
                    delta_angle_phi = d_phi / (sigma_phi + 1e-12)
                    delta_angle_theta = d_theta / (sigma_theta + 1e-12)
                    value = math.exp(-0.5 * (delta_angle_phi ** 2 + delta_angle_theta ** 2))
                    accumulator[pi, tj] += weight * value

        # 峰值检测: 滑动窗口寻找局部最大
        peak_mask = np.zeros_like(accumulator, dtype=bool)
        N = 3  # 窗口大小 (3x3)
        half_win = N // 2
        for i in range(phi_bins):
            for j in range(theta_bins):
                val = accumulator[i, j]
                is_max = True
                for di in range(-half_win, half_win + 1):
                    pi = i + di
                    if pi < 0 or pi >= phi_bins:
                        continue
                    for dj in range(-half_win, half_win + 1):
                        tj = j + dj
                        if tj < 0:
                            tj += theta_bins
                        elif tj >= theta_bins:
                            tj -= theta_bins
                        if accumulator[pi, tj] > val:
                            is_max = False
                            break
                    if not is_max:
                        break
                if is_max and val > 1e-8:
                    peak_mask[i, j] = True

        # 获取峰值方向列表 (φ, θ)
        peaks: List[Tuple[float, int, int]] = []
        for i in range(phi_bins):
            for j in range(theta_bins):
                if peak_mask[i, j]:
                    peaks.append((accumulator[i, j], i, j))
        # 按投票值降序排序
        peaks.sort(key=lambda x: x[0], reverse=True)
        if self.num_major_orientations and self.num_major_orientations > 0:
            peaks = peaks[:self.num_major_orientations]

        # ========================
        # 阶段3: 基于体素的区域生长提取岩面
        # ========================
        discontinuities: List[Discontinuity] = []

        n_voxels = len(voxels)
        neighbors: List[set] = [set() for _ in range(n_voxels)]

        # 建立索引映射字典 (按最小子体素网格索引)
        x_min_map: Dict[int, List[int]] = {}
        x_max_map: Dict[int, List[int]] = {}
        y_min_map: Dict[int, List[int]] = {}
        y_max_map: Dict[int, List[int]] = {}
        z_min_map: Dict[int, List[int]] = {}
        z_max_map: Dict[int, List[int]] = {}

        min_unit = self.voxel_size / 2.0  # 最小网格边长

        for idx, vox in enumerate(voxels):
            size_factor = int(round(vox.size / min_unit))  # 大体素=2, 小体素=1
            ix = int(math.floor((vox.center[0] - min_x) / min_unit))
            iy = int(math.floor((vox.center[1] - min_y) / min_unit))
            iz = int(math.floor((vox.center[2] - min_z) / min_unit))
            if size_factor == 2:
                x_min_idx = ix
                x_max_idx = ix + 1
                y_min_idx = iy
                y_max_idx = iy + 1
                z_min_idx = iz
                z_max_idx = iz + 1
            else:
                x_min_idx = x_max_idx = ix
                y_min_idx = y_max_idx = iy
                z_min_idx = z_max_idx = iz

            x_min_map.setdefault(x_min_idx, []).append(idx)
            x_max_map.setdefault(x_max_idx, []).append(idx)
            y_min_map.setdefault(y_min_idx, []).append(idx)
            y_max_map.setdefault(y_max_idx, []).append(idx)
            z_min_map.setdefault(z_min_idx, []).append(idx)
            z_max_map.setdefault(z_max_idx, []).append(idx)

            vox.range = (x_min_idx, x_max_idx, y_min_idx, y_max_idx, z_min_idx, z_max_idx)

        # 计算邻接: 只检查正方向以避免重复
        for i, vox in enumerate(voxels):
            x_min_idx, x_max_idx, y_min_idx, y_max_idx, z_min_idx, z_max_idx = vox.range

            # X 正方向邻居
            neighbor_candidates = x_min_map.get(x_max_idx + 1, [])
            for j in neighbor_candidates:
                if j == i:
                    continue
                vox_j = voxels[j]
                xm0, xm1, ym0, ym1, zm0, zm1 = vox_j.range
                if (y_min_idx <= ym1 and ym0 <= y_max_idx and
                        z_min_idx <= zm1 and zm0 <= z_max_idx):
                    neighbors[i].add(j)
                    neighbors[j].add(i)

            # Y 正方向邻居
            neighbor_candidates = y_min_map.get(y_max_idx + 1, [])
            for j in neighbor_candidates:
                if j == i:
                    continue
                vox_j = voxels[j]
                xm0, xm1, ym0, ym1, zm0, zm1 = vox_j.range
                if (x_min_idx <= xm1 and xm0 <= x_max_idx and
                        z_min_idx <= zm1 and zm0 <= z_max_idx):
                    neighbors[i].add(j)
                    neighbors[j].add(i)

            # Z 正方向邻居
            neighbor_candidates = z_min_map.get(z_max_idx + 1, [])
            for j in neighbor_candidates:
                if j == i:
                    continue
                vox_j = voxels[j]
                xm0, xm1, ym0, ym1, zm0, zm1 = vox_j.range
                if (x_min_idx <= xm1 and xm0 <= x_max_idx and
                        y_min_idx <= ym1 and ym0 <= y_max_idx):
                    neighbors[i].add(j)
                    neighbors[j].add(i)

        # 标记数组
        used_voxel = [False] * n_voxels              # 共面体素使用标记
        assigned_point = np.zeros(total_points, dtype=bool)  # 点使用标记

        # 对每个主方向进行区域生长
        for _, phi_idx, theta_idx in peaks:
            # 计算该峰值对应的单位法向量
            phi_val = float(phi_idx)
            theta_val = float(theta_idx)
            phi_rad = math.radians(phi_val)
            theta_rad = math.radians(theta_val)
            cluster_normal = np.array([math.sin(phi_rad) * math.cos(theta_rad),
                                       math.sin(phi_rad) * math.sin(theta_rad),
                                       math.cos(phi_rad)], dtype=float)

            # 寻找未使用且与该主方向接近的种子体素
            for vi in coplanar_indices:
                if used_voxel[vi]:
                    continue
                voxel = voxels[vi]
                n = np.array(voxel.normal, dtype=float)
                # 法向夹角判断
                if np.dot(cluster_normal, n) < cos_t1:
                    continue

                # 选择此体素作为种子
                used_voxel[vi] = True

                # 初始化平面簇
                cluster_point_indices: List[int] = []
                for pid in voxel.points:
                    if not assigned_point[pid]:
                        assigned_point[pid] = True
                        cluster_point_indices.append(pid)

                # 计算簇参考平面参数 (通过种子质心)
                seed_center = np.array(voxel.center)
                cluster_d = -float(np.dot(cluster_normal, seed_center))

                # BFS 队列
                queue = [vi]
                while queue:
                    curr = queue.pop(0)
                    for nei in neighbors[curr]:
                        if used_voxel[nei]:
                            continue
                        neighbor_vox = voxels[nei]
                        nei_center = np.array(neighbor_vox.center)
                        plane_dist = abs(np.dot(cluster_normal, nei_center) + cluster_d)

                        if neighbor_vox.vtype == "cop":
                            # 邻居为共面体素
                            n2 = np.array(neighbor_vox.normal, dtype=float)
                            dot_val = float(np.dot(cluster_normal, n2))
                            if dot_val >= cos_t2_max and plane_dist <= self.max_distance:
                                # 法向差在最大允许范围内
                                if dot_val >= cos_t2_min:
                                    # 法向足够接近 (高质量邻居) - 整体并入
                                    used_voxel[nei] = True
                                    for pid in neighbor_vox.points:
                                        if not assigned_point[pid]:
                                            assigned_point[pid] = True
                                            cluster_point_indices.append(pid)
                                    queue.append(nei)
                                else:
                                    # 法向偏离中等 - 仅并入符合单点条件的点
                                    used_voxel[nei] = True  # 标记为已处理
                                    if normals_pt is not None:
                                        for pid in neighbor_vox.points:
                                            if assigned_point[pid]:
                                                continue
                                            pn = normals_pt[pid]
                                            if (np.dot(cluster_normal, pn) >= cos_t3 and
                                                    abs(np.dot(cluster_normal, coords[pid]) + cluster_d)
                                                    <= self.max_distance):
                                                assigned_point[pid] = True
                                                cluster_point_indices.append(pid)
                        else:
                            # 邻居为非共面或稀疏体素
                            if plane_dist <= self.max_distance and normals_pt is not None:
                                for pid in neighbor_vox.points:
                                    if assigned_point[pid]:
                                        continue
                                    pn = normals_pt[pid]
                                    if (np.dot(cluster_normal, pn) >= cos_t3 and
                                            abs(np.dot(cluster_normal, coords[pid]) + cluster_d)
                                            <= self.max_distance):
                                        assigned_point[pid] = True
                                        cluster_point_indices.append(pid)
                            # 注意: 非共面/稀疏体素不标记 used_voxel, 允许其它平面继续利用其剩余点

                # 检查簇内点数是否足够构成面 (过滤过小簇)
                if len(cluster_point_indices) == 0:
                    continue

                # 拟合平面并创建 Discontinuity 对象
                plane = self._FitPlaneFromPoints(coords, np.array(cluster_point_indices, dtype=int))
                dip, dip_dir = self._ComputeDipAndDipDirection(plane)
                segment = Segment(plane=plane,
                                  point_indices=cluster_point_indices,
                                  trace_length=0.0)
                discontinuity = Discontinuity(segments=[segment],
                                              dip=dip,
                                              dip_direction=dip_dir,
                                              roughness=plane.rmse,
                                              algorithm_name=self.name)
                discontinuities.append(discontinuity)

        return discontinuities

    def _FitPlaneFromPoints(self, coords: np.ndarray, inlier_indices: np.ndarray) -> Plane:
        """
        功能简介:
            对给定点集进行最小二乘平面拟合, 返回 Plane 对象。

        实现思路:
            1) 取出索引集合对应的点坐标, 计算质心;
            2) 计算去中心化协方差矩阵并特征分解;
            3) 取最小特征值对应特征向量作为法向量 (单位化);
            4) 计算平面截距 d = -n · centroid;
            5) 计算所有点到平面的距离均值的平方根 (RMSE);
            6) 构造 Plane 对象保存结果。

        输入:
            coords: np.ndarray, 形状 (N, 3), 全部点坐标数组。
            inlier_indices: np.ndarray, 形状 (M,), 属于该平面的点索引。

        输出:
            plane: Plane, 包含法向、截距、质心、内点索引与拟合 RMSE。
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
        # 计算 RMSE
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
        功能简介:
            计算平面法向对应的倾角和倾向。

        实现说明:
            假定坐标系 x-东 (East), y-北 (North), z-上 (Up):
            - 倾角 dip 为平面与水平面的夹角, 计算公式:
              dip = atan2(水平分量长度, |nz|)。
            - 倾向 dip_direction 为法向在水平面投影相对北方向的方位角, 计算公式:
              dip_direction = atan2(nx, ny), 并转换到 [0, 360) 区间。

        输入:
            plane: Plane, 已拟合平面对象, 至少包含 plane.normal。

        输出:
            (dip, dip_direction): (float, float)
                dip: 倾角 (度)
                dip_direction: 倾向 (度, 以北为 0°, 顺时针)
        """
        nx, ny, nz = plane.normal
        # 单位化法向
        norm_len = math.sqrt(nx * nx + ny * ny + nz * nz) + 1e-12
        nx /= norm_len
        ny /= norm_len
        nz /= norm_len
        # 翻转到上半球
        if nz < 0.0:
            nx = -nx
            ny = -ny
            nz = -nz
        horizontal = math.sqrt(nx * nx + ny * ny)
        dip = math.degrees(math.atan2(horizontal, abs(nz)))
        azimuth = math.degrees(math.atan2(nx, ny))
        if azimuth < 0.0:
            azimuth += 360.0
        return float(dip), float(azimuth)
