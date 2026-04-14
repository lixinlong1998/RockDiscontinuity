# RockDiscontinuity/src/rock_discon_extract/algorithms/detector_supervoxel.py

from typing import List, Dict, Tuple, Set
import math
import numpy as np

from typing import Any, Dict, Optional, Set, Tuple
import os
import time

from .base import PlaneDetectionAlgorithm
from ..geometry import Discontinuity, Plane, Segment
from ..pointcloud import PointCloud
from ..logging_utils import LoggerManager, Timer

# 尝试导入 Open3D 库
try:
    import open3d as o3d
except ImportError:
    o3d = None


class SupervoxelDetector(PlaneDetectionAlgorithm):
    """
    基于超体素与区域生长的高精度平面检测算法 (Supervoxel-RegionGrowing).

    功能简介:
        针对包含大量平面结构面的岩体点云, 通过体素化、局部 RANSAC 平面提取、边缘平面分割、
        超体素生长以及基于片段的区域生长, 自动提取不连续面 (结构面) 的平面片段并进行合并,
        最终输出一组代表实际结构面的 Discontinuity 对象。

    实现思路(详细):
        1. 体素划分:
            - 使用 voxel_size 对点云做规则 3D 网格划分, 为每个点分配体素索引 (vx, vy, vz)。
            - 构建 voxel_id -> 点索引列表, 以及 point_index -> voxel_id 映射。
        2. 体素内局部平面提取 (voxel-patch):
            - 对每个体素 voxel_i 内点集执行一次 RANSAC:
                * 使用 ransac_distance 作为距离阈值提取局部平面。
                * 对得到的 inlier 再用 ransac_angle 做法向角度过滤, 确保 patch 内点在法向上也高度一致。
            - 若过滤后的 inlier 数 ≥ min_plane_points:
                * 该体素被视为 coplanar voxel, inlier 作为一个 voxel-patch, 记为一个生长单元 cluster。
                * 其余点作为 remain_points[voxel_i]。
            - 否则:
                * 该体素被视为 non-coplanar 或噪声体素, 所有点加入 remain_points[voxel_i]。
        3. 非共面体素内边缘平面提取 (edge-patch) 与拼接:
            - 对每个体素 voxel_i 的剩余点集 remain_points[voxel_i]:
                * 若点数 > min_edge_points, 则循环执行:
                    (a) 在该体素剩余点中执行 RANSAC(距离阈值 ransac_distance):
                        - 得到候选 edge-patch 的 inlier。
                        - 用 ransac_angle 对 inlier 做法向过滤, 获取精确 edge-patch。
                    (b) 若 edge-patch 点数 ≥ min_edge_patch_points:
                        - 在 voxel_i 的邻域体素中搜索已有平面 cluster, 计算:
                            + mw: edge-patch 点到候选平面的平均距离 (需 < edge_distance)。
                            + nw: edge-patch 平面法向与候选平面法向的夹角 (需 < edge_angle)。
                            + Pw = sqrt( (mw/edge_distance)^2 + (nw/edge_angle)^2 ) 作为相似性指标。
                        - 若存在满足阈值的邻近平面, 选取 Pw 最小者, 将 edge-patch 并入该平面 cluster。
                        - 否则, 将该 edge-patch 作为新的 cluster (新的平面生长单元)。
                        - 从 remain_points[voxel_i] 中移除 edge-patch 点, 再次循环。
                    (c) 若 edge-patch 点数 < min_edge_patch_points:
                        - 认为该体素剩余点无法构成稳定平面, 停止对该体素的边缘检测。
        4. 超体素平面生长 (Supervoxel segmentation):
            - 每个 cluster (voxel-patch 或 edge-patch) 都作为一个 grow_unit 种子:
                * seed_voxels = cluster["voxel_ids"] (可能跨多个体素)。
                * 构建候选体素集合: 所有 seed_voxels 及其一圈 26 邻域体素。
                * 从这些体素的 remain_points 中收集候选散点 candidate_points。
            - 以 cluster["orig_points"] 作为初始平面点集, 设置初始阈值:
                * dist_th = super_distance, ang_th = super_angle。
            - 迭代区域生长:
                * 在当前阈值下, 从 candidate_points 中选取“距离 < dist_th 且 法向角度 < ang_th”的点加入平面。
                * 对加入后的点集做 PCA, 拟合新的平面法向, 计算与原始法向的夹角变化 orientation_diff。
                * 若 orientation_diff ≤ max_refit_error, 接受当前点集为 supervoxel patch。
                * 否则收缩 dist_th, ang_th (按 distance_step, angle_step) 并重新尝试。
            - 对每个 supervoxel patch:
                * 用最终点集重新统计其覆盖的 voxel_ids (从 point_voxel_map 反查)。
                * 将 patch 内点从 remain_points 中删除, 不再参与其它 patch 生长。
        5. 基于平面片段的区域生长 (Patch-based region growing):
            - 将所有 supervoxel patch 存于 supervoxels 列表, 每个元素包含:
                * points: 点索引集合。
                * voxel_ids: 该 patch 覆盖的体素集合。
                * normal, d: 平面参数。
                * error: 生长过程中的最终法向变化。
            - 构建 patch 邻接关系:
                * 对每个体素 vid, 其内部所有 patch 互为邻居。
                * 对 vid 的 26 邻域体素中的 patch, 与 vid 内的 patch 互为邻居。
            - 按 error 从小到大排序 patch, 逐个作为 seed 做区域生长:
                * 若邻居 patch 与 seed 在法向夹角 < patch_angle, 质心到 seed 平面距离 < patch_distance,
                  则将其合并进 seed, 更新 seed 点集与 voxel_ids, 并重新拟合平面。
                * 同时将该邻居的邻居加入待检查队列, 实现多步生长。
            - 每次生长结束, 使用合并后的所有点重新拟合平面, 生成一个 Discontinuity。
    """

    def __init__(
            self,
            voxel_size: float = 0.05,
            ransac_distance: float = 0.05,
            ransac_angle: float = 5.0,
            min_plane_points: int = 30,
            edge_distance: float = 0.05,
            edge_angle: float = 5.0,
            min_edge_points: int = 30,
            min_edge_patch_points: int = 20,
            super_distance: float = 0.15,
            super_angle: float = 15.0,
            max_refit_error: float = 5.0,
            distance_step: float = 0.01,
            angle_step: float = 1.0,
            patch_distance: float = 0.30,
            patch_angle: float = 20.0,
            voxel_patch_thickness: float = None,  # 新增：厚度带半厚度
            voxel_patch_bitmap_B: float = None,  # 新增：bitmap 分辨率 B
    ):
        """
        输入变量:
            voxel_size: float
                体素划分网格尺寸(单位: m)。
            ransac_distance: float
                初始 RANSAC 平面拟合的距离阈值。
            ransac_angle: float
                初始 RANSAC 平面拟合的法向夹角阈值(度), 用于对 inlier 做二次筛选。
            min_plane_points: int
                体素内平面被接受为 voxel-patch 的最少点数。
            edge_distance: float
                边缘平面与邻域平面拼接的距离阈值 Mw。
            edge_angle: float
                边缘平面与邻域平面拼接的法向角度阈值 Nw(度)。
            min_edge_points: int
                启动边缘平面提取的 voxel 剩余点数阈值, 少于此值不做边缘检测。
            min_edge_patch_points: int
                单个 edge-patch 被接受的最少点数。
            super_distance: float
                超体素区域生长初始距离阈值。
            super_angle: float
                超体素区域生长初始法向角度阈值(度)。
            max_refit_error: float
                超体素生长过程中允许的最大法向变化(度)。
            distance_step: float
                超体素生长每轮缩小的距离步长。
            angle_step: float
                超体素生长每轮缩小的角度步长。
            patch_distance: float
                patch-level 区域生长时的距离阈值。
            patch_angle: float
                patch-level 区域生长时的法向角度阈值(度)。
            voxel_patch_thickness: float
                厚度带半厚度
            voxel_patch_bitmap_B: float
                bitmap 分辨率 B
        """
        super().__init__(name="Supervoxel")
        self.voxel_size = voxel_size
        self.ransac_distance = ransac_distance
        self.ransac_angle = ransac_angle
        self.min_plane_points = min_plane_points
        self.edge_distance = edge_distance
        self.edge_angle = edge_angle
        self.min_edge_points = min_edge_points
        self.min_edge_patch_points = min_edge_patch_points
        self.super_distance = super_distance
        self.super_angle = super_angle
        self.max_refit_error = max_refit_error
        self.distance_step = distance_step
        self.angle_step = angle_step
        self.patch_distance = patch_distance
        self.patch_angle = patch_angle
        self.voxel_patch_thickness = voxel_patch_thickness
        self.voxel_patch_bitmap_B = voxel_patch_bitmap_B

        self.logger = LoggerManager.GetLogger(self.__class__.__name__)

        # 预计算用于角度判断的cos值（注: 需将角度阈值从度数转换为余弦值比较）
        self._cos_edge_angle = math.cos(math.radians(self.edge_angle))
        self._cos_patch_angle = math.cos(math.radians(self.patch_angle))

    def _DetectImpl(self, point_cloud: PointCloud) -> List[Discontinuity]:
        """
        功能简介:
            在输入点云上执行 Supervoxel-RegionGrowing 平面检测, 返回一组 Discontinuity 结构面。
        实现思路:
            - 调用 _Voxelize 将点云划分为体素。
            - 在每个体素内执行局部 RANSAC, 生成初始 voxel-patch 和 remain_points。
            - 在 remain_points 中对 non-coplanar voxel 进行多次 RANSAC, 提取 edge-patch 并与邻域平面拼接。
            - 将所有 patch 作为种子, 在其邻域剩余点中执行超体素生长, 形成 supervoxel patch。
            - 对 supervoxel patch 执行基于邻接关系的区域生长, 输出合并后的 Discontinuity。
        输入变量:
            point_cloud: PointCloud
                输入的点云对象, 要求至少包含 points(N,3) 和 normals(N,3)。
        输出变量:
            discontinuities: List[Discontinuity]
                检测到的结构面列表。
        """

        def get_point_cloud_data(point_cloud: PointCloud):
            points = point_cloud.points

            # 将点的坐标和法向提取为 NumPy 数组，方便向量化计算
            coords = np.array([[p.x, p.y, p.z] for p in points], dtype=float)
            # 提取法向量，并标记法向是否有效 (非零)
            normals = []
            for p in points:
                nx, ny, nz = p.normal
                # 法向非零则认为有效
                if nx == 0 and ny == 0 and nz == 0:
                    normals.append([0, 0, 0])
                else:
                    # 将法向量归一化以确保角度计算准确
                    norm_len = math.sqrt(nx * nx + ny * ny + nz * nz) + 1e-12
                    normals.append((nx / norm_len, ny / norm_len, nz / norm_len))
            normals = np.array(normals, dtype=float)
            return coords, normals

        # -------------------------
        # main
        # -------------------------
        if o3d is None:
            # Open3D 库不可用，无法执行RANSAC平面提取
            self.logger.error("Open3D not found. SupervoxelDetector requires Open3D for plane segmentation.")
            return []

        coords, normals = get_point_cloud_data(point_cloud)  # (N, 3)
        num_points = coords.shape[0]
        normals_valid = ~np.isnan(normals).any(axis=1)
        if num_points == 0:
            self.logger.warning("Supervoxel: 输入点云为空，返回空结果.")
            return []
        else:
            self.logger.info(f"SupervoxelDetector starts: {num_points} points.")

        # -------------------------
        # Step 1: 点云体素化
        # -------------------------
        with Timer("Supervoxel: Voxelization", self.logger):
            pts_min = coords.min(axis=0)
            voxel_indices = np.floor((coords - pts_min) / self.voxel_size).astype(int)

            voxel_map: Dict[Tuple[int, int, int], List[int]] = {}
            point_voxel_map: List[Tuple[int, int, int]] = [None] * num_points

            for i, vid in enumerate(map(tuple, voxel_indices)):
                point_voxel_map[i] = vid
                if vid in voxel_map:
                    voxel_map[vid].append(i)
                else:
                    voxel_map[vid] = [i]

        self.logger.info(f"Voxelization: generated {len(voxel_map)} voxels with size {self.voxel_size:.3f} m.")

        # -------------------------
        # Step 2: 体素内局部 RANSAC 提取 voxel-patch
        # -------------------------
        clusters: List[Dict] = []  # 每个 cluster 是一个初始平面片 (voxel-patch 或 edge-patch)
        remain_points: Dict[Tuple[int, int, int], Set[int]] = {}
        with Timer("Supervoxel: Initial voxel-plane extraction", self.logger):
            for voxel_id, pts_indices_list in voxel_map.items():
                pts_indices = np.array(pts_indices_list, dtype=int)
                if pts_indices.size < self.min_plane_points:
                    remain_points[voxel_id] = set(pts_indices.tolist())
                    continue

                patch, remain_set = self._ExtractVoxelPatchByRansacBitmapLcc(
                    coords=coords,
                    normals=normals,
                    normals_valid=normals_valid,
                    pts_indices=pts_indices,
                    voxel_id=voxel_id,
                    num_iterations=1000,
                    connectivity=8,
                )

                if patch is None:
                    remain_points[voxel_id] = set(pts_indices.tolist())
                else:
                    clusters.append(patch)
                    remain_points[voxel_id] = remain_set

            for vid in voxel_map.keys():
                remain_points.setdefault(vid, set())
        self.logger.info(
            f"Initial voxel-plane extraction: {len(clusters)} plane seeds, "
            f"{sum(len(s) for s in remain_points.values())} remain points."
        )

        # -------------------------
        # Step 3: 边缘平面提取 (edge-patch) 与拼接
        # -------------------------
        with Timer("Supervoxel: Edge patch extraction", self.logger):
            new_clusters: List[Dict] = []

            for voxel_id, pts_set in list(remain_points.items()):
                if pts_set is None or len(pts_set) == 0:
                    continue
                if len(pts_set) <= self.min_edge_points:
                    # 剩余点过少, 不做边缘提取
                    continue

                testable = True
                while len(pts_set) > self.min_edge_points and testable:
                    # 注意：这里 pts_set 是 remain_points[voxel_id] 的动态剩余点
                    pts_indices = np.array(list(pts_set), dtype=int)

                    # 用与 Step2 相同的 “RANSAC + 厚度带 + bitmap-LCC” 提取 edge-patch
                    patch, edge_remain_set = self._ExtractVoxelPatchByRansacBitmapLcc(
                        coords=coords,
                        normals=normals,
                        normals_valid=normals_valid,
                        pts_indices=pts_indices,
                        voxel_id=voxel_id,
                        num_iterations=500,  # 与原 Step3 一致
                        connectivity=8,
                        min_patch_points=self.min_edge_patch_points,  # 关键：保持 Step3 的接受阈值
                    )

                    if patch is None:
                        # 未能得到足够稳定的 edge-patch，停止对该 voxel 的边缘提取
                        testable = False
                        break

                    # 由 patch 得到本次 edge-patch 的内点集合、平面参数（替代原 plane_model + inliers）
                    inlier_set = set(patch["points"])
                    patch_normal = np.array(patch["normal"], dtype=float)
                    pd = float(patch["d"])

                    # 兼容原有逻辑：edge_remain_set 已由函数返回（等价于 pts_set - inlier_set）
                    # edge_remain_set: Set[int]
                    if len(inlier_set) > self.min_edge_patch_points:
                        neighbor_voxels = self._get_neighbor_voxels(voxel_id)
                        best_sim = 100.0
                        best_cluster_idx = None

                        # 在邻域体素中查找已有平面 cluster
                        for neighbor_vid in neighbor_voxels:
                            for idx, cluster in enumerate(clusters):
                                if neighbor_vid not in cluster["voxel_ids"]:
                                    continue
                                plane_normal = np.array(cluster["normal"], dtype=float)
                                plane_d = cluster["d"]

                                pts_edge = coords[list(inlier_set)]
                                if pts_edge.shape[0] == 0:
                                    continue
                                distances = np.abs(np.dot(pts_edge, plane_normal) + plane_d)
                                mw = float(distances.mean()) if distances.size > 0 else float("inf")

                                # 计算 patch 平面与 cluster 平面的法向夹角
                                cos_angle = float(np.clip(
                                    np.abs(np.dot(patch_normal, plane_normal)), -1.0, 1.0
                                ))
                                nw = math.degrees(math.acos(cos_angle))

                                if mw < self.edge_distance and nw < self.edge_angle:
                                    # 计算相似性 P_w
                                    dist_norm = self.edge_distance if self.edge_distance > 0 else 1e-6
                                    ang_norm = self.edge_angle if self.edge_angle > 0 else 1e-6
                                    Pw = math.sqrt((mw / dist_norm) ** 2 + (nw / ang_norm) ** 2)
                                    if Pw < best_sim:
                                        best_sim = Pw
                                        best_cluster_idx = idx

                        if best_cluster_idx is not None:
                            # 拼接到已有平面 cluster
                            clusters[best_cluster_idx]["points"].update(inlier_set)
                            if voxel_id not in clusters[best_cluster_idx]["voxel_ids"]:
                                clusters[best_cluster_idx]["voxel_ids"].append(voxel_id)
                        else:
                            # 新建一个 edge-patch cluster
                            new_clusters.append({
                                "voxel_ids": [voxel_id],
                                "points": set(inlier_set),
                                "orig_points": set(inlier_set),
                                "normal": (float(patch_normal[0]), float(patch_normal[1]), float(patch_normal[2])),
                                "d": float(pd)
                            })

                        # 更新该体素剩余点（继续 while 尝试下一块 edge-patch）
                        pts_set = edge_remain_set
                    else:
                        # patch 点太少, 认为无法成面, 停止对该 voxel 的边缘提取
                        testable = False

                remain_points[voxel_id] = pts_set

            clusters.extend(new_clusters)

        self.logger.info(f"Edge patch extraction: clusters extended to {len(clusters)} seeds.")

        # -------------------------
        # Step 4: 超体素分割 (Supervoxel segmentation)
        # -------------------------
        supervoxels: List[Dict] = []

        with Timer("Supervoxel: Supervoxel segmentation", self.logger):
            for ci, cluster in enumerate(clusters):
                record_normal = np.array(cluster["normal"], dtype=float)
                record_d = float(cluster["d"])

                # 该 patch 覆盖的所有体素
                seed_voxels: Set[Tuple[int, int, int]] = set(cluster["voxel_ids"])

                # 构建候选体素区域: seed_voxels + 每个 seed_voxel 的 26 邻域
                neighbor_voxels: Set[Tuple[int, int, int]] = set()
                for sv in seed_voxels:
                    neighbor_voxels.add(sv)
                    neighbor_voxels.update(self._get_neighbor_voxels(sv))

                neighbor_remains: Dict[Tuple[int, int, int], np.ndarray] = {}
                for nv in neighbor_voxels:
                    pts_set = remain_points.get(nv)
                    if pts_set:
                        neighbor_remains[nv] = np.array(list(pts_set), dtype=int)
                    else:
                        neighbor_remains[nv] = np.array([], dtype=int)

                dist_th = self.super_distance
                ang_th = self.super_angle
                orientation_diff = float("inf")
                best_points_set: Set[int] = set()
                best_normal = record_normal.copy()
                best_d = record_d

                with Timer(f"Supervoxel growth for seed {ci}", self.logger):
                    while orientation_diff > self.max_refit_error and (dist_th > 0 or ang_th > 0):
                        current_points: Set[int] = set(cluster["orig_points"])
                        current_normal = record_normal.copy()
                        current_d = record_d
                        cos_ang_th = math.cos(math.radians(ang_th))

                        # 从候选区域中吸收符合当前阈值的 remain 点
                        for nv, nv_indices in neighbor_remains.items():
                            if nv_indices.size == 0:
                                continue

                            dists = np.abs(np.dot(coords[nv_indices], current_normal) + current_d)

                            valid_mask = normals_valid[nv_indices]
                            if not np.any(valid_mask):
                                continue
                            cosines = np.abs(np.dot(normals[nv_indices], current_normal))

                            mask = (dists < dist_th) & (cosines >= cos_ang_th) & valid_mask
                            if np.any(mask):
                                selected_indices = nv_indices[mask]
                                current_points.update(selected_indices.tolist())

                        if len(current_points) < 3:
                            break

                        pts_array = coords[list(current_points)]
                        centroid = pts_array.mean(axis=0)
                        cov = np.cov(pts_array - centroid, rowvar=False)
                        eig_vals, eig_vecs = np.linalg.eigh(cov)
                        new_normal = eig_vecs[:, int(np.argmin(eig_vals))]

                        if new_normal[2] < 0:
                            new_normal = -new_normal
                        new_d = -float(new_normal.dot(centroid))

                        cos_angle = float(np.clip(
                            np.abs(new_normal.dot(record_normal)), -1.0, 1.0
                        ))
                        orientation_diff = math.degrees(math.acos(cos_angle))

                        if orientation_diff <= self.max_refit_error:
                            best_points_set = set(current_points)
                            best_normal = new_normal
                            best_d = new_d
                            break

                        dist_th -= self.distance_step
                        ang_th -= self.angle_step
                        if dist_th < 0:
                            dist_th = 0.0
                        if ang_th < 0:
                            ang_th = 0.0

                # 构建 supervoxel patch 结果
                if len(best_points_set) == 0:
                    best_points_set = set(cluster["points"])
                    best_normal = record_normal
                    best_d = record_d
                    orientation_diff = 0.0

                # 根据最终点集重新统计体素覆盖范围
                best_voxel_ids: Set[Tuple[int, int, int]] = set()
                for pid in best_points_set:
                    vid = point_voxel_map[pid]
                    best_voxel_ids.add(vid)
                    # 从 remain_points 中移除这些点
                    if vid in remain_points and pid in remain_points[vid]:
                        remain_points[vid].discard(pid)

                supervoxels.append({
                    "points": best_points_set,
                    "voxel_ids": best_voxel_ids,
                    "normal": tuple(best_normal.tolist()),
                    "d": float(best_d),
                    "error": float(orientation_diff),
                })

        self.logger.info(f"Supervoxel segmentation: generated {len(supervoxels)} supervoxel patches.")

        # -------------------------
        # Step 5: Patch-based 区域生长
        # -------------------------
        discontinuities: List[Discontinuity] = []

        with Timer("Supervoxel: Patch-based region growing", self.logger):
            n_patches = len(supervoxels)
            if n_patches == 0:
                return []

            patch_neighbors: Dict[int, Set[int]] = {i: set() for i in range(n_patches)}
            voxel_to_patches: Dict[Tuple[int, int, int], List[int]] = {}

            # 体素到 patch 的映射
            for i, patch in enumerate(supervoxels):
                for vid in patch["voxel_ids"]:
                    voxel_to_patches.setdefault(vid, []).append(i)

            # 建立 patch 邻接关系
            for vid, plist in voxel_to_patches.items():
                # 同体素内 patch 互为邻居
                for p in plist:
                    for q in plist:
                        if p != q:
                            patch_neighbors[p].add(q)
                # 邻域体素内的 patch 互为邻居
                for nbr in self._get_neighbor_voxels(vid):
                    if nbr not in voxel_to_patches:
                        continue
                    for p in plist:
                        for q in voxel_to_patches[nbr]:
                            if p != q:
                                patch_neighbors[p].add(q)

            unmerged: Set[int] = set(range(n_patches))
            patch_order = sorted(unmerged, key=lambda i: supervoxels[i]["error"])

            while patch_order:
                seed_id = patch_order[0]
                patch_order.remove(seed_id)
                if seed_id not in unmerged:
                    continue

                seed_patch = supervoxels[seed_id]
                seed_normal = np.array(seed_patch["normal"], dtype=float)
                seed_d = float(seed_patch["d"])

                neighbor_set = {nid for nid in patch_neighbors[seed_id] if nid in unmerged}
                neighbor_list = sorted(list(neighbor_set), key=lambda j: supervoxels[j]["error"])

                merged_patch_ids = [seed_id]

                while neighbor_list:
                    j = neighbor_list[0]
                    neighbor_list.pop(0)
                    if j not in unmerged:
                        continue

                    j_patch = supervoxels[j]
                    j_normal = np.array(j_patch["normal"], dtype=float)

                    # 法向夹角
                    cos_angle = float(np.clip(
                        np.abs(seed_normal.dot(j_normal)), -1.0, 1.0
                    ))
                    ang_diff = math.degrees(math.acos(cos_angle))

                    # 距离差: 邻居 patch 质心到 seed 平面的距离
                    j_points = coords[list(j_patch["points"])]
                    if j_points.size > 0:
                        j_centroid = j_points.mean(axis=0)
                        dist_diff = abs(seed_normal.dot(j_centroid) + seed_d)
                    else:
                        dist_diff = 0.0

                    if ang_diff < self.patch_angle and dist_diff < self.patch_distance:
                        # 合并 patch j 至 seed
                        unmerged.discard(j)
                        merged_patch_ids.append(j)

                        seed_patch["points"].update(j_patch["points"])
                        seed_patch["voxel_ids"].update(j_patch["voxel_ids"])

                        # 重新拟合 seed_patch 平面
                        pts_all = coords[list(seed_patch["points"])]
                        centroid_all = pts_all.mean(axis=0)
                        cov_all = np.cov(pts_all - centroid_all, rowvar=False)
                        eig_vals_all, eig_vecs_all = np.linalg.eigh(cov_all)
                        normal_all = eig_vecs_all[:, int(np.argmin(eig_vals_all))]
                        if normal_all[2] < 0:
                            normal_all = -normal_all
                        seed_normal = normal_all
                        seed_d = -float(normal_all.dot(centroid_all))
                        seed_patch["normal"] = tuple(seed_normal.tolist())
                        seed_patch["d"] = seed_d

                        # 将 j 的邻居加入待检查队列
                        for k in patch_neighbors[j]:
                            if k in unmerged and k not in neighbor_set:
                                neighbor_set.add(k)
                                neighbor_list.append(k)
                        neighbor_list = sorted(neighbor_list, key=lambda x: supervoxels[x]["error"])

                # 生成最终平面 Discontinuity
                # 收集合并的所有片段的点索引
                all_points = set()
                segments_list: List[Segment] = []
                for pid in merged_patch_ids:
                    all_points |= supervoxels[pid]["points"]
                    # 为每个片段创建 Segment 对象 (使用合并前各自的平面参数)
                    plane_normal = supervoxels[pid]["normal"]
                    plane_d = supervoxels[pid]["d"]
                    # 计算片段平面的质心和RMSE
                    pts = coords[list(supervoxels[pid]["points"])]
                    centroid = pts.mean(axis=0) if pts.shape[0] > 0 else np.array([0, 0, 0], dtype=float)
                    # 计算RMSE: 点到该片段平面的平均距离
                    distances = np.abs(np.dot(pts, plane_normal) + plane_d)
                    rmse = math.sqrt((distances ** 2).mean()) if distances.size > 0 else 0.0
                    plane_obj = Plane(plane_normal, plane_d, tuple(centroid.tolist()),
                                      inlier_indices=list(supervoxels[pid]["points"]), rmse=rmse)
                    segments_list.append(Segment(plane_obj, list(supervoxels[pid]["points"])))

                # 统一整个平面的平面参数：对all_points拟合
                pts_all = coords[list(all_points)]
                centroid_all = pts_all.mean(axis=0)
                cov_all = np.cov(pts_all - centroid_all, rowvar=False) if pts_all.shape[0] >= 3 else None
                if cov_all is not None:
                    eig_vals_all, eig_vecs_all = np.linalg.eigh(cov_all)
                    plane_normal = eig_vecs_all[:, np.argmin(eig_vals_all)]
                else:
                    # 点数过少无法拟合平面，用第一个segment的平面
                    plane_normal = np.array(segments_list[0].plane.normal)

                # 假设 normals 有效
                patch_normals = normals[list(all_points)]
                valid_mask = ~np.isnan(patch_normals).any(axis=1)
                if np.any(valid_mask):
                    mean_normal = patch_normals[valid_mask].mean(axis=0)
                    if mean_normal.dot(plane_normal) < 0:
                        plane_normal = -plane_normal

                d_val = -float(plane_normal.dot(centroid_all))
                # 计算整体平面的 dip 和 dip_direction
                norm_len = math.sqrt(plane_normal.dot(plane_normal)) + 1e-12
                plane_norm_unit = plane_normal / norm_len
                nx, ny, nz = plane_norm_unit.tolist()
                # # 将法向统一到上半球
                # if nz < 0:
                #     nx, ny, nz = -nx, -ny, -nz
                horizontal = math.sqrt(nx * nx + ny * ny)
                dip = math.degrees(math.atan2(horizontal, abs(nz)))
                azimuth = math.degrees(math.atan2(nx, ny))
                if azimuth < 0:
                    azimuth += 360.0
                dip_dir = azimuth
                # 计算整体平面粗糙度 (RMSE)
                distances_all = np.abs(np.dot(pts_all, [nx, ny, nz]) + d_val)
                overall_rmse = math.sqrt((distances_all ** 2).mean()) if distances_all.size > 0 else 0.0
                plane_obj = Plane((nx, ny, nz), d_val, tuple(centroid_all.tolist()), inlier_indices=list(all_points),
                                  rmse=overall_rmse)
                # 构造 Discontinuity 对象
                discontinuity = Discontinuity(segments_list, plane_obj, dip, dip_dir, roughness=overall_rmse,
                                              algorithm_name=self.name)
                discontinuities.append(discontinuity)

        self.logger.info(f"Patch-based region growing complete: {len(discontinuities)} planes detected.")
        return discontinuities

    def _get_neighbor_voxels(self, voxel_id: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """
        功能简介:
            给定体素索引 (vx, vy, vz), 返回其 3x3x3 邻域中除自身外的 26 个邻居体素索引。
        实现思路:
            - 遍历 dx, dy, dz ∈ {-1, 0, 1} 的所有组合。
            - 排除 (0, 0, 0), 其余 (vx+dx, vy+dy, vz+dz) 视为邻居。
        输入变量:
            voxel_id: Tuple[int, int, int]
                体素索引 (vx, vy, vz)。
        输出变量:
            neighbors: List[Tuple[int, int, int]]
                邻域体素索引列表。
        """
        vx, vy, vz = voxel_id
        neighbors: List[Tuple[int, int, int]] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    neighbors.append((vx + dx, vy + dy, vz + dz))
        return neighbors

    def _ExtractVoxelPatchByRansacBitmapLcc(
            self,
            coords: np.ndarray,
            normals: np.ndarray,
            normals_valid: np.ndarray,
            pts_indices: np.ndarray,
            voxel_id: Tuple[int, int, int],
            num_iterations: int = 1000,
            connectivity: int = 8,
            min_patch_points: int = None,
    ) -> Tuple[Optional[Dict], Set[int]]:
        """
        功能简介:
            在单个 voxel 内执行 Open3D RANSAC 拟合平面，并在“厚度带”内构建二维投影 bitmap，
            取最大连通域(LCC)对应的点作为 voxel-patch 内点。

        实现思路(详细):
            1) Open3D segment_plane 得到候选平面；
            2) 在 voxel 全体点上取厚度带兼容点: |n·p + d| <= band；
            3) 可选法向角度约束: |n(p)·n_plane| >= cos(ransac_angle)，且 normals_valid=True；
            4) 将兼容点投影到平面局部 2D(u,v)，按分辨率 B 栅格化成稀疏 bitmap(cell->point_ids)；
            5) 在 cell 图上做连通域(4/8 邻域)，取“点数最多”的连通域；
            6) 若 LCC 点数 >= min_plane_points，则输出 patch dict，否则输出 None。

        输入:
            coords: (N,3)
            normals: (N,3) 单位法向
            normals_valid: (N,) bool
            pts_indices: (M,) voxel 内全局点索引
            voxel_id: 体素 id
            num_iterations: RANSAC 迭代次数
            connectivity: 4 或 8

        输出:
            patch: dict 或 None（字段与 clusters.append(...) 兼容）
            remain_set: voxel 内剩余点（全局索引）
        """
        if min_patch_points is None:
            min_patch_points = int(self.min_plane_points)
        else:
            min_patch_points = int(min_patch_points)

        if pts_indices.size == 0:
            return None, set()

        if connectivity not in (4, 8):
            raise ValueError("connectivity 仅支持 4 或 8")

        # thickness 与 B 的取值
        band = float(self.voxel_patch_thickness) if self.voxel_patch_thickness is not None else float(
            self.ransac_distance)
        B = self.voxel_patch_bitmap_B
        if B is None:
            # 【推测】默认回退到 voxel_size，以保证旧工程可直接运行；推荐显式设置 B
            B = float(self.voxel_size)
        B = float(B)
        if B <= 0:
            raise ValueError("voxel_patch_bitmap_B(B) 必须 > 0")

        # --- 1) Open3D RANSAC ---
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords[pts_indices])
        plane_model, inliers_local = pcd.segment_plane(
            distance_threshold=float(self.ransac_distance),
            ransac_n=3,
            num_iterations=int(num_iterations),
        )
        inliers_local = np.asarray(inliers_local, dtype=int)
        if inliers_local.size == 0:
            return None, set(pts_indices.tolist())

        # --- 2) 平面参数归一化 ---
        a, b, c, d = plane_model
        norm_len = math.sqrt(a * a + b * b + c * c) + 1e-12
        a, b, c, d = a / norm_len, b / norm_len, c / norm_len, d / norm_len
        n_plane = np.array([a, b, c], dtype=float)

        # --- 3) 厚度带兼容点筛选（对 voxel 全体点，而不是仅 RANSAC inliers）---
        v_xyz = coords[pts_indices]  # (M,3)
        dist = np.abs(v_xyz @ n_plane + d)  # (M,)
        band_mask = dist <= band

        if self.ransac_angle > 0:
            cos_th = math.cos(math.radians(float(self.ransac_angle)))
            v_global = pts_indices
            v_valid = normals_valid[v_global]
            if np.any(v_valid):
                cosines = np.abs(normals[v_global] @ n_plane)
            else:
                cosines = np.zeros(v_global.shape[0], dtype=float)
            angle_mask = (cosines >= cos_th) & v_valid
            compat_mask = band_mask & angle_mask
        else:
            compat_mask = band_mask
        # compat_mask = band_mask

        compat_local = np.nonzero(compat_mask)[0]
        if compat_local.size == 0:
            return None, set(pts_indices.tolist())

        compat_global = pts_indices[compat_local]  # (K,)
        compat_xyz = coords[compat_global]  # (K,3)

        # --- 4) 构造平面局部 2D 基，投影并稀疏栅格化 ---
        p0 = -float(d) * n_plane
        if abs(n_plane[2]) < 0.9:
            ref = np.array([0.0, 0.0, 1.0], dtype=float)
        else:
            ref = np.array([0.0, 1.0, 0.0], dtype=float)
        u = np.cross(ref, n_plane)
        u = u / (np.linalg.norm(u) + 1e-12)
        v = np.cross(n_plane, u)  # 已单位

        rel = compat_xyz - p0[None, :]
        x2 = rel @ u
        y2 = rel @ v
        x_min = float(np.min(x2))
        y_min = float(np.min(y2))
        ix = np.floor((x2 - x_min) / B).astype(np.int64)
        iy = np.floor((y2 - y_min) / B).astype(np.int64)

        cell_to_points: Dict[Tuple[int, int], List[int]] = {}
        for k in range(compat_global.shape[0]):
            key = (int(ix[k]), int(iy[k]))
            cell_to_points.setdefault(key, []).append(int(compat_global[k]))

        cells = list(cell_to_points.keys())
        if len(cells) == 0:
            return None, set(pts_indices.tolist())

        # --- 5) 最大连通域(LCC) ---
        if connectivity == 4:
            nbr = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        elif connectivity == 8:
            nbr = [
                (1, 0), (-1, 0), (0, 1), (0, -1),
                (1, 1), (1, -1), (-1, 1), (-1, -1),
            ]

        cell_set = set(cells)
        visited: Set[Tuple[int, int]] = set()
        best_cells: Set[Tuple[int, int]] = set()
        best_cnt = 0

        for st in cells:
            if st in visited:
                continue
            stack = [st]
            visited.add(st)
            comp_cells: Set[Tuple[int, int]] = set()
            comp_cnt = 0
            while stack:
                cur = stack.pop()
                comp_cells.add(cur)
                comp_cnt += len(cell_to_points[cur])
                cx, cy = cur
                for dx, dy in nbr:
                    nb_cell = (cx + dx, cy + dy)
                    if nb_cell in cell_set and nb_cell not in visited:
                        visited.add(nb_cell)
                        stack.append(nb_cell)
            if comp_cnt > best_cnt:
                best_cnt = comp_cnt
                best_cells = comp_cells

        inlier_set: Set[int] = set()
        for cc in best_cells:
            inlier_set.update(cell_to_points[cc])

        # --- 6) 输出 patch / remain ---
        if len(inlier_set) < min_patch_points:
            return None, set(pts_indices.tolist())

        remain_set = set(pts_indices.tolist()) - inlier_set
        patch = {
            "voxel_ids": [voxel_id],
            "points": set(inlier_set),
            "orig_points": set(inlier_set),
            "normal": (float(a), float(b), float(c)),
            "d": float(d),
        }
        return patch, remain_set


if __name__ == "__main__":
    # 简单命令行测试: python detector_supervoxel.py <point_cloud_file>
    import sys
    from ..io_pointcloud import PointCloudIO

    if len(sys.argv) < 2:
        print("Usage: python detector_supervoxel.py <point_cloud_file>")
        sys.exit(0)

    pc_path = sys.argv[1]
    pc = PointCloudIO.ReadPointCloud(pc_path)
    detector = SupervoxelDetector()
    discontinuities = detector.DetectPlanes(pc)
    print(f"Detected {len(discontinuities)} planes.")
