# RockDiscontinuity/src/rock_discon_extract/algorithms/detector_supervoxel.py

from typing import List, Dict, Tuple, Set
import math
import numpy as np

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
    基于超体素划分和区域生长的高精度平面检测算法 (Supervoxel-RegionGrowing).

    算法流程概要:
        1. **空间体素划分**: 按指定网格尺寸将点云划分为体素子空间，加速局部处理。
        2. **局部共面性检测**: 对每个体素内采用 RANSAC 提取最佳平面作为生长单元 (Growth Unit)，
           若平面内点数量低于阈值则视为无有效平面，该体素点保留为剩余点。
        3. **边界细节提取**: 针对含有多个平面的体素(非共面体素)的剩余点，迭代执行 RANSAC 提取边缘小平面 (Edge Patch)。
           对于每个提取出的边缘平面:
             - 若内点数足够大，则根据距离和法向相似性 (阈值 Mw、Nw) 将其合并到相邻体素的生长单元，或作为新生长单元保存:contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}。
             - 若内点数过少，则停止该体素的边缘提取:contentReference[oaicite:2]{index=2}:contentReference[oaicite:3]{index=3}。
        4. **超体素分割**: 将每个生长单元作为初始种子，与其邻近体素的剩余离散点进行局部区域生长，形成超体素平面片段 (Supervoxel Patch):contentReference[oaicite:4]{index=4}:contentReference[oaicite:5]{index=5}。
           区域生长采用逐步收缩距离和角度阈值的方法限制误差累积: 初始阈值较大 (superdis, superang)，逐次减小阈值 (disstep, angstep)，直至平面法向变动小于 Maxerror:contentReference[oaicite:6]{index=6}:contentReference[oaicite:7]{index=7}。这样可精细控制增长过程，保留更多局部细节:contentReference[oaicite:8]{index=8}:contentReference[oaicite:9]{index=9}。
        5. **基于片段的区域生长**: 将得到的所有超体素平面片段按照累积误差从小到大排序，依次作为种子，将其相邻的未合并片段按距离和角度阈值 (growdis, growang) 合并，生成完整的平面:contentReference[oaicite:10]{index=10}:contentReference[oaicite:11]{index=11}。该过程不会产生新的边界，仅融合原有片段边界:contentReference[oaicite:12]{index=12}。

    输入参数:
        voxel_size: float = 0.05
            体素划分的网格尺寸 (单位: 米). 默认为 0.05m，如无文献明确指出则采用此点云分辨率:contentReference[oaicite:13]{index=13}。
        ransac_distance: float = 0.05
            RANSAC 局部平面拟合的距离阈值 (randis)，用于共面性检测和边缘平面提取:contentReference[oaicite:14]{index=14}。
        ransac_angle: float = 5.0
            RANSAC 平面拟合的法向角度阈值 (ranang, 当前实现未直接用到，可用于避免提取近似平行的重复平面)。
        min_plane_points: int = 30
            接受为有效平面的最少点数 (Minplane)。体素内平面若少于此点数则视为无效，所有点作为剩余点保留。
        edge_distance: float = 0.05
            边缘区域平面合并的距离阈值 Mw (edgedis):contentReference[oaicite:15]{index=15}:contentReference[oaicite:16]{index=16}。
        edge_angle: float = 5.0
            边缘平面合并的法向角阈值 Nw (edgeang):contentReference[oaicite:17]{index=17}:contentReference[oaicite:18]{index=18}。
        min_edge_points: int = 30
            触发边缘迭代检测的剩余点数量下限 (Minedge):contentReference[oaicite:19]{index=19}。仅当体素剩余点数 > 此值时才尝试提取边缘平面。
        min_edge_patch_points: int = 20
            边缘平面可接受的最少点数阈值 (Ranedge):contentReference[oaicite:20]{index=20}。若提取的边缘平面内点数 <= 此值，则停止该体素的边缘提取:contentReference[oaicite:21]{index=21}:contentReference[oaicite:22]{index=22}。
        super_distance: float = 0.15
            超体素生长初始距离阈值 (superdis):contentReference[oaicite:23]{index=23}。较大的初值允许充分合并邻域离散点，逐步收缩提高精度。
        super_angle: float = 15.0
            超体素生长初始法向角阈值 (superang):contentReference[oaicite:24]{index=24} (单位: 度)。
        max_refit_error: float = 5.0
            超体素生长迭代的最大法向改变阈值 (Maxerror，单位: 度):contentReference[oaicite:26]{index=26}。当新增点导致平面法向改变小于此值时，停止迭代。
        distance_step: float = 0.01
            超体素生长阈值收缩步长 (距离). 每次迭代后距离阈值减少量。
        angle_step: float = 1.0
            超体素生长阈值收缩步长 (角度, 度). 每次迭代后角度阈值减少量。
        patch_distance: float = 0.30
            平面片段合并的距离阈值 (growdis):contentReference[oaicite:27]{index=27}。用于最终 Patch-Level 区域生长时判断片段是否属于同一平面。
        patch_angle: float = 20.0
            平面片段合并的法向角阈值 (growang, 度):contentReference[oaicite:28]{index=28}。

    输出:
        DetectPlanes 返回 List[Discontinuity]，其中每个 Discontinuity 表示一个检测到的岩体结构面，包括一个或多个 Segment（连通片段）及其平面参数。
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
            patch_angle: float = 20.0
    ):
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

        # 预计算用于角度判断的cos值（注: 需将角度阈值从度数转换为余弦值比较）
        self._cos_edge_angle = math.cos(math.radians(self.edge_angle))
        self._cos_patch_angle = math.cos(math.radians(self.patch_angle))

    def _DetectImpl(self, point_cloud: PointCloud) -> List[Discontinuity]:
        """
        在输入点云上执行 Supervoxel-RegionGrowing 算法，返回检测出的结构面列表。
        要求 PointCloud 每个点已预先计算法向量，否则无法进行法向角约束。
        """
        if o3d is None:
            # Open3D 库不可用，无法执行RANSAC平面提取
            self.logger.error("Open3D not found. SupervoxelDetector requires Open3D for plane segmentation.")
            return []

        points = point_cloud.points
        num_points = len(points)
        if num_points == 0:
            self.logger.warning("Supervoxel: 输入点云为空，返回空结果.")
            return []

        # 将点的坐标和法向提取为 NumPy 数组，方便向量化计算
        coords = np.array([[p.x, p.y, p.z] for p in points], dtype=float)
        # 提取法向量，并标记法向是否有效 (非零)
        normals = []
        normals_valid = []
        for p in points:
            if hasattr(p, 'normal') and p.normal is not None:
                nx, ny, nz = p.normal
                # 法向非零则认为有效
                if nx == 0 and ny == 0 and nz == 0:
                    normals_valid.append(False)
                else:
                    normals_valid.append(True)
                # 将法向量归一化以确保角度计算准确
                norm_len = math.sqrt(nx * nx + ny * ny + nz * nz) + 1e-12
                normals.append((nx / norm_len, ny / norm_len, nz / norm_len))
            else:
                normals_valid.append(False)
                normals.append((0.0, 0.0, 0.0))
        normals = np.array(normals, dtype=float)
        normals_valid = np.array(normals_valid, dtype=bool)

        # Step 1: 体素划分
        with Timer("Supervoxel: Voxelization", self.logger):
            # 计算点云边界框，用于规范体素坐标
            pts_min = coords.min(axis=0)
            # 使用最小坐标为原点，将每点的体素索引计算为整数三元组
            voxel_indices = np.floor((coords - pts_min) / self.voxel_size).astype(int)
            # 建立 voxel_id -> 点索引 列表 的映射
            voxel_map: Dict[Tuple[int, int, int], List[int]] = {}
            point_voxel_map: List[Tuple[int, int, int]] = [None] * num_points
            for i, vid in enumerate(map(tuple, voxel_indices)):
                point_voxel_map[i] = vid
                if vid in voxel_map:
                    voxel_map[vid].append(i)
                else:
                    voxel_map[vid] = [i]
            # 计算平均点数
            voxel_avg_pts = np.array([len(voxel_indices) for vid, voxel_indices in voxel_map.items()], dtype=int).mean()
        self.logger.info(f"Voxelization: generated {len(voxel_map)} voxels with size {self.voxel_size:.3f}m.")
        self.logger.info(f"Voxelization: average of points in voxels is {voxel_avg_pts}.")

        # Step 2: 局部共面性检测 (体素内 RANSAC 拟合平面)
        clusters = []  # 用于保存所有初始平面生长单元 (grow_units)
        remain_points: Dict[Tuple[int, int, int], Set[int]] = {}  # 体素剩余点映射 (集合形式便于后续操作)
        with Timer("Supervoxel: Initial plane extraction", self.logger):
            for voxel_id, pts_indices in voxel_map.items():
                pts_indices = np.array(pts_indices, dtype=int)
                if pts_indices.size < self.min_plane_points:
                    # 点数过少，不执行平面提取，所有点作为剩余点
                    remain_points[voxel_id] = set(pts_indices.tolist())
                    continue
                # 使用 Open3D RANSAC 提取该体素内的最佳拟合平面
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(coords[pts_indices])
                plane_model, inliers = pcd.segment_plane(distance_threshold=self.ransac_distance,
                                                         ransac_n=3, num_iterations=1000)
                inliers = np.array(inliers, dtype=int)
                if inliers.size == 0:
                    # 未找到平面（理论上不会发生，除非RANSAC故障），将所有点视为剩余
                    remain_points[voxel_id] = set(pts_indices.tolist())
                    continue
                # 将局部平面内点映射回全局索引
                plane_inliers_global = pts_indices[inliers]
                # 计算剩余离散点
                inlier_set = set(plane_inliers_global.tolist())
                remain_set = set(pts_indices.tolist()) - inlier_set
                # 如果平面内点少于最小数量阈值，则不作为有效平面
                if len(inlier_set) < self.min_plane_points:
                    # 所有点并入剩余集合
                    remain_points[voxel_id] = set(pts_indices.tolist())
                else:
                    # 存储生长单元 (初始平面片段)
                    # 确保平面法向量规范化，并指向上半球 (nz为正)，以便统一倾角方向:contentReference[oaicite:29]{index=29}:contentReference[oaicite:30]{index=30}
                    a, b, c, d = plane_model  # 平面方程 ax+by+cz+d=0
                    norm_len = math.sqrt(a * a + b * b + c * c) + 1e-12
                    a, b, c, d = a / norm_len, b / norm_len, c / norm_len, d / norm_len
                    # if c < 0:
                    #     # 翻转法向使其朝向上方
                    #     a, b, c, d = -a, -b, -c, -d
                    # 记录该平面生长单元的信息
                    clusters.append({
                        "voxel_ids": [voxel_id],
                        "points": set(plane_inliers_global.tolist()),
                        "orig_points": set(plane_inliers_global.tolist()),  # 原始种子平面点集
                        "normal": (a, b, c),
                        "d": d
                    })
                    # 更新剩余点
                    remain_points[voxel_id] = remain_set
            # 若某体素在 voxel_map 中存在但未在 remain_points 中记录，补充空集合
            for vid in voxel_map.keys():
                remain_points.setdefault(vid, set())
        self.logger.info(
            f"Initial plane extraction: {len(clusters)} plane seeds found, remaining points in {sum(len(s) for s in remain_points.values())} points.")

        # Step 3: 边界信息计算 (边缘平面提取与合并)
        with Timer("Supervoxel: Edge patch extraction", self.logger):
            # 遍历每个体素的剩余点集
            new_clusters = []  # 保存新提取的边缘平面片段
            for voxel_id, pts_set in list(remain_points.items()):
                if pts_set is None or len(pts_set) == 0:
                    continue  # 无剩余点
                # 若剩余点数量不足阈值，则跳过边缘提取
                if len(pts_set) <= self.min_edge_points:
                    continue
                testable = True
                # 对该体素进行迭代边缘平面提取
                while len(pts_set) > self.min_edge_points and testable:
                    pts_indices = np.array(list(pts_set), dtype=int)
                    # 用 Open3D 在剩余点中提取平面 (边缘patch)
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(coords[pts_indices])
                    plane_model, inliers = pcd.segment_plane(distance_threshold=self.ransac_distance,
                                                             ransac_n=3, num_iterations=500)
                    inliers = np.array(inliers, dtype=int)
                    if inliers.size == 0:
                        break  # 无法再提取平面
                    patch_inliers_global = pts_indices[inliers]
                    # 计算该边缘平面剩余点
                    inlier_set = set(patch_inliers_global.tolist())
                    edge_remain_set = pts_set - inlier_set
                    # 边缘平面点数
                    if len(inlier_set) > self.min_edge_patch_points:
                        # 计算与邻近生长单元的相似性
                        neighbor_voxels = self._get_neighbor_voxels(voxel_id)
                        best_sim = 100
                        best_cluster_idx = None
                        # 逐一检查邻居体素的生长单元
                        for neighbor_vid in neighbor_voxels:
                            # 查找该邻域体素对应的生长单元(平面)
                            for idx, cluster in enumerate(clusters):  # 在整个cluster里找?为什么不建立map
                                if neighbor_vid in cluster["voxel_ids"]:
                                    # 取该平面参数用于相似性计算
                                    plane_normal = cluster["normal"]
                                    plane_d = cluster["d"]
                                    # 计算边缘patch与该平面的平均距离 mw:contentReference[oaicite:31]{index=31}:contentReference[oaicite:32]{index=32}
                                    distances = np.abs(np.dot(coords[list(inlier_set)], plane_normal) + plane_d)
                                    mw = distances.mean() if len(distances) > 0 else float('inf')
                                    # 计算法向夹角 nw:contentReference[oaicite:33]{index=33}:contentReference[oaicite:34]{index=34}
                                    # plane_normal已归一化，需计算patch平面的法向
                                    pa, pb, pc, pd = plane_model
                                    norm_len = math.sqrt(pa * pa + pb * pb + pc * pc) + 1e-12
                                    pa, pb, pc, pd = pa / norm_len, pb / norm_len, pc / norm_len, pd / norm_len
                                    # 取绝对值以表示最小夹角
                                    cos_angle = abs(pa * plane_normal[0] + pb * plane_normal[1] + pc * plane_normal[2])
                                    # 避免浮点误差域外
                                    cos_angle = max(min(cos_angle, 1.0), -1.0)
                                    nw = math.degrees(math.acos(cos_angle))
                                    # 检查距离和角度阈值
                                    if mw < self.edge_distance and nw < self.edge_angle:
                                        # 计算相似性指标 Pw:contentReference[oaicite:35]{index=35}:contentReference[oaicite:36]{index=36}
                                        # 将 mw 和 nw 正规化到阈值范围
                                        Pw = math.sqrt(
                                            (mw / (self.edge_distance if self.edge_distance != 0 else 1e-6)) ** 2 +
                                            (nw / (self.edge_angle if self.edge_angle != 0 else 1e-6)) ** 2)
                                        if Pw < best_sim:
                                            best_sim = Pw
                                            best_cluster_idx = idx
                        if best_cluster_idx is not None:
                            # 将该边缘patch合并到相邻最佳平面生长单元中:contentReference[oaicite:37]{index=37}:contentReference[oaicite:38]{index=38}
                            clusters[best_cluster_idx]["points"].update(inlier_set)
                            # 记录该邻域体素也包含此patch所在体素
                            if voxel_id not in clusters[best_cluster_idx]["voxel_ids"]:
                                clusters[best_cluster_idx]["voxel_ids"].append(voxel_id)
                            # 注意: 此处不重新拟合平面参数，以原生长单元平面为准
                        else:
                            # 无相近平面，则将该patch作为新平面生长单元:contentReference[oaicite:39]{index=39}
                            # 归一化 patch 平面参数并朝上
                            pa, pb, pc, pd = plane_model
                            norm_len = math.sqrt(pa * pa + pb * pb + pc * pc) + 1e-12
                            pa, pb, pc, pd = pa / norm_len, pb / norm_len, pc / norm_len, pd / norm_len
                            if pc < 0:
                                pa, pb, pc, pd = -pa, -pb, -pc, -pd
                            new_clusters.append({
                                "voxel_ids": [voxel_id],
                                "points": set(inlier_set),
                                "orig_points": set(inlier_set),  # 新patch本身作为种子
                                "normal": (pa, pb, pc),
                                "d": pd
                            })
                        # 更新该体素剩余点集，并继续迭代检测
                        pts_set = edge_remain_set
                    else:
                        # 提取的patch太小，停止边缘提取:contentReference[oaicite:40]{index=40}
                        # 将该patch点重新并入剩余
                        pts_set = pts_set  # （保持不变，相当于不移除这些点）
                        testable = False
                # 更新全局 remain_points
                remain_points[voxel_id] = pts_set
            # 将新提取的边缘patch合并到clusters列表
            if new_clusters:
                clusters.extend(new_clusters)
        self.logger.info(
            f"Edge patch extraction: added {len(new_clusters)} edge patches, total planes now {len(clusters)}.")

        # Step 4: 超体素分割 (以每个平面生长单元为种子扩张)
        supervoxels = []  # 存储超体素平面片段结果
        with Timer("Supervoxel: Supervoxel segmentation", self.logger):
            for ci, cluster in enumerate(clusters):
                # 每个生长单元作为初始种子
                # 记录原始平面的法向用于误差评估
                record_normal = cluster["normal"]
                record_voxel = cluster["voxel_ids"][0]  # 选用初始平面所在体素
                # 获取该体素的直接邻域体素（3x3x3 邻域，不含自身）
                neighbor_voxels = self._get_neighbor_voxels(record_voxel)
                # 备份邻域体素的剩余点（这些将在迭代中重复使用）
                neighbor_remains: Dict[Tuple[int, int, int], np.ndarray] = {}
                for nv in neighbor_voxels:
                    if nv in remain_points and remain_points[nv]:
                        neighbor_remains[nv] = np.array(list(remain_points[nv]), dtype=int)
                    else:
                        neighbor_remains[nv] = np.array([], dtype=int)
                # 初始化动态阈值和法向差
                dist_th = self.super_distance
                ang_th = self.super_angle
                orientation_diff = float('inf')
                best_points_set: Set[int] = set()  # 保存最终选定的点集合
                best_normal = record_normal
                best_d = cluster["d"]
                # 迭代收缩生长阈值，直至法向改变小于阈值或阈值缩减完毕
                with Timer(f"Supervoxel growth for seed {ci}", self.logger):
                    while orientation_diff > self.max_refit_error and (dist_th > 0 or ang_th > 0):
                        # 每次迭代从原始种子平面重新开始
                        current_points = set(cluster["orig_points"])
                        current_normal = record_normal
                        current_d = cluster["d"]
                        cos_ang_th = math.cos(math.radians(ang_th))
                        # 尝试将邻域所有剩余点中符合当前阈值条件的加入平面
                        for nv, nv_indices in neighbor_remains.items():
                            if nv_indices.size == 0:
                                continue
                            # 计算每个候选点到当前平面的距离
                            dists = np.abs(np.dot(coords[nv_indices], current_normal) + current_d)
                            # 计算每个候选点法向与平面法向的夹角 (以cos比较)
                            # 若点无法向则略过
                            valid_mask = normals_valid[nv_indices]
                            if not np.any(valid_mask):
                                continue
                            cosines = np.abs(np.dot(normals[nv_indices], current_normal))
                            # 筛选满足距离和角度阈值的点
                            mask = (dists < dist_th) & (cosines >= cos_ang_th) & valid_mask
                            if np.any(mask):
                                # 将符合条件的点加入当前平面集
                                selected_indices = nv_indices[mask]
                                current_points.update(selected_indices.tolist())
                        # 拟合包含新增点的平面，计算法向变化
                        if len(current_points) < 3:
                            # 点数不足三点无法拟合，退出
                            break
                        pts_array = coords[list(current_points)]
                        centroid = pts_array.mean(axis=0)
                        # PCA 计算平面法向 (协方差矩阵最小特征向量)
                        cov = np.cov(pts_array - centroid, rowvar=False)
                        eig_vals, eig_vecs = np.linalg.eigh(cov)
                        # 最小特征值对应的特征向量即法向方向
                        new_normal = eig_vecs[:, np.argmin(eig_vals)]
                        # 确保法向朝上
                        if new_normal[2] < 0:
                            new_normal = -new_normal
                        # 计算平面方程截距 d: -n·centroid
                        new_d = -float(new_normal.dot(centroid))
                        # 计算与原始法向的夹角差
                        cos_angle = abs(new_normal.dot(record_normal))
                        cos_angle = max(min(cos_angle, 1.0), -1.0)
                        orientation_diff = math.degrees(math.acos(cos_angle))
                        if orientation_diff <= self.max_refit_error:
                            # 达到精度要求，保存结果
                            best_points_set = set(current_points)
                            best_normal = tuple(new_normal.tolist())
                            best_d = new_d
                            break
                        # 若未达到精度，缩小阈值继续迭代
                        dist_th -= self.distance_step
                        ang_th -= self.angle_step
                        if dist_th < 0:
                            dist_th = 0.0
                        if ang_th < 0:
                            ang_th = 0.0
                # 构建超体素平面片段结果
                if len(best_points_set) == 0:
                    # 若未能扩张任何点，则仍使用原始平面
                    best_points_set = cluster["points"]
                    best_normal = cluster["normal"]
                    best_d = cluster["d"]
                    orientation_diff = 0.0
                supervoxels.append({
                    "points": best_points_set,
                    "voxel_ids": set(cluster["voxel_ids"]),
                    "normal": best_normal,
                    "d": best_d,
                    "error": orientation_diff,
                })
                # 将该平面片段的所有点标记为已分配（从剩余集中移除）
                for pid in best_points_set:
                    vid = point_voxel_map[pid]
                    if vid in remain_points and pid in remain_points[vid]:
                        remain_points[vid].discard(pid)
        self.logger.info(f"Supervoxel segmentation: generated {len(supervoxels)} supervoxel patches.")

        # Step 5: 基于平面片段的区域生长 (Patch merging)
        discontinuities: List[Discontinuity] = []
        with Timer("Supervoxel: Patch-based region growing", self.logger):
            n_patches = len(supervoxels)
            if n_patches == 0:
                return []  # 没有任何片段
            # 计算片段邻接关系: 若两个片段各包含的体素相邻，则视为邻居
            patch_neighbors: Dict[int, Set[int]] = {i: set() for i in range(n_patches)}
            # 建立体素到片段的映射
            voxel_to_patches: Dict[Tuple[int, int, int], List[int]] = {}
            for i, patch in enumerate(supervoxels):
                for vid in patch["voxel_ids"]:
                    voxel_to_patches.setdefault(vid, []).append(i)
            # 确定每个patch的邻居patch集合
            for vid, plist in voxel_to_patches.items():
                # 当前体素的所有patch彼此不是同一片段
                for p in plist:
                    for q in plist:
                        if p != q:
                            patch_neighbors[p].add(q)
                # 跨体素邻域:
                for nbr in self._get_neighbor_voxels(vid):
                    if nbr in voxel_to_patches:
                        for p in plist:
                            for q in voxel_to_patches[nbr]:
                                if p != q:
                                    patch_neighbors[p].add(q)
            # 未合并的patch集合
            unmerged: Set[int] = set(range(n_patches))
            # 按累计误差θ从小到大排序patch列表 (用于确定生长顺序):contentReference[oaicite:41]{index=41}
            patch_order = sorted(unmerged, key=lambda i: supervoxels[i]["error"])
            while patch_order:
                seed_id = patch_order[0]
                patch_order.remove(seed_id)
                if seed_id not in unmerged:
                    continue  # 已被合并则跳过
                unmerged.discard(seed_id)
                # 准备合并该seed的邻居patch
                seed_patch = supervoxels[seed_id]
                neighbor_set = {nid for nid in patch_neighbors[seed_id] if nid in unmerged}
                # 将邻居按误差排序
                neighbor_list = sorted(list(neighbor_set), key=lambda j: supervoxels[j]["error"])
                # 合并邻居patch
                merged_patch_ids = [seed_id]
                while neighbor_list:
                    j = neighbor_list[0]
                    neighbor_list.pop(0)
                    if j not in unmerged:
                        continue
                    # 计算seed与邻居patch平面的距离和角度差
                    # 使用seed_patch的平面参数
                    sn = np.array(seed_patch["normal"])
                    sd = seed_patch["d"]
                    jn = np.array(supervoxels[j]["normal"])
                    # 平面法向角差
                    cos_ang = abs(sn.dot(jn))
                    cos_ang = max(min(cos_ang, 1.0), -1.0)
                    ang_diff = math.degrees(math.acos(cos_ang))
                    # 平面距离差: 邻居patch质心到seed平面的距离
                    j_points = coords[list(supervoxels[j]["points"])]
                    if j_points.shape[0] == 0:
                        dist_diff = float('inf')
                    else:
                        j_centroid = j_points.mean(axis=0)
                        dist_diff = abs(sn.dot(j_centroid) + sd)
                    if ang_diff < self.patch_angle and dist_diff < self.patch_distance:
                        # 满足合并条件:contentReference[oaicite:42]{index=42}, 合并patch j到seed
                        unmerged.discard(j)
                        merged_patch_ids.append(j)
                        # 扩展seed_patch覆盖范围: 更新点集和体素集 (用于后续邻居判断)
                        seed_patch["points"].update(supervoxels[j]["points"])
                        seed_patch["voxel_ids"].update(supervoxels[j]["voxel_ids"])
                        # 更新seed_patch的平面参数 (重新拟合合并后的所有点)
                        pts_all = coords[list(seed_patch["points"])]
                        centroid_all = pts_all.mean(axis=0)
                        cov_all = np.cov(pts_all - centroid_all, rowvar=False)
                        eig_vals_all, eig_vecs_all = np.linalg.eigh(cov_all)
                        normal_all = eig_vecs_all[:, np.argmin(eig_vals_all)]
                        if normal_all[2] < 0:
                            normal_all = -normal_all
                        seed_patch["normal"] = tuple(normal_all.tolist())
                        seed_patch["d"] = -float(normal_all.dot(centroid_all))
                        # 将新邻居加入队列
                        for k in patch_neighbors[j]:
                            if k in unmerged and k not in neighbor_set:
                                neighbor_set.add(k)
                                neighbor_list.append(k)
                        neighbor_list = sorted(neighbor_list, key=lambda j: supervoxels[j]["error"])
                    # 若不合并，则该邻居不再考虑 (continue to next neighbor)
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
                    if plane_normal[2] < 0:
                        plane_normal = -plane_normal
                else:
                    # 点数过少无法拟合平面，用第一个segment的平面
                    plane_normal = np.array(segments_list[0].plane.normal)
                d_val = -float(plane_normal.dot(centroid_all))
                # 计算整体平面的 dip 和 dip_direction
                norm_len = math.sqrt(plane_normal.dot(plane_normal)) + 1e-12
                plane_norm_unit = plane_normal / norm_len
                nx, ny, nz = plane_norm_unit.tolist()
                # 将法向统一到上半球
                if nz < 0:
                    nx, ny, nz = -nx, -ny, -nz
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
                discontinuity = Discontinuity(segments_list,plane_obj, dip, dip_dir, roughness=overall_rmse,
                                              algorithm_name=self.name)
                discontinuities.append(discontinuity)
        self.logger.info(f"Patch-based region growing complete: {len(discontinuities)} planes detected.")
        return discontinuities

    def _get_neighbor_voxels(self, voxel_id: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """
        返回给定体素索引的 26 个邻近体素索引（不包括自身）:contentReference[oaicite:43]{index=43}。
        """
        vx, vy, vz = voxel_id
        neighbors = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    neighbors.append((vx + dx, vy + dy, vz + dz))
        return neighbors


# 测试接口（与 RansacDetector、RegionGrowingDetector 保持一致）
if __name__ == "__main__":
    # 简单功能测试（可在实际数据上替换）
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
    # 可进一步导出结果以使用 ResultsExporter 或可视化
