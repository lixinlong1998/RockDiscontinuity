# RockDiscontinuity/src/rock_discon_extract/algorithms/detector_supervoxel.py

from typing import List, Dict, Tuple, Set, Optional
import math
import os
import time
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
            sample_spacing: Optional[float] = None,
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
        """
        super().__init__(name="Supervoxel")
        self.voxel_size = voxel_size
        # 点云采样间隔 s: 用于在RANSAC内点投影平面上构建bitmap网格(grid size)
        # 该参数必须由外部传入(固定值), 与 voxel_size 类似
        if sample_spacing is None:
            raise ValueError("sample_spacing(点云采样间隔s)必须由外部传入，例如 0.05")
        if sample_spacing <= 0:
            raise ValueError("sample_spacing 必须为正数")
        self.sample_spacing = float(sample_spacing)
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

        self.logger = LoggerManager.GetLogger(self.__class__.__name__)

        # 预计算用于角度判断的cos值（注: 需将角度阈值从度数转换为余弦值比较）
        self._cos_edge_angle = math.cos(math.radians(self.edge_angle))
        self._cos_patch_angle = math.cos(math.radians(self.patch_angle))

        # -------------------------
        # Monitoring counters (for debugging / logging)
        # -------------------------
        self._mon_bitmap_calls = 0
        self._mon_bitmap_inliers = 0
        self._mon_bitmap_connected = 0
        self._mon_bitmap_leftover = 0

    def _bitmapMCCfilter(
            self,
            coords: np.ndarray,
            inliers_global: np.ndarray,
            plane_model: Tuple[float, float, float, float],
            grid_size: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float, float]]:
        """
        功能简介:
            在 pcd.segment_plane 得到 (plane_model, inliers) 之后, 按如下逻辑筛选连通的平面 patch 内点:
                1) 将 inliers 投影到 plane_model 平面上
                2) 以点云采样间隔 s 作为 grid size 构建 bitmap grids
                3) 在 bitmap 上找到最大连通域 C
                4) 对 C 做形态学闭运算(膨胀+腐蚀)以“闭合边界/填补小孔洞”(允许原始点云起伏导致的空洞)
                5) 取闭运算后的 C' 对应的闭合边界区域内的 inliers 作为最终 connected inliers
                6) 用 connected inliers 重新估计 plane_model 作为该 patch 的平面参数
            其余的 inliers 作为 leftover_inliers 返回, 由调用方归入 remain_set 参与后续计算。

        设计动机:
            - 先前使用凸包作为“外包闭合边界”会抹平凹形边界, 更易产生块状/马赛克边界。
            - 改为“最大连通域外轮廓(离散栅格) + 形态学闭运算”可更贴合真实 patch 形状, 且允许内部空洞。

        输入变量及类型:
            coords: np.ndarray, shape=(N,3)
                全局点坐标
            inliers_global: np.ndarray, shape=(K,)
                本次RANSAC inliers 对应的全局索引
            plane_model: Tuple[float,float,float,float]
                本次RANSAC输出平面参数 (a,b,c,d), 满足 ax+by+cz+d=0 (不保证归一化)
            grid_size: Optional[float]
                bitmap 网格大小; 若为 None 则使用 self.sample_spacing

        输出变量及类型:
            connected_inliers_global: np.ndarray, shape=(Kc,)
                闭合边界区域内的 inliers(全局索引)
            leftover_inliers_global: np.ndarray, shape=(Kl,)
                闭合边界区域外的原 inliers(全局索引), 应回收进 remain
            refined_plane_model: Tuple[float,float,float,float]
                用 connected_inliers 重新估计得到的平面参数 (a,b,c,d), 单位法向
        """
        with Timer("Supervoxel: _bitmapMCCfilter", self.logger):
            t0 = time.perf_counter()

            # -------------------------
            # 0. 参数与输入检查
            # -------------------------
            if grid_size is None:
                grid_size = self.sample_spacing
            if grid_size <= 0:
                raise ValueError(f"grid_size 必须为正数, 当前={grid_size}")

            if inliers_global is None:
                return np.array([], dtype=int), np.array([], dtype=int), plane_model
            inliers_global = np.asarray(inliers_global, dtype=int)
            if inliers_global.size == 0:
                return np.array([], dtype=int), np.array([], dtype=int), plane_model

            pts = coords[inliers_global]
            if pts.ndim != 2 or pts.shape[1] != 3:
                raise ValueError("coords 必须为 shape=(N,3) 的数组")

            # -------------------------
            # 1. 平面归一化 + 点投影
            # -------------------------
            a, b, c, d = plane_model
            n = np.array([a, b, c], dtype=float)
            n_norm = float(np.linalg.norm(n) + 1e-12)
            n = n / n_norm
            d = float(d / n_norm)

            signed_dist = np.dot(pts, n) + d
            pts_proj = pts - signed_dist[:, None] * n[None, :]

            # -------------------------
            # 2. 构建平面局部2D坐标系(u,v), 计算(U,V)
            # -------------------------
            if abs(n[2]) < 0.9:
                ref = np.array([0.0, 0.0, 1.0], dtype=float)
            else:
                ref = np.array([0.0, 1.0, 0.0], dtype=float)

            u = np.cross(ref, n)
            u = u / (np.linalg.norm(u) + 1e-12)
            v = np.cross(n, u)

            origin = pts_proj.mean(axis=0)
            rel = pts_proj - origin[None, :]
            uv = np.empty((pts_proj.shape[0], 2), dtype=float)
            uv[:, 0] = np.dot(rel, u)
            uv[:, 1] = np.dot(rel, v)

            # -------------------------
            # 3. bitmap 栅格化(稠密网格)
            # -------------------------
            u_min = float(uv[:, 0].min())
            v_min = float(uv[:, 1].min())

            ix = np.floor((uv[:, 0] - u_min) / grid_size).astype(np.int32)
            iy = np.floor((uv[:, 1] - v_min) / grid_size).astype(np.int32)

            nx = int(ix.max()) + 1
            ny = int(iy.max()) + 1

            # 防止极端情况下bitmap过大导致内存问题
            MAX_CELLS = 4_000_000  # 约 2000x2000
            if nx * ny > MAX_CELLS:
                # 回退: 直接返回原始inliers(避免崩溃). 该情况极少见, 记录告警方便排查。
                refined = (float(n[0]), float(n[1]), float(n[2]), float(d))
                self.logger.warning(
                    f"_bitmapMCCfilter: bitmap过大({nx}x{ny}={nx * ny}), fallback to raw inliers."
                )
                return inliers_global.copy(), np.array([], dtype=int), refined

            bitmap = np.zeros((nx, ny), dtype=bool)
            bitmap[ix, iy] = True

            # -------------------------
            # 4. 最大连通域(MCC)提取: 8邻域
            # -------------------------
            visited = np.zeros_like(bitmap, dtype=bool)
            best_cells = []
            best_size = 0

            # 只遍历占据单元
            occ_cells = np.argwhere(bitmap)
            # neighbor offsets (8)
            nbr8 = [(-1, -1), (-1, 0), (-1, 1),
                    (0, -1), (0, 1),
                    (1, -1), (1, 0), (1, 1)]

            for sx, sy in occ_cells:
                if visited[sx, sy]:
                    continue
                stack = [(int(sx), int(sy))]
                visited[sx, sy] = True
                comp = [(int(sx), int(sy))]

                while stack:
                    cx, cy = stack.pop()
                    for dx, dy in nbr8:
                        nx2, ny2 = cx + dx, cy + dy
                        if nx2 < 0 or nx2 >= nx or ny2 < 0 or ny2 >= ny:
                            continue
                        if bitmap[nx2, ny2] and (not visited[nx2, ny2]):
                            visited[nx2, ny2] = True
                            stack.append((nx2, ny2))
                            comp.append((nx2, ny2))

                if len(comp) > best_size:
                    best_size = len(comp)
                    best_cells = comp

            if best_size < 3:
                refined = (float(n[0]), float(n[1]), float(n[2]), float(d))
                return inliers_global.copy(), np.array([], dtype=int), refined

            comp_mask = np.zeros_like(bitmap, dtype=bool)
            comp_mask[tuple(np.array(best_cells).T)] = True

            # -------------------------
            # 5. 形态学闭运算: dilation -> erosion (3x3结构元)
            #    目的: 闭合边界、填补小孔洞, 但不依赖法向滤波
            # -------------------------
            def _DilateBool(mask: np.ndarray) -> np.ndarray:
                # 3x3 max over neighborhood
                pad = np.pad(mask, ((1, 1), (1, 1)), mode="constant", constant_values=False)
                out = np.zeros_like(mask, dtype=bool)
                # 9 shifts OR
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        out |= pad[1 + dx:1 + dx + mask.shape[0], 1 + dy:1 + dy + mask.shape[1]]
                return out

            def _ErodeBool(mask: np.ndarray) -> np.ndarray:
                # 3x3 min over neighborhood
                pad = np.pad(mask, ((1, 1), (1, 1)), mode="constant", constant_values=False)
                out = np.ones_like(mask, dtype=bool)
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        out &= pad[1 + dx:1 + dx + mask.shape[0], 1 + dy:1 + dy + mask.shape[1]]
                return out

            closed_mask = _ErodeBool(_DilateBool(comp_mask))

            # -------------------------
            # 6. 根据闭合后的区域掩膜选择 connected inliers
            # -------------------------
            inside_mask = closed_mask[ix, iy]
            connected_inliers_global = inliers_global[inside_mask]
            leftover_inliers_global = inliers_global[~inside_mask]

            if connected_inliers_global.size < 3:
                refined = (float(n[0]), float(n[1]), float(n[2]), float(d))
                return inliers_global.copy(), np.array([], dtype=int), refined

            # -------------------------
            # 7. 用 connected_inliers 重估平面参数 (PCA最小二乘)
            # -------------------------
            pts_conn = coords[connected_inliers_global]
            centroid = pts_conn.mean(axis=0)
            X = pts_conn - centroid[None, :]
            cov = (X.T @ X) / float(max(pts_conn.shape[0], 1))

            eigvals, eigvecs = np.linalg.eigh(cov)
            normal_refit = eigvecs[:, 0]
            normal_refit = normal_refit / (np.linalg.norm(normal_refit) + 1e-12)

            if float(np.dot(normal_refit, n)) < 0:
                normal_refit = -normal_refit

            d_refit = -float(np.dot(normal_refit, centroid))
            refined_plane_model = (
            float(normal_refit[0]), float(normal_refit[1]), float(normal_refit[2]), float(d_refit))

            # -------------------------
            # 8. log / 耗时 / 可视化(可选)
            # -------------------------
            t1 = time.perf_counter()
            self.logger.debug(
                f"_bitmapMCCfilter: inliers={inliers_global.size}, grid=({nx},{ny}), "
                f"MCC={best_size}, closed_added={int(closed_mask.sum() - comp_mask.sum())}, "
                f"connected={connected_inliers_global.size}, leftover={leftover_inliers_global.size}, "
                f"grid_size={grid_size} | 耗时 {(t1 - t0):.4f}s"
            )

            debug_dir = os.environ.get("BITMAP_MCC_DEBUG_DIR", "").strip()
            if debug_dir:
                try:
                    os.makedirs(debug_dir, exist_ok=True)
                    tag = f"{int(time.time() * 1000)}"
                    np.savez_compressed(
                        os.path.join(debug_dir, f"bitmap_mcc_{tag}.npz"),
                        uv=uv,
                        ix=ix,
                        iy=iy,
                        bitmap=bitmap.astype(np.uint8),
                        comp_mask=comp_mask.astype(np.uint8),
                        closed_mask=closed_mask.astype(np.uint8),
                        inliers_global=inliers_global,
                        connected_inliers_global=connected_inliers_global,
                        leftover_inliers_global=leftover_inliers_global,
                        origin=origin,
                        u=u,
                        v=v,
                        plane_model=np.asarray([n[0], n[1], n[2], d], dtype=float),
                        refined_plane_model=np.asarray(refined_plane_model, dtype=float),
                        grid_size=np.asarray([grid_size], dtype=float)
                    )

                    if o3d is not None:
                        pcd_proj = o3d.geometry.PointCloud()
                        pcd_proj.points = o3d.utility.Vector3dVector(pts_proj)
                        o3d.io.write_point_cloud(os.path.join(debug_dir, f"bitmap_mcc_{tag}_proj.ply"), pcd_proj)

                        pcd_conn = o3d.geometry.PointCloud()
                        pcd_conn.points = o3d.utility.Vector3dVector(pts_conn)
                        o3d.io.write_point_cloud(os.path.join(debug_dir, f"bitmap_mcc_{tag}_connected.ply"), pcd_conn)

                except Exception as e:
                    self.logger.warning(f"_bitmapMCCfilter debug export failed: {e}")

            return connected_inliers_global, leftover_inliers_global, refined_plane_model

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

            # -------------------------
            # Monitoring: basic stats
            # -------------------------
            valid_normals = int(np.count_nonzero(normals_valid))
            valid_ratio = float(valid_normals) / float(max(num_points, 1))
            self.logger.info(f"[MON] Points={num_points}, ValidNormals={valid_normals} ({valid_ratio:.2%})")
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

            # Monitoring: voxel occupancy stats
            voxel_counts = np.array([len(v) for v in voxel_map.values()], dtype=int) if len(
                voxel_map) > 0 else np.array([], dtype=int)
            if voxel_counts.size > 0:
                self.logger.info(
                    f"[MON] VoxelOcc: mean={voxel_counts.mean():.1f}, "
                    f"p50={np.percentile(voxel_counts, 50):.0f}, p90={np.percentile(voxel_counts, 90):.0f}, "
                    f"min={voxel_counts.min()}, max={voxel_counts.max()}"
                )

            # -------------------------
            # Step 2: 体素内局部 RANSAC 提取 voxel-patch
            # -------------------------
            clusters: List[Dict] = []  # 每个 cluster 是一个初始平面片 (voxel-patch 或 edge-patch)
            remain_points: Dict[Tuple[int, int, int], Set[int]] = {}
            with Timer("Supervoxel: Initial voxel-plane extraction", self.logger):
                for voxel_id, pts_indices_list in voxel_map.items():
                    pts_indices = np.array(pts_indices_list, dtype=int)
                    if pts_indices.size < self.min_plane_points:
                        # 点数不足, 整个体素视为 non-coplanar, 所有点进入 remain_points
                        remain_points[voxel_id] = set(pts_indices.tolist())
                        continue

                    # 在该体素内执行 RANSAC 拟合平面
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(coords[pts_indices])
                    plane_model, inliers = pcd.segment_plane(
                        distance_threshold=self.ransac_distance,
                        ransac_n=3,
                        num_iterations=1000
                    )
                    inliers = np.array(inliers, dtype=int)
                    if inliers.size == 0:
                        # 未找到平面, 所有点进入 remain_points
                        remain_points[voxel_id] = set(pts_indices.tolist())
                        continue

                    # RANSAC 平面参数归一化
                    a, b, c, d = plane_model
                    norm_len = math.sqrt(a * a + b * b + c * c) + 1e-12
                    a, b, c, d = a / norm_len, b / norm_len, c / norm_len, d / norm_len
                    plane_normal = np.array([a, b, c], dtype=float)

                    # 将局部 inlier 映射回全局索引
                    voxel_inliers_global = pts_indices[inliers]

                    # 对 inlier 做 bitmap 最大连通域 + 外包闭合边界筛选(允许空洞), 并用边界内点重估 plane_model
                    connected_inliers_global, leftover_inliers_global, refined_plane_model = self._bitmapMCCfilter(
                        coords=coords,
                        inliers_global=voxel_inliers_global,
                        plane_model=(a, b, c, d),
                        grid_size=self.sample_spacing
                    )
                    filtered_inliers_global = connected_inliers_global
                    a, b, c, d = refined_plane_model
                    plane_normal = np.array([a, b, c], dtype=float)

                    inlier_set = set(filtered_inliers_global.tolist())
                    remain_set = set(pts_indices.tolist()) - inlier_set

                    if len(inlier_set) < self.min_plane_points:
                        # 平面内点太少, 整体视为剩余点
                        remain_points[voxel_id] = set(pts_indices.tolist())
                    else:
                        clusters.append({
                            "voxel_ids": [voxel_id],
                            "points": set(filtered_inliers_global.tolist()),
                            "orig_points": set(filtered_inliers_global.tolist()),
                            "normal": (float(a), float(b), float(c)),
                            "d": float(d)
                        })
                        remain_points[voxel_id] = remain_set

                for vid in voxel_map.keys():
                    remain_points.setdefault(vid, set())

            self.logger.info(
                f"Initial voxel-plane extraction: {len(clusters)} plane seeds, "
                f"{sum(len(s) for s in remain_points.values())} remain points."
            )

            # Monitoring: seed cluster size & remain stats
            if len(clusters) > 0:
                seed_sizes = np.array([len(c.get("points", [])) for c in clusters], dtype=int)
                self.logger.info(
                    f"[MON] SeedClusters: n={len(clusters)}, meanPts={seed_sizes.mean():.1f}, "
                    f"p50={np.percentile(seed_sizes, 50):.0f}, p90={np.percentile(seed_sizes, 90):.0f}, "
                    f"minPts={seed_sizes.min()}, maxPts={seed_sizes.max()}"
                )
            remain_total = int(sum(len(s) for s in remain_points.values()))
            remain_vox = int(sum(1 for s in remain_points.values() if len(s) > 0))
            self.logger.info(f"[MON] RemainAfterStep2: points={remain_total}, voxels={remain_vox}")

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
                        pts_indices = np.array(list(pts_set), dtype=int)
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(coords[pts_indices])
                        plane_model, inliers = pcd.segment_plane(
                            distance_threshold=self.ransac_distance,
                            ransac_n=3,
                            num_iterations=500
                        )
                        inliers = np.array(inliers, dtype=int)
                        if inliers.size == 0:
                            break

                        edge_inliers_global = pts_indices[inliers]

                        # 对 edge-patch 做 bitmap 最大连通域 + 外包闭合边界筛选(允许空洞), 并用边界内点重估 plane_model
                        pa, pb, pc, pd = plane_model
                        norm_len = math.sqrt(pa * pa + pb * pb + pc * pc) + 1e-12
                        pa, pb, pc, pd = pa / norm_len, pb / norm_len, pc / norm_len, pd / norm_len
                        patch_normal = np.array([pa, pb, pc], dtype=float)

                        connected_edge_inliers_global, leftover_edge_inliers_global, refined_plane_model = self._bitmapMCCfilter(
                            coords=coords,
                            inliers_global=edge_inliers_global,
                            plane_model=(pa, pb, pc, pd),
                            grid_size=self.sample_spacing
                        )
                        filtered_edge_inliers_global = connected_edge_inliers_global
                        pa, pb, pc, pd = refined_plane_model
                        patch_normal = np.array([pa, pb, pc], dtype=float)

                        inlier_set = set(filtered_edge_inliers_global.tolist())
                        edge_remain_set = pts_set - inlier_set

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
                                    "normal": (float(pa), float(pb), float(pc)),
                                    "d": float(pd)
                                })

                            # 更新该体素剩余点
                            pts_set = edge_remain_set
                        else:
                            # patch 点太少, 认为无法成面, 停止对该 voxel 的边缘提取
                            testable = False

                    remain_points[voxel_id] = pts_set

                clusters.extend(new_clusters)

            self.logger.info(f"Edge patch extraction: clusters extended to {len(clusters)} seeds.")

            # Monitoring: remain stats after edge extraction
            remain_total = int(sum(len(s) for s in remain_points.values()))
            remain_vox = int(sum(1 for s in remain_points.values() if len(s) > 0))
            self.logger.info(f"[MON] RemainAfterStep3: points={remain_total}, voxels={remain_vox}")

            # -------------------------
            # Step 4: 超体素分割 (Supervoxel segmentation)
            # -------------------------
            supervoxels: List[Dict] = []

            # Monitoring: supervoxel growth summary (aggregate)
            mon_sv_seeds = len(clusters)
            mon_sv_iters_total = 0
            mon_sv_absorb_total = 0
            mon_sv_best_pts_total = 0
            mon_sv_success = 0
            mon_verbose = os.environ.get("SUPERVOXEL_MON_VERBOSE", "0").strip() in ("1", "true", "True")

            with Timer("Supervoxel: Supervoxel segmentation", self.logger):
                for ci, cluster in enumerate(clusters):
                    record_normal = np.array(cluster["normal"], dtype=float)
                    record_d = float(cluster["d"])

                    # Monitoring: per-seed stats
                    mon_seed_iters = 0
                    mon_seed_absorbed = 0
                    mon_seed_cand = 0

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

                            mon_seed_iters += 1
                            current_points: Set[int] = set(cluster["orig_points"])
                            current_normal = record_normal.copy()
                            current_d = record_d
                            cos_ang_th = math.cos(math.radians(ang_th))

                            # 从候选区域中吸收符合当前阈值的 remain 点
                            for nv, nv_indices in neighbor_remains.items():
                                if nv_indices.size == 0:
                                    continue

                                mon_seed_cand += int(nv_indices.size)

                                dists = np.abs(np.dot(coords[nv_indices], current_normal) + current_d)

                                valid_mask = normals_valid[nv_indices]
                                if not np.any(valid_mask):
                                    continue
                                cosines = np.abs(np.dot(normals[nv_indices], current_normal))

                                mask = (dists < dist_th) & (cosines >= cos_ang_th) & valid_mask
                                if np.any(mask):
                                    selected_indices = nv_indices[mask]
                                    current_points.update(selected_indices.tolist())

                                    mon_seed_absorbed += int(selected_indices.size)

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

                    # Monitoring: aggregate update
                    mon_sv_iters_total += int(mon_seed_iters)
                    mon_sv_absorb_total += int(mon_seed_absorbed)
                    mon_sv_best_pts_total += int(len(best_points_set))
                    if orientation_diff <= self.max_refit_error:
                        mon_sv_success += 1
                    if mon_verbose:
                        self.logger.info(
                            f"[MON] SeedGrow ci={ci}: iters={mon_seed_iters}, candPts={mon_seed_cand}, "
                            f"absorbedPts={mon_seed_absorbed}, bestPts={len(best_points_set)}, "
                            f"finalDistTh={dist_th:.3f}, finalAngTh={ang_th:.1f}, orientDiff={orientation_diff:.2f}"
                        )

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

            # Monitoring: supervoxel summary & remain stats
            if len(supervoxels) > 0:
                sv_sizes = np.array([len(sv.get("points", [])) for sv in supervoxels], dtype=int)
                self.logger.info(
                    f"[MON] Supervoxels: n={len(supervoxels)}, meanPts={sv_sizes.mean():.1f}, "
                    f"p50={np.percentile(sv_sizes, 50):.0f}, p90={np.percentile(sv_sizes, 90):.0f}, "
                    f"minPts={sv_sizes.min()}, maxPts={sv_sizes.max()}"
                )
            remain_total = int(sum(len(s) for s in remain_points.values()))
            remain_vox = int(sum(1 for s in remain_points.values() if len(s) > 0))
            self.logger.info(f"[MON] RemainAfterStep4: points={remain_total}, voxels={remain_vox}")
            self.logger.info(
                f"[MON] SuperGrowAgg: seeds={mon_sv_seeds}, success={mon_sv_success}, "
                f"itersTotal={mon_sv_iters_total}, absorbedTotal={mon_sv_absorb_total}, "
                f"bestPtsTotal={mon_sv_best_pts_total}"
            )
            # bitmap filter aggregated stats
            try:
                if self._mon_bitmap_calls > 0:
                    conn_ratio = (float(self._mon_bitmap_connected) / float(max(self._mon_bitmap_inliers, 1)))
                    self.logger.info(
                        f"[MON] BitmapMCC: calls={self._mon_bitmap_calls}, "
                        f"inliers={self._mon_bitmap_inliers}, connected={self._mon_bitmap_connected}, "
                        f"leftover={self._mon_bitmap_leftover}, connRatio={conn_ratio:.2%}"
                    )
            except Exception:
                pass

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
                    plane_obj = Plane((nx, ny, nz), d_val, tuple(centroid_all.tolist()),
                                      inlier_indices=list(all_points),
                                      rmse=overall_rmse)
                    # 构造 Discontinuity 对象
                    discontinuity = Discontinuity(segments_list, plane_obj, dip, dip_dir, roughness=overall_rmse,
                                                  algorithm_name=self.name)
                    discontinuities.append(discontinuity)

            self.logger.info(f"Patch-based region growing complete: {len(discontinuities)} planes detected.")

            # Monitoring: plane size stats & final unassigned points
            if len(discontinuities) > 0:
                plane_sizes = np.array(
                    [len(d.segments[0].points) if (d.segments and hasattr(d.segments[0], "points")) else 0 for d in
                     discontinuities], dtype=int)
                if plane_sizes.size > 0:
                    self.logger.info(
                        f"[MON] PlaneSizes: meanPts={plane_sizes.mean():.1f}, p50={np.percentile(plane_sizes, 50):.0f}, "
                        f"p90={np.percentile(plane_sizes, 90):.0f}, minPts={plane_sizes.min()}, maxPts={plane_sizes.max()}"
                    )
            remain_total = int(sum(len(s) for s in remain_points.values()))
            remain_vox = int(sum(1 for s in remain_points.values() if len(s) > 0))
            self.logger.info(f"[MON] UnassignedRemainEnd: points={remain_total}, voxels={remain_vox}")
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

        # -------------------------
        # Monitoring: basic stats
        # -------------------------
        valid_normals = int(np.count_nonzero(normals_valid))
        valid_ratio = float(valid_normals) / float(max(num_points, 1))
        self.logger.info(f"[MON] Points={num_points}, ValidNormals={valid_normals} ({valid_ratio:.2%})")
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

        # Monitoring: voxel occupancy stats
        voxel_counts = np.array([len(v) for v in voxel_map.values()], dtype=int) if len(voxel_map) > 0 else np.array([], dtype=int)
        if voxel_counts.size > 0:
            self.logger.info(
                f"[MON] VoxelOcc: mean={voxel_counts.mean():.1f}, "
                f"p50={np.percentile(voxel_counts,50):.0f}, p90={np.percentile(voxel_counts,90):.0f}, "
                f"min={voxel_counts.min()}, max={voxel_counts.max()}"
            )

        # -------------------------
        # Step 2: 体素内局部 RANSAC 提取 voxel-patch
        # -------------------------
        clusters: List[Dict] = []  # 每个 cluster 是一个初始平面片 (voxel-patch 或 edge-patch)
        remain_points: Dict[Tuple[int, int, int], Set[int]] = {}
        with Timer("Supervoxel: Initial voxel-plane extraction", self.logger):
            for voxel_id, pts_indices_list in voxel_map.items():
                pts_indices = np.array(pts_indices_list, dtype=int)
                if pts_indices.size < self.min_plane_points:
                    # 点数不足, 整个体素视为 non-coplanar, 所有点进入 remain_points
                    remain_points[voxel_id] = set(pts_indices.tolist())
                    continue

                # 在该体素内执行 RANSAC 拟合平面
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(coords[pts_indices])
                plane_model, inliers = pcd.segment_plane(
                    distance_threshold=self.ransac_distance,
                    ransac_n=3,
                    num_iterations=1000
                )
                inliers = np.array(inliers, dtype=int)
                if inliers.size == 0:
                    # 未找到平面, 所有点进入 remain_points
                    remain_points[voxel_id] = set(pts_indices.tolist())
                    continue

                # RANSAC 平面参数归一化
                a, b, c, d = plane_model
                norm_len = math.sqrt(a * a + b * b + c * c) + 1e-12
                a, b, c, d = a / norm_len, b / norm_len, c / norm_len, d / norm_len
                plane_normal = np.array([a, b, c], dtype=float)

                # 将局部 inlier 映射回全局索引
                voxel_inliers_global = pts_indices[inliers]

                # 对 inlier 做 bitmap 最大连通域 + 外包闭合边界筛选(允许空洞), 并用边界内点重估 plane_model
                connected_inliers_global, leftover_inliers_global, refined_plane_model = self._bitmapMCCfilter(
                    coords=coords,
                    inliers_global=voxel_inliers_global,
                    plane_model=(a, b, c, d),
                    grid_size=self.sample_spacing
                )
                filtered_inliers_global = connected_inliers_global
                a, b, c, d = refined_plane_model
                plane_normal = np.array([a, b, c], dtype=float)

                inlier_set = set(filtered_inliers_global.tolist())
                remain_set = set(pts_indices.tolist()) - inlier_set

                if len(inlier_set) < self.min_plane_points:
                    # 平面内点太少, 整体视为剩余点
                    remain_points[voxel_id] = set(pts_indices.tolist())
                else:
                    clusters.append({
                        "voxel_ids": [voxel_id],
                        "points": set(filtered_inliers_global.tolist()),
                        "orig_points": set(filtered_inliers_global.tolist()),
                        "normal": (float(a), float(b), float(c)),
                        "d": float(d)
                    })
                    remain_points[voxel_id] = remain_set

            for vid in voxel_map.keys():
                remain_points.setdefault(vid, set())

        self.logger.info(
            f"Initial voxel-plane extraction: {len(clusters)} plane seeds, "
            f"{sum(len(s) for s in remain_points.values())} remain points."
        )

        # Monitoring: seed cluster size & remain stats
        if len(clusters) > 0:
            seed_sizes = np.array([len(c.get("points", [])) for c in clusters], dtype=int)
            self.logger.info(
                f"[MON] SeedClusters: n={len(clusters)}, meanPts={seed_sizes.mean():.1f}, "
                f"p50={np.percentile(seed_sizes,50):.0f}, p90={np.percentile(seed_sizes,90):.0f}, "
                f"minPts={seed_sizes.min()}, maxPts={seed_sizes.max()}"
            )
        remain_total = int(sum(len(s) for s in remain_points.values()))
        remain_vox = int(sum(1 for s in remain_points.values() if len(s) > 0))
        self.logger.info(f"[MON] RemainAfterStep2: points={remain_total}, voxels={remain_vox}")

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
                    pts_indices = np.array(list(pts_set), dtype=int)
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(coords[pts_indices])
                    plane_model, inliers = pcd.segment_plane(
                        distance_threshold=self.ransac_distance,
                        ransac_n=3,
                        num_iterations=500
                    )
                    inliers = np.array(inliers, dtype=int)
                    if inliers.size == 0:
                        break

                    edge_inliers_global = pts_indices[inliers]

                    # 对 edge-patch 做 bitmap 最大连通域 + 外包闭合边界筛选(允许空洞), 并用边界内点重估 plane_model
                    pa, pb, pc, pd = plane_model
                    norm_len = math.sqrt(pa * pa + pb * pb + pc * pc) + 1e-12
                    pa, pb, pc, pd = pa / norm_len, pb / norm_len, pc / norm_len, pd / norm_len
                    patch_normal = np.array([pa, pb, pc], dtype=float)

                    connected_edge_inliers_global, leftover_edge_inliers_global, refined_plane_model = self._bitmapMCCfilter(
                        coords=coords,
                        inliers_global=edge_inliers_global,
                        plane_model=(pa, pb, pc, pd),
                        grid_size=self.sample_spacing
                    )
                    filtered_edge_inliers_global = connected_edge_inliers_global
                    pa, pb, pc, pd = refined_plane_model
                    patch_normal = np.array([pa, pb, pc], dtype=float)

                    inlier_set = set(filtered_edge_inliers_global.tolist())
                    edge_remain_set = pts_set - inlier_set

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
                                "normal": (float(pa), float(pb), float(pc)),
                                "d": float(pd)
                            })

                        # 更新该体素剩余点
                        pts_set = edge_remain_set
                    else:
                        # patch 点太少, 认为无法成面, 停止对该 voxel 的边缘提取
                        testable = False

                remain_points[voxel_id] = pts_set

            clusters.extend(new_clusters)

        self.logger.info(f"Edge patch extraction: clusters extended to {len(clusters)} seeds.")

        # Monitoring: remain stats after edge extraction
        remain_total = int(sum(len(s) for s in remain_points.values()))
        remain_vox = int(sum(1 for s in remain_points.values() if len(s) > 0))
        self.logger.info(f"[MON] RemainAfterStep3: points={remain_total}, voxels={remain_vox}")

        # -------------------------
        # Step 4: 超体素分割 (Supervoxel segmentation)
        # -------------------------
        supervoxels: List[Dict] = []

        # Monitoring: supervoxel growth summary (aggregate)
        mon_sv_seeds = len(clusters)
        mon_sv_iters_total = 0
        mon_sv_absorb_total = 0
        mon_sv_best_pts_total = 0
        mon_sv_success = 0
        mon_verbose = os.environ.get("SUPERVOXEL_MON_VERBOSE", "0").strip() in ("1", "true", "True")

        with Timer("Supervoxel: Supervoxel segmentation", self.logger):
            for ci, cluster in enumerate(clusters):
                record_normal = np.array(cluster["normal"], dtype=float)
                record_d = float(cluster["d"])

                # Monitoring: per-seed stats
                mon_seed_iters = 0
                mon_seed_absorbed = 0
                mon_seed_cand = 0

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

                        mon_seed_iters += 1
                        current_points: Set[int] = set(cluster["orig_points"])
                        current_normal = record_normal.copy()
                        current_d = record_d
                        cos_ang_th = math.cos(math.radians(ang_th))

                        # 从候选区域中吸收符合当前阈值的 remain 点
                        for nv, nv_indices in neighbor_remains.items():
                            if nv_indices.size == 0:
                                continue

                            mon_seed_cand += int(nv_indices.size)

                            dists = np.abs(np.dot(coords[nv_indices], current_normal) + current_d)

                            valid_mask = normals_valid[nv_indices]
                            if not np.any(valid_mask):
                                continue
                            cosines = np.abs(np.dot(normals[nv_indices], current_normal))

                            mask = (dists < dist_th) & (cosines >= cos_ang_th) & valid_mask
                            if np.any(mask):
                                selected_indices = nv_indices[mask]
                                current_points.update(selected_indices.tolist())

                                mon_seed_absorbed += int(selected_indices.size)

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

                # Monitoring: aggregate update
                mon_sv_iters_total += int(mon_seed_iters)
                mon_sv_absorb_total += int(mon_seed_absorbed)
                mon_sv_best_pts_total += int(len(best_points_set))
                if orientation_diff <= self.max_refit_error:
                    mon_sv_success += 1
                if mon_verbose:
                    self.logger.info(
                        f"[MON] SeedGrow ci={ci}: iters={mon_seed_iters}, candPts={mon_seed_cand}, "
                        f"absorbedPts={mon_seed_absorbed}, bestPts={len(best_points_set)}, "
                        f"finalDistTh={dist_th:.3f}, finalAngTh={ang_th:.1f}, orientDiff={orientation_diff:.2f}"
                    )

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

        # Monitoring: supervoxel summary & remain stats
        if len(supervoxels) > 0:
            sv_sizes = np.array([len(sv.get("points", [])) for sv in supervoxels], dtype=int)
            self.logger.info(
                f"[MON] Supervoxels: n={len(supervoxels)}, meanPts={sv_sizes.mean():.1f}, "
                f"p50={np.percentile(sv_sizes,50):.0f}, p90={np.percentile(sv_sizes,90):.0f}, "
                f"minPts={sv_sizes.min()}, maxPts={sv_sizes.max()}"
            )
        remain_total = int(sum(len(s) for s in remain_points.values()))
        remain_vox = int(sum(1 for s in remain_points.values() if len(s) > 0))
        self.logger.info(f"[MON] RemainAfterStep4: points={remain_total}, voxels={remain_vox}")
        self.logger.info(
            f"[MON] SuperGrowAgg: seeds={mon_sv_seeds}, success={mon_sv_success}, "
            f"itersTotal={mon_sv_iters_total}, absorbedTotal={mon_sv_absorb_total}, "
            f"bestPtsTotal={mon_sv_best_pts_total}"
        )
        # bitmap filter aggregated stats
        try:
            if self._mon_bitmap_calls > 0:
                conn_ratio = (float(self._mon_bitmap_connected) / float(max(self._mon_bitmap_inliers, 1)))
                self.logger.info(
                    f"[MON] BitmapMCC: calls={self._mon_bitmap_calls}, "
                    f"inliers={self._mon_bitmap_inliers}, connected={self._mon_bitmap_connected}, "
                    f"leftover={self._mon_bitmap_leftover}, connRatio={conn_ratio:.2%}"
                )
        except Exception:
            pass

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

        # Monitoring: plane size stats & final unassigned points
        if len(discontinuities) > 0:
            plane_sizes = np.array([len(d.segments[0].points) if (d.segments and hasattr(d.segments[0], "points")) else 0 for d in discontinuities], dtype=int)
            if plane_sizes.size > 0:
                self.logger.info(
                    f"[MON] PlaneSizes: meanPts={plane_sizes.mean():.1f}, p50={np.percentile(plane_sizes,50):.0f}, "
                    f"p90={np.percentile(plane_sizes,90):.0f}, minPts={plane_sizes.min()}, maxPts={plane_sizes.max()}"
                )
        remain_total = int(sum(len(s) for s in remain_points.values()))
        remain_vox = int(sum(1 for s in remain_points.values() if len(s) > 0))
        self.logger.info(f"[MON] UnassignedRemainEnd: points={remain_total}, voxels={remain_vox}")
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
