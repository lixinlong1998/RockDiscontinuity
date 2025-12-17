from typing import List, Dict, Optional, Tuple
import math
import numpy as np
from scipy.spatial import ConvexHull
from .logging_utils import LoggerManager


class Point:
    """
    功能简介:
        表示点云中的单个点, 包含坐标、颜色、强度、法向等属性.
    """

    def __init__(
            self,
            x: float,
            y: float,
            z: float,
            r: float = 0.0,
            g: float = 0.0,
            b: float = 0.0,
            intensity: float = 0.0,
            normal: Optional[Tuple[float, float, float]] = None,
            curvature: float = 0.0,
            features: Optional[Dict[str, float]] = None,
            point_id: Optional[int] = None
    ):
        self.x = x
        self.y = y
        self.z = z
        self.r = r
        self.g = g
        self.b = b
        self.intensity = intensity
        self.normal = normal
        self.curvature = curvature
        self.features = features if features is not None else {}
        self.point_id = point_id


class BoundingBox:
    """
    功能简介:
        表示点云或子区域的轴对齐包围盒(AABB).
    """

    def __init__(
            self,
            min_x: float,
            min_y: float,
            min_z: float,
            max_x: float,
            max_y: float,
            max_z: float
    ):
        self.min_x = min_x
        self.min_y = min_y
        self.min_z = min_z
        self.max_x = max_x
        self.max_y = max_y
        self.max_z = max_z

    def Contains(self, point: Point) -> bool:
        return (self.min_x <= point.x <= self.max_x and
                self.min_y <= point.y <= self.max_y and
                self.min_z <= point.z <= self.max_z)


class Plane:
    """
    功能简介:
        表示三维空间中的一个平面及其拟合统计信息.
    """

    def __init__(
            self,
            normal: Tuple[float, float, float],
            d: float,
            centroid: Tuple[float, float, float],
            inlier_indices: Optional[List[int]] = None,
            rmse: float = 0.0
    ):
        self.normal = normal
        self.d = d
        self.centroid = centroid
        self.inlier_indices = inlier_indices if inlier_indices is not None else []
        self.rmse = rmse

    def DistanceToPoint(self, point: Point) -> float:
        nx, ny, nz = self.normal
        return (nx * point.x + ny * point.y + nz * point.z + self.d)

    def AngleWithPlane(self, other: "Plane") -> float:
        nx1, ny1, nz1 = self.normal
        nx2, ny2, nz2 = other.normal
        dot = nx1 * nx2 + ny1 * ny2 + nz1 * nz2
        mag1 = math.sqrt(nx1 * nx1 + ny1 * ny1 + nz1 * nz1)
        mag2 = math.sqrt(nx2 * nx2 + ny2 * ny2 + nz2 * nz2)
        cos_theta = max(min(dot / (mag1 * mag2 + 1e-12), 1.0), -1.0)
        return math.degrees(math.acos(abs(cos_theta)))


class Segment:
    """
    功能简介:
        表示某个结构面在点云上的单个连通片.
    参数说明:
        plane:
            对应的平面对象(Plane), 内含法向量、截距、质心等信息.
        point_indices: List[int]
            该连通片上所有点在全局点云 PointCloud.points 中的索引.
        trace_length: float
            该连通片在空间中的迹长(可由后续几何分析得到), 默认 0.0.
    """

    def __init__(
            self,
            plane: Plane,
            point_indices: List[int],
            trace_length: float = 0.0
    ):
        self.plane = plane
        self.point_indices = point_indices
        self.trace_length = trace_length


class Discontinuity:
    """
    功能简介:
        表示一个岩体结构面(Discontinuity)，由若干个 Segment 组成，并在后处理阶段
        通过 ComputeGeometry 计算其几何属性(凸包、多边形面积、迹长、宽度等)。

    主要属性(与本次修改相关):
        segments: List[Segment]
            该结构面包含的所有连通片，每个 Segment 中通常包含:
                - plane: Plane, 对该连通片拟合的平面
                - point_indices: List[int], 所在点云中的点索引
        point_indices: List[int]
            该结构面包含的所有点在全局点云中的索引(可包含重复, 由外部填充)。

        # 凸包/多边形相关:
        polygon_point_indices: List[int]
            结构面边界多边形的点索引(全局点云索引, 按凸包顺序排列)。
        polygon_points: Optional[np.ndarray]
            对应 polygon_point_indices 的原始 3D 坐标, 形状 (M, 3)。
        polygon_points_proj: Optional[np.ndarray]
            polygon_points 在代表平面上的投影 3D 坐标, 形状 (M, 3)。

        # 质心:
        centroid: Optional[np.ndarray]
            该结构面所有点的原始 3D 质心, 形状 (3,)。
        centroid_proj: Optional[np.ndarray]
            centroid 投影到代表平面上的 3D 坐标, 形状 (3,)。

        # 面积与迹线/宽度:
        area: float
            多边形面积, 在代表平面上的面积(与坐标单位的平方一致)。
        trace_length: float
            基于 polygon_points_proj 计算的迹线长度(最大点对距离)。
        trace_vertex1: Optional[np.ndarray]
            迹线端点 1, 3D 坐标(在平面上的投影点)。
        trace_vertex2: Optional[np.ndarray]
            迹线端点 2, 3D 坐标(在平面上的投影点)。
        width_max: float
            沿迹线正交方向, 在多边形内部采样得到的最大宽度。
        width_avg: float
            同上, 采样宽度的平均值。
        width_min: float
            同上, 采样宽度的最小值。

    典型调用流程:
        1) 检测算法生成 Discontinuity 对象, 填充 segments 和 point_indices;
        2) 在导出/分析前调用:
           disc.ComputeGeometry(point_cloud)
        3) 之后即可使用:
           - disc.area
           - disc.polygon_points / disc.polygon_points_proj
           - disc.trace_length, disc.trace_vertex1/2
           - disc.width_max / width_avg / width_min
           进行可视化和统计分析。
    """

    def __init__(
            self,
            segments: List[Segment],
            plane: Plane,
            dip: float,
            dip_direction: float,
            roughness: float = 0.0,
            algorithm_name: str = ""
    ):
        # 创建时需给出的参数
        self.segments = segments  # 结构面包含的连通片列表
        self.plane = plane
        self.dip = dip  # 倾角/倾向 (创建时给出)
        self.dip_direction = dip_direction
        self.roughness = roughness  # 粗糙度 统计指标, 创建时给出
        self.algorithm_name = algorithm_name
        self.point_indices: List[int] = self._CollectAllPointIndices()  # 结构面所有点的全局索引(各segment.point_indices 的并集)

        # 凸包边界点信息(在调用 ComputeGeometry 后填充)
        self.polygon_point_indices: List[int] = []  # 凸包边界点在全局点云中的索引(按凸包顺序)
        self.polygon_points: Optional[np.ndarray] = None  # 原始 3D 边界点坐标数组, shape (M, 3)
        self.polygon_points_proj: Optional[np.ndarray] = None  # 投影到结构面平面的 3D 边界点, shape (M, 3)
        self.area: float = 0.0  # 结构面多边形面积(单位与坐标单位平方一致)
        self.centroid: Optional[np.ndarray] = None  # 所有点的原始三维质心, shape (3,)
        self.centroid_proj: Optional[np.ndarray] = None  # centroid 投影到结构面平面的 3D 点, shape (3,)

        # 迹线和宽度(在调用 ComputeGeometry 后填充)
        self.trace_length: float = 0.0  # 基于 polygon_points_proj 的迹长
        self.trace_vertex1: Optional[np.ndarray] = None  # 迹线端点1, 3D 点(投影点)
        self.trace_vertex2: Optional[np.ndarray] = None  # 迹线端点2, 3D 点(投影点)
        self.width_max: float = 0.0  # 迹线正交方向的最大宽度
        self.width_avg: float = 0.0  # 平均宽度
        self.width_min: float = 0.0  # 最小宽度

        # Polygon 几何结果的缓存标记(用于避免重复计算)
        self._polygon_valid: bool = False
        self._cached_plane_normal: Optional[np.ndarray] = None
        self._cached_plane_d: Optional[float] = None

        # 结构面的聚类信息
        self.cluster_id = -1
        self.cluster_membership = 0.0
        self.cluster_is_noise = True
        self.cluster_confidence = 0.0

    def _CollectAllPointIndices(self) -> List[int]:
        """
        功能简介:
            汇总所有 segment 的点索引, 取并集并排序, 得到当前结构面上
            所有属于该结构面的点在全局点云中的索引列表.

        输入:
            无(直接使用 self.segments).

        输出:
            indices: List[int]
                去重且排序后的全局点索引列表.
        """
        indices: List[int] = []
        for seg in self.segments:
            indices.extend(seg.point_indices)
        if not indices:
            return []
        # 去重并排序
        indices_unique = sorted(set(int(i) for i in indices))
        return indices_unique

    # ---------------------------------------------------------
    # 多边形/面积/迹长/宽度计算
    # ---------------------------------------------------------
    def ComputeGeometry(self, point_cloud: "PointCloud", force_recompute: bool = False) -> None:
        """
        功能简介:
            基于当前结构面的所有点(由 self.point_indices 指定),
            在代表平面上计算:
                - 凸包边界多边形(原始/投影 3D 边界点及其索引)
                - 多边形面积
                - 迹线(最大点对距离)及其 3D 端点
                - 沿迹线正交方向的宽度统计(width_max/width_avg/width_min)

        实现思路(更新版):
            0) 引入简单缓存机制:
               - 若上一次已计算过多边形且:
                    * 本次未传入新的 plane, 或
                    * 传入的 plane 与上次的平面参数近似一致,
                 且 force_recompute=False, 则直接返回, 避免重复计算.
            1) 若 plane 参数给定, 直接作为代表平面 plane_ref;
               否则从 segments 中筛选 plane 不为 None 的平面,
               使用简单打分(例如: segment 点数)选出一个代表平面。
            2) 基于去重后的self.point_indices提取所有 3D 点 pts3d。
            3) 计算所有点的原始质心 centroid, 并投影到 plane_ref 得到 centroid_proj。
            4) 将所有点投影到 plane_ref, 得到 pts3d_proj。
            5) 在 plane_ref 上构造局部 2D 坐标系 (u, v), 以 centroid_proj 为原点,
               将 pts3d_proj 映射到 2D: pts2d = (dot(q,u), dot(q,v))。
            6) 对 pts2d 做 ConvexHull, 得到边界顶点顺序 hull.vertices, 从而得到:
                - polygon_point_indices (全局索引)
                - polygon_points (原始 3D)
                - polygon_points_proj (投影 3D)
            7) 使用“鞋带公式”在 2D 平面上计算多边形面积, 赋值到 self.area。
            8) 基于 2D 边界多边形:
                - 使用“最远点对”求迹线端点(2D), 迹长为最大点对距离;
                - 将端点映射回 3D(利用 polygon_points_proj), 填充 trace_vertex1/2, trace_length。
            9) 估算点间平均间距:
                - 令 avg_spacing ≈ sqrt(area / N_all), N_all 为结构面的点数。
               按此步长在迹线上均匀采样若干点(至少 3 个, 至多 200 个)。
               对每个采样点:
                - 在 2D 中沿迹线的正交方向构造一条直线, 与凸多边形求交;
                - 若求交得到一段线段, 其长度即为该位置的宽度;
               汇总所有宽度, 计算 width_max/width_avg/width_min。

        输入:
            point_cloud: PointCloud
                全局点云对象, 用于根据索引访问点坐标。
            plane: Optional[Plane]
                代表该结构面的平面; 若为 None, 则从 self.segments 中自动选取。
            force_recompute: bool
                若为 True, 无视缓存, 强制重新计算。

        输出:
            无, 但会原地更新:
                self.polygon_point_indices
                self.polygon_points
                self.polygon_points_proj
                self.centroid
                self.centroid_proj
                self.area
                self.trace_length
                self.trace_vertex1 / self.trace_vertex2
                self.width_max / self.width_avg / self.width_min
        """
        # ---------- -1. 缓存判断: 若已有有效结果且平面未变化, 且不强制重算, 则直接返回 ----------
        if (not force_recompute) and getattr(self, "_polygon_valid", False):
            # 未传入新平面, 默认沿用上一次的代表平面
            if self.plane is None:
                return
            # 传入了新平面, 若与上次使用的平面足够接近, 也直接返回
            if getattr(self, "_cached_plane_normal", None) is not None:
                n_new = np.asarray(self.plane.normal, dtype=np.float64)
                d_new = float(self.plane.d)
                if (np.allclose(self._cached_plane_normal, n_new, atol=1e-6) and
                        abs(self._cached_plane_d - d_new) < 1e-6):
                    return

        # ---------- 0. 内部工具函数 ----------
        def _ResetGeometryFields() -> None:
            """重置所有几何相关字段以及缓存标记。"""
            self.polygon_point_indices = []
            self.polygon_points = None
            self.polygon_points_proj = None
            self.centroid = None
            self.centroid_proj = None
            self.area = 0.0
            self.trace_length = 0.0
            self.trace_vertex1 = None
            self.trace_vertex2 = None
            self.width_max = 0.0
            self.width_avg = 0.0
            self.width_min = 0.0
            # 缓存无效
            self._polygon_valid = False
            self._cached_plane_normal = None
            self._cached_plane_d = None

        def _MarkPolygonValid(plane_used: "Plane") -> None:
            """标记当前多边形结果已对应该平面, 后续可直接复用。"""
            self._polygon_valid = True
            self._cached_plane_normal = np.asarray(plane_used.normal, dtype=np.float64)
            self._cached_plane_d = float(plane_used.d)

        # ---------- 1. 汇总点索引 ----------
        if not self.point_indices:
            _ResetGeometryFields()
            return

        if len(self.point_indices) < 3:
            # 点数不足以形成多边形, 只记录原始质心等简单信息
            pts3d_small = np.array(
                [[point_cloud.points[int(i)].x,
                  point_cloud.points[int(i)].y,
                  point_cloud.points[int(i)].z] for i in self.point_indices], dtype=np.float64)
            _ResetGeometryFields()
            self.polygon_point_indices = self.point_indices
            self.polygon_points = pts3d_small
            if pts3d_small.shape[0] > 0:
                self.centroid = np.mean(pts3d_small, axis=0)
            return

        # ---------- 2. 构造 3D 点集合 ----------
        pts3d = np.array(
            [[point_cloud.points[i].x,
              point_cloud.points[i].y,
              point_cloud.points[i].z] for i in self.point_indices], dtype=np.float64)

        centroid_raw = np.mean(pts3d, axis=0)
        self.centroid = centroid_raw

        # ---------- 3. 选取代表平面 plane_ref ----------
        if self.plane is not None:
            plane_ref = self.plane
        else:
            candidate_planes: List["Plane"] = []
            for seg in self.segments:
                if getattr(seg, "plane", None) is not None:
                    candidate_planes.append(seg.plane)

            if not candidate_planes:
                # 无可用平面, 无法进行平面投影
                _ResetGeometryFields()
                self.centroid = centroid_raw
                return

            def _ScorePlane(p: "Plane") -> float:
                count = 0
                for seg in self.segments:
                    if seg.plane is p:
                        count += len(seg.point_indices)
                return float(count)

            plane_ref = max(candidate_planes, key=_ScorePlane)

        # 代表平面的法向与质心
        n = np.array(plane_ref.normal, dtype=np.float64)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            _ResetGeometryFields()
            self.centroid = centroid_raw
            return
        n = n / n_norm

        plane_centroid = np.array(plane_ref.centroid, dtype=np.float64)

        # ---------- 4. 质心/所有点投影到平面 ----------
        vec_c = centroid_raw - plane_centroid
        dist_c = float(np.dot(vec_c, n))
        centroid_proj = centroid_raw - dist_c * n
        self.centroid_proj = centroid_proj

        vec_all = pts3d - plane_centroid[None, :]
        dist_all = vec_all @ n
        pts3d_proj = pts3d - dist_all[:, None] * n

        # ---------- 5. 构造局部 2D 坐标系 (u, v) ----------
        ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(np.dot(ref, n))) > 0.9:
            ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        u = np.cross(ref, n)
        u_norm = np.linalg.norm(u)
        if u_norm < 1e-12:
            _ResetGeometryFields()
            self.centroid = centroid_raw
            self.centroid_proj = centroid_proj
            return
        u = u / u_norm
        v = np.cross(n, u)

        q = pts3d_proj - centroid_proj[None, :]
        u_coord = q @ u
        v_coord = q @ v
        pts2d = np.stack([u_coord, v_coord], axis=1)  # (N, 2)

        # ---------- 6. 计算 2D 凸包, 得到边界多边形 ----------
        try:
            hull = ConvexHull(pts2d)
        except Exception:
            _ResetGeometryFields()
            self.centroid = centroid_raw
            self.centroid_proj = centroid_proj
            return

        indices_unique = np.unique(np.asarray(self.point_indices, dtype=int))
        hull_indices_local = hull.vertices  # 在 self.point_indices 中的下标
        polygon_indices_global = indices_unique[hull_indices_local]
        polygon_points = pts3d[hull_indices_local]
        polygon_points_proj = pts3d_proj[hull_indices_local]

        self.polygon_point_indices = polygon_indices_global.tolist()
        self.polygon_points = polygon_points
        self.polygon_points_proj = polygon_points_proj

        poly2d = pts2d[hull_indices_local]  # (M, 2)
        m = poly2d.shape[0]
        if m < 3:
            # 只有 1–2 个边界点, 面积/宽度为 0
            self.area = 0.0
            self.trace_length = 0.0
            self.trace_vertex1 = None
            self.trace_vertex2 = None
            self.width_max = 0.0
            self.width_avg = 0.0
            self.width_min = 0.0
            _MarkPolygonValid(plane_ref)
            return

        # ---------- 7. 使用鞋带公式计算面积 ----------
        x = poly2d[:, 0]
        y = poly2d[:, 1]
        area = 0.5 * abs(
            float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        )
        self.area = float(area)

        # ---------- 8. 计算迹线(最大点对距离) ----------
        trace_len_sq = 0.0
        idx1 = 0
        idx2 = 1
        for i in range(m):
            pi = poly2d[i]
            for j in range(i + 1, m):
                pj = poly2d[j]
                d2 = float(np.sum((pi - pj) ** 2))
                if d2 > trace_len_sq:
                    trace_len_sq = d2
                    idx1 = i
                    idx2 = j

        if trace_len_sq <= 0.0:
            self.trace_length = 0.0
            self.trace_vertex1 = None
            self.trace_vertex2 = None
            self.width_max = 0.0
            self.width_avg = 0.0
            self.width_min = 0.0
            _MarkPolygonValid(plane_ref)
            return

        trace_length = float(math.sqrt(trace_len_sq))
        self.trace_length = trace_length
        self.trace_vertex1 = polygon_points_proj[idx1].astype(np.float64)
        self.trace_vertex2 = polygon_points_proj[idx2].astype(np.float64)

        # ---------- 9. 沿迹线正交方向估算宽度统计 ----------
        n_points_all = len(self.point_indices)
        if area > 0.0 and n_points_all > 0.0:
            avg_spacing = math.sqrt(area / n_points_all)
        else:
            avg_spacing = 0.0

        if avg_spacing <= 0.0:
            self.width_max = 0.0
            self.width_avg = 0.0
            self.width_min = 0.0
            _MarkPolygonValid(plane_ref)
            return

        est_samples = int(trace_length / avg_spacing)
        num_samples = max(3, min(200, est_samples))
        if num_samples < 3:
            num_samples = 3

        p0_2d = poly2d[idx1]
        p1_2d = poly2d[idx2]
        trace_vec = p1_2d - p0_2d
        trace_len = float(np.linalg.norm(trace_vec))
        if trace_len <= 1e-12:
            self.width_max = 0.0
            self.width_avg = 0.0
            self.width_min = 0.0
            _MarkPolygonValid(plane_ref)
            return

        trace_dir = trace_vec / trace_len
        ortho_dir = np.array([-trace_dir[1], trace_dir[0]], dtype=np.float64)

        widths: List[float] = []

        for k in range(num_samples):
            t = k / float(num_samples - 1)
            sample_center = p0_2d + t * trace_vec

            intersection_points: List[np.ndarray] = []

            for i in range(m):
                pa = poly2d[i]
                pb = poly2d[(i + 1) % m]
                edge_vec = pb - pa

                A = np.column_stack((ortho_dir, -edge_vec))  # 2x2
                b = pa - sample_center

                detA = float(np.linalg.det(A))
                if abs(detA) < 1e-12:
                    continue

                try:
                    params = np.linalg.solve(A, b)
                except Exception:
                    continue

                u_param = float(params[1])
                if -1e-8 <= u_param <= 1.0 + 1e-8:
                    inter_pt = pa + u_param * edge_vec
                    intersection_points.append(inter_pt)

            if len(intersection_points) < 2:
                continue

            inter_arr = np.vstack(intersection_points)
            proj_vals = inter_arr @ ortho_dir
            idx_sort = np.argsort(proj_vals)
            p_min = inter_arr[idx_sort[0]]
            p_max = inter_arr[idx_sort[-1]]
            width_val = float(np.linalg.norm(p_max - p_min))
            if width_val > 0.0:
                widths.append(width_val)

        if not widths:
            self.width_max = 0.0
            self.width_avg = 0.0
            self.width_min = 0.0
            _MarkPolygonValid(plane_ref)
            return

        widths_arr = np.asarray(widths, dtype=np.float64)
        self.width_max = float(np.max(widths_arr))
        self.width_avg = float(np.mean(widths_arr))
        self.width_min = float(np.min(widths_arr))

        # 计算流程完全结束, 标记本次结果可缓存复用
        _MarkPolygonValid(plane_ref)


class Cluster:
    """
    功能简介:
        表示算法中的一般聚类结果(如法向聚类), 不一定直接对应结构面.
    """

    def __init__(
            self,
            point_indices: List[int],
            feature_center: Optional[Dict[str, float]] = None
    ):
        self.point_indices = point_indices
        self.feature_center = feature_center if feature_center is not None else {}


class RockBlock:
    """
    功能简介:
        表示由多个结构面围成的岩体块体(预留).
    """

    def __init__(self, discontinuities: List[Discontinuity]):
        self.discontinuities = discontinuities
        # TODO: 可扩展体积、重心等属性



