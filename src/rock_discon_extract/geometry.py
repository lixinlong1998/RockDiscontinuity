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
        表示岩体中的一个结构面(节理/断层等), 包含一个或多个连通片段.
    参数说明:
        segments: List[Segment]
            组成该结构面的所有连通片(一个结构面可以由多个离散片段组成).
        dip: float
            倾角(度).
        dip_direction: float
            倾向(度, 0~360).
        roughness: float
            结构面粗糙度指标(由外部算法定义), 默认 0.0.
        algorithm_name: str
            生成该结构面的算法名称(例如 'RANSAC_open3d').
    """

    def __init__(
            self,
            segments: List[Segment],
            dip: float,
            dip_direction: float,
            roughness: float = 0.0,
            algorithm_name: str = ""
    ):
        self.segments = segments
        self.dip = dip
        self.dip_direction = dip_direction
        self.roughness = roughness
        self.algorithm_name = algorithm_name

        # 结构面上所有点的全局索引(各 segment.point_indices 的并集)
        self.point_indices: List[int] = self._CollectAllPointIndices()

        # 凸包边界点信息(在调用 ComputePolygonAndArea 后填充)
        #   polygon_point_indices: 凸包边界点在全局点云中的索引(按凸包顺序)
        #   polygon_points: 对应的 3D 坐标数组, 形状 (M, 3)
        #   area: 结构面多边形面积(单位与坐标单位平方一致)
        self.polygon_point_indices: List[int] = []
        self.polygon_points: Optional[np.ndarray] = None
        self.area: float = 0.0

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

    def ComputePolygonAndArea(self, point_cloud, plane=None) -> None:
        """
        功能简介:
            对当前 Discontinuity 对应的所有点进行平面投影与凸包计算,
            得到边界多边形顶点坐标与对应面积, 并保存在 polygon_points、
            polygon_point_indices 和 area 中。

        实现思路:
            1) 从 self.point_indices 中去重后提取所有 3D 坐标;
            2) 选择代表平面 plane_ref:
               - 若参数 plane 不为 None, 使用该平面;
               - 否则, 若 self.segments 非空, 使用第一个 segment 的 plane;
               - 否则无法计算, 将 area 置为 0.0, polygon_points 置为 None;
            3) 取平面法向 n 和质心 centroid, 构造局部 2D 坐标系 (u, v):
               - 选取 ref 向量(避免与 n 平行), 令 u = normalize(ref × n),
                 再令 v = n × u;
            4) 将所有点相对 centroid 平移, 投影到 (u, v) 上得到 2D 坐标;
            5) 使用 ConvexHull 对 2D 点集做凸包, 得到顶点顺序 hull.vertices;
            6) 根据凸包索引得到:
               - polygon_point_indices: 对应全局索引;
               - polygon_points: 对应 3D 坐标;
            7) 使用鞋带公式在 2D 坐标下计算多边形面积, 赋值给 self.area。

        输入:
            point_cloud:
                PointCloud 对象, 需包含属性 points, 且 points[i] 至少有 x,y,z.
            plane:
                可选的代表平面(Plane). 若为 None, 默认取第一个 segment 的 plane.

        输出:
            无(结果直接写入对象属性).
        """
        # 点数不足三点, 无法形成多边形
        if not self.point_indices or len(self.point_indices) < 3:
            self.polygon_point_indices = []
            self.polygon_points = None
            self.area = 0.0
            return

        # 代表平面
        plane_ref = plane
        if plane_ref is None:
            if not self.segments:
                self.polygon_point_indices = []
                self.polygon_points = None
                self.area = 0.0
                return
            plane_ref = self.segments[0].plane

        # 去重后的全局点索引
        indices_unique = np.unique(np.asarray(self.point_indices, dtype=int))
        if indices_unique.shape[0] < 3:
            # 仍不足 3 点, 仅记录点而不定义面积
            pts3d = np.array(
                [[point_cloud.points[i].x,
                  point_cloud.points[i].y,
                  point_cloud.points[i].z]
                 for i in indices_unique],
                dtype=float
            )
            self.polygon_point_indices = indices_unique.tolist()
            self.polygon_points = pts3d
            self.area = 0.0
            return

        # 提取 3D 坐标
        pts3d = np.array(
            [[point_cloud.points[int(i)].x,
              point_cloud.points[int(i)].y,
              point_cloud.points[int(i)].z]
             for i in indices_unique],
            dtype=float
        )

        # 法向与质心
        n = np.array(plane_ref.normal, dtype=float)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            # 平面退化, 不做凸包, 仅记录所有点
            self.polygon_point_indices = indices_unique.tolist()
            self.polygon_points = pts3d
            self.area = 0.0
            return
        n = n / n_norm
        centroid = np.array(plane_ref.centroid, dtype=float)

        # 构造局部坐标系 (u, v)
        ref = np.array([0.0, 0.0, 1.0], dtype=float)
        if abs(np.dot(n, ref)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0], dtype=float)
        u = np.cross(ref, n)
        u_norm = np.linalg.norm(u)
        if u_norm < 1e-12:
            self.polygon_point_indices = indices_unique.tolist()
            self.polygon_points = pts3d
            self.area = 0.0
            return
        u = u / u_norm
        v = np.cross(n, u)

        # 投影到 2D
        q = pts3d - centroid
        u_coord = q @ u
        v_coord = q @ v
        pts2d = np.stack([u_coord, v_coord], axis=1)

        # 计算凸包
        try:
            hull = ConvexHull(pts2d)
        except Exception:
            # 若凸包失败, 退化为使用全部点
            self.polygon_point_indices = indices_unique.tolist()
            self.polygon_points = pts3d
            self.area = 0.0
            return

        hull_indices = hull.vertices  # 在 indices_unique 内部的索引
        boundary_points_3d = pts3d[hull_indices]
        boundary_indices_global = indices_unique[hull_indices]

        self.polygon_point_indices = boundary_indices_global.tolist()
        self.polygon_points = boundary_points_3d

        # 鞋带公式计算面积(2D 凸包)
        x = pts2d[hull_indices, 0]
        y = pts2d[hull_indices, 1]
        self.area = float(
            0.5 * abs(
                np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
            )
        )


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
