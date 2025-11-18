from typing import List, Dict, Optional, Tuple
import math

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
