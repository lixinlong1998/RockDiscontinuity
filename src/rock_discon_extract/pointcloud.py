from typing import List

from .logging_utils import LoggerManager
from .geometry import Point, BoundingBox, Discontinuity


class PointCloud:
    """
    功能简介:
        作为整个框架的基础数据容器, 表示一个完整的点云数据集.
    """

    def __init__(self, points: List[Point]):
        self.logger = LoggerManager.GetLogger(self.__class__.__name__)
        self.points = points
        self.bounding_box = self._ComputeBoundingBox()
        self.spatial_index = None  # 预留 KDTree 等

    def _ComputeBoundingBox(self) -> BoundingBox:
        min_x = min(p.x for p in self.points)
        min_y = min(p.y for p in self.points)
        min_z = min(p.z for p in self.points)
        max_x = max(p.x for p in self.points)
        max_y = max(p.y for p in self.points)
        max_z = max(p.z for p in self.points)
        return BoundingBox(min_x, min_y, min_z, max_x, max_y, max_z)

    def EstimateNormals(self, k_neighbor: int = 20) -> None:
        """
        功能简介:
            使用 k 近邻方法估计所有点的法向量.
        """
        self.logger.info("EstimateNormals is not implemented yet.")
        # TODO: 实现 KDTree + PCA 求法向
        pass

    def VoxelDownsample(self, voxel_size: float) -> "PointCloud":
        """
        功能简介:
            使用体素栅格下采样点云.
        """
        self.logger.info("VoxelDownsample is not implemented yet.")
        # TODO: 实现体素下采样, 当前先返回自身占位
        return self


class Chunk:
    """
    功能简介:
        表示一个大点云中的子区域(分块), 用于分块处理和并行计算.
    """

    def __init__(self, chunk_id: int, point_cloud: PointCloud):
        self.chunk_id = chunk_id
        self.point_cloud = point_cloud
        self.bounding_box = point_cloud.bounding_box
        self.discontinuities: List[Discontinuity] = []
