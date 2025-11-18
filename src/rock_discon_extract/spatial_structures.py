from typing import Dict, List, Optional, Tuple

from .logging_utils import LoggerManager
from .geometry import Plane
from .pointcloud import PointCloud


class Voxel:
    """
    功能简介:
        表示体素网格中的一个单元.
    """

    def __init__(
        self,
        voxel_id: Tuple[int, int, int],
        center: Tuple[float, float, float],
        size: float,
        point_indices: Optional[List[int]] = None
    ):
        self.voxel_id = voxel_id
        self.center = center
        self.size = size
        self.point_indices = point_indices if point_indices is not None else []
        self.local_plane: Optional[Plane] = None


class VoxelGrid:
    """
    功能简介:
        管理整个点云的体素划分, 提供从坐标到 Voxel 的映射.
    """

    def __init__(self, point_cloud: PointCloud, voxel_size: float):
        self.logger = LoggerManager.GetLogger(self.__class__.__name__)
        self.point_cloud = point_cloud
        self.voxel_size = voxel_size
        self.voxels: Dict[Tuple[int, int, int], Voxel] = {}
        self._BuildVoxelGrid()

    def _BuildVoxelGrid(self) -> None:
        self.logger.info("Building voxel grid (not implemented yet).")
        # TODO: 根据 point_cloud.points 将点分配到各 Voxel
        pass

    def GetNeighborVoxelIds(self, voxel_id: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """
        功能简介:
            返回给定体素索引的邻域体素索引列表.
        """
        # TODO: 实现邻域体素查找
        return []


class SuperVoxel:
    """
    功能简介:
        表示由多个体素合并形成的超体素.
    """

    def __init__(
        self,
        super_id: int,
        voxel_ids: List[Tuple[int, int, int]],
        point_indices: List[int]
    ):
        self.super_id = super_id
        self.voxel_ids = voxel_ids
        self.point_indices = point_indices
        self.plane: Optional[Plane] = None
        self.neighbor_super_ids: List[int] = []
        self.accumulated_error: float = 0.0
