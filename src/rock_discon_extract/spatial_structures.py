from typing import Dict, List, Optional, Tuple

import math

from .logging_utils import LoggerManager
from .geometry import Plane
from .pointcloud import PointCloud


class Voxel:
    """
    功能简介:
        表示体素网格中的一个单元.

    实现思路:
        - 使用整数三元组 voxel_id = (i, j, k) 表示体素索引;
        - center 为体素中心坐标, size 为边长;
        - point_indices 存储落入该体素的全局点索引;
        - local_plane 可选, 用于缓存在该体素内拟合得到的局部平面。
    """

    def __init__(
        self,
        voxel_id: Tuple[int, int, int],
        center: Tuple[float, float, float],
        size: float,
        point_indices: Optional[List[int]] = None,
    ):
        self.voxel_id = voxel_id
        self.center = center
        self.size = float(size)
        self.point_indices = point_indices if point_indices is not None else []
        self.local_plane: Optional[Plane] = None


class VoxelGrid:
    """
    功能简介:
        管理整个点云的体素划分, 提供从坐标到 Voxel 的映射。

    实现思路:
        - 根据 PointCloud.bounding_box 计算轴对齐包围盒;
        - 给定 voxel_size, 使用 floor((x - min_x)/size) 计算体素索引 (i,j,k);
        - 使用字典 self.voxels 存储所有非空体素;
        - 提供邻域体素查询接口 GetNeighborVoxelIds, 默认 26 邻接;
        - 可被 MOE、HT-RG 等算法复用。
    """

    def __init__(self, point_cloud: PointCloud, voxel_size: float):
        self.logger = LoggerManager.GetLogger(self.__class__.__name__)
        self.point_cloud = point_cloud
        self.voxel_size = float(voxel_size)
        self.voxels: Dict[Tuple[int, int, int], Voxel] = {}

        # 整体范围
        self.min_x = 0.0
        self.min_y = 0.0
        self.min_z = 0.0
        self.max_x = 0.0
        self.max_y = 0.0
        self.max_z = 0.0

        self._BuildVoxelGrid()

    # ------------------------------------------------------------------
    # 体素构建
    # ------------------------------------------------------------------
    def _BuildVoxelGrid(self) -> None:
        """
        功能简介:
            根据 point_cloud.points 将点分配到各 Voxel 中, 构建稀疏体素网格。

        实现思路:
            1) 从 point_cloud.bounding_box 中读取整体范围, 若不存在则遍历点计算;
            2) 对于每个点, 使用 floor((coord - min)/voxel_size) 计算体素索引 (i,j,k);
            3) 在 self.voxels 中创建或获取对应 Voxel, 记录 point_indices;
            4) 仅创建非空体素。
        """
        points = self.point_cloud.points
        if not points:
            self.logger.warning("VoxelGrid: point cloud is empty, nothing to build.")
            return

        bb = getattr(self.point_cloud, "bounding_box", None)
        if bb is not None:
            self.min_x = float(bb.min_x)
            self.min_y = float(bb.min_y)
            self.min_z = float(bb.min_z)
            self.max_x = float(bb.max_x)
            self.max_y = float(bb.max_y)
            self.max_z = float(bb.max_z)
        else:
            xs = [p.x for p in points]
            ys = [p.y for p in points]
            zs = [p.z for p in points]
            self.min_x = float(min(xs))
            self.min_y = float(min(ys))
            self.min_z = float(min(zs))
            self.max_x = float(max(xs))
            self.max_y = float(max(ys))
            self.max_z = float(max(zs))

        inv_size = 1.0 / self.voxel_size

        for idx, p in enumerate(points):
            i = int(math.floor((p.x - self.min_x) * inv_size))
            j = int(math.floor((p.y - self.min_y) * inv_size))
            k = int(math.floor((p.z - self.min_z) * inv_size))
            voxel_id = (i, j, k)

            if voxel_id not in self.voxels:
                center = (
                    self.min_x + (i + 0.5) * self.voxel_size,
                    self.min_y + (j + 0.5) * self.voxel_size,
                    self.min_z + (k + 0.5) * self.voxel_size,
                )
                self.voxels[voxel_id] = Voxel(
                    voxel_id=voxel_id,
                    center=center,
                    size=self.voxel_size,
                    point_indices=[idx],
                )
            else:
                self.voxels[voxel_id].point_indices.append(idx)

        self.logger.info(
            f"VoxelGrid: built {len(self.voxels)} non-empty voxels "
            f"from {len(points)} points, voxel_size={self.voxel_size}."
        )

    # ------------------------------------------------------------------
    # 邻域查询
    # ------------------------------------------------------------------
    def GetNeighborVoxelIds(
        self, voxel_id: Tuple[int, int, int]
    ) -> List[Tuple[int, int, int]]:
        """
        功能简介:
            返回给定体素索引的邻域体素索引列表(26 邻接)。
        """
        if voxel_id not in self.voxels:
            return []

        i0, j0, k0 = voxel_id
        neighbor_ids: List[Tuple[int, int, int]] = []

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    nb_id = (i0 + dx, j0 + dy, k0 + dz)
                    if nb_id in self.voxels:
                        neighbor_ids.append(nb_id)

        return neighbor_ids

    # ------------------------------------------------------------------
    # 便捷访问接口
    # ------------------------------------------------------------------
    def GetVoxel(self, voxel_id: Tuple[int, int, int]) -> Optional[Voxel]:
        """
        功能简介:
            根据体素索引获取 Voxel 对象, 若不存在则返回 None。
        """
        return self.voxels.get(voxel_id)

    def GetAllVoxels(self) -> List[Voxel]:
        """
        功能简介:
            返回当前网格中所有非空体素对象列表。
        """
        return list(self.voxels.values())


class SuperVoxel:
    """
    功能简介:
        表示由多个体素合并形成的超体素.

    实现思路:
        - super_id 为全局唯一 ID;
        - voxel_ids 为组成该超体素的体素索引列表;
        - point_indices 为所有包含点的全局索引集合;
        - plane 可选, 表示在该超体素上拟合得到的局部平面;
        - neighbor_super_ids 用于存储超体素级别的邻接关系。
    """

    def __init__(
        self,
        super_id: int,
        voxel_ids: List[Tuple[int, int, int]],
        point_indices: List[int],
    ):
        self.super_id = int(super_id)
        self.voxel_ids = voxel_ids
        self.point_indices = point_indices
        self.plane: Optional[Plane] = None
        self.neighbor_super_ids: List[int] = []
        self.accumulated_error: float = 0.0

    def AddNeighbor(self, other_id: int) -> None:
        """
        功能简介:
            在超体素之间建立邻接关系(去重)。
        """
        if other_id not in self.neighbor_super_ids:
            self.neighbor_super_ids.append(other_id)

    def SetPlane(self, plane: Plane) -> None:
        """
        功能简介:
            绑定在该超体素上拟合得到的平面对象。
        """
        self.plane = plane
