from typing import List

from .geometry import Discontinuity
from .pointcloud import PointCloud


class Visualizer:
    """
    功能简介:
        提供与点云和结构面相关的可视化接口(占位).
    """

    def ShowPointCloud(self, point_cloud: PointCloud) -> None:
        # TODO: 使用 open3d 或其它库实现
        pass

    def ShowDiscontinuities(
        self,
        point_cloud: PointCloud,
        discontinuities: List[Discontinuity]
    ) -> None:
        # TODO: 实现结构面结果可视化
        pass
