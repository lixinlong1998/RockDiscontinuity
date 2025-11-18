from typing import List

from .base import PlaneDetectionAlgorithm
from ..geometry import Discontinuity
from ..pointcloud import PointCloud


class RansacDetector(PlaneDetectionAlgorithm):
    """
    功能简介:
        基于 RANSAC 的平面检测算法(骨架).
    """

    def __init__(
        self,
        distance_threshold: float,
        angle_threshold: float,
        min_inliers: int
    ):
        super().__init__(name="RANSAC")
        self.distance_threshold = distance_threshold
        self.angle_threshold = angle_threshold
        self.min_inliers = min_inliers

    def _DetectImpl(self, point_cloud: PointCloud) -> List[Discontinuity]:
        self.logger.info("RANSAC detection not implemented yet.")
        # TODO: 实现 RANSAC 平面检测
        return []
