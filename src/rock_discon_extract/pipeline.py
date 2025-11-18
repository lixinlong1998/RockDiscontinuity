from typing import Dict, List

from .logging_utils import LoggerManager
from .geometry import Discontinuity
from .pointcloud import PointCloud
from .algorithms.base import PlaneDetectionAlgorithm


class RockDiscontinuityPipeline:
    """
    功能简介:
        统一管理点云预处理、算法执行和结果汇总的管线类.
    """

    def __init__(
        self,
        point_cloud: PointCloud,
        algorithms: List[PlaneDetectionAlgorithm]
    ):
        self.logger = LoggerManager.GetLogger(self.__class__.__name__)
        self.point_cloud = point_cloud
        self.algorithms = algorithms
        self.results: Dict[str, List[Discontinuity]] = {}

    def RunAll(self) -> Dict[str, List[Discontinuity]]:
        for algo in self.algorithms:
            self.logger.info(f"Running algorithm: {algo.name}")
            discontinuities = algo.DetectPlanes(self.point_cloud)
            self.results[algo.name] = discontinuities
        return self.results

    def RunOne(self, algo_name: str) -> List[Discontinuity]:
        for algo in self.algorithms:
            if algo.name == algo_name:
                self.logger.info(f"Running algorithm: {algo.name}")
                discontinuities = algo.DetectPlanes(self.point_cloud)
                self.results[algo.name] = discontinuities
                return discontinuities
        self.logger.warning(f"Algorithm {algo_name} not found.")
        return []
