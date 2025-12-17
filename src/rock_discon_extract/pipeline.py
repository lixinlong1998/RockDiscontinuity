from typing import List, Optional, Dict, Tuple

from .logging_utils import LoggerManager
from .geometry import Discontinuity
from .pointcloud import PointCloud
from .algorithms.base import PlaneDetectionAlgorithm, ClusteringAlgorithm, PlaneClusterInfo


class RockDiscontinuityPipeline:
    """
    功能简介:
        统一管理点云预处理、算法执行和结果汇总的管线类.
    """

    def __init__(
            self,
            point_cloud: PointCloud,
            detect_algorithms: List[PlaneDetectionAlgorithm],
            cluster_algorithms: List[ClusteringAlgorithm],
            manual_dip_dirs: Optional[List[Tuple[float, float]]] = None

    ):
        self.logger = LoggerManager.GetLogger(self.__class__.__name__)
        self.point_cloud = point_cloud
        self.detect_algorithms = detect_algorithms
        self.cluster_algorithms = cluster_algorithms
        self.manual_dip_dirs = manual_dip_dirs
        self.results: Dict[(str, str), (List[Discontinuity], List[PlaneClusterInfo])] = {}

    def RunAll(self) -> Dict[Tuple[str, str], Tuple[List[Discontinuity], List[PlaneClusterInfo]]]:
        for detect_algo in self.detect_algorithms:
            for cluster_algo in self.cluster_algorithms:
                self.logger.info(f"Running detect algorithm: {detect_algo.name}")
                self.logger.info(f"Running cluster algorithm: {cluster_algo.name}")
                discontinuities = detect_algo.DetectPlanes(self.point_cloud)
                discontinuities = detect_algo.DisconGeometry(self.point_cloud, discontinuities)
                discontinuities, clusters = cluster_algo.ClusterPlanes(discontinuities, self.manual_dip_dirs)
                self.results[(detect_algo.name, cluster_algo.name)] = (discontinuities, clusters)
        return self.results

    def RunOne(self, detect_algo: str, cluster_algo) -> Dict[
        Tuple[str, str], Tuple[List[Discontinuity], List[PlaneClusterInfo]]]:
        for detect_algo in self.detect_algorithms:
            if detect_algo.name != detect_algo:
                continue
            for cluster_algo in self.cluster_algorithms:
                if cluster_algo.name != cluster_algo:
                    continue
                self.logger.info(f"Running detect algorithm: {detect_algo.name}")
                self.logger.info(f"Running cluster algorithm: {cluster_algo.name}")
                discontinuities = detect_algo.DetectPlanes(self.point_cloud)
                discontinuities = detect_algo.DisconGeometry(self.point_cloud, discontinuities)
                discontinuities, clusters = cluster_algo.ClusterPlanes(discontinuities, self.manual_dip_dirs)
                self.results[(detect_algo.name, cluster_algo.name)] = (discontinuities, clusters)
        self.logger.warning(f"Algorithm {detect_algo} or {cluster_algo} not found.")
        return self.results
