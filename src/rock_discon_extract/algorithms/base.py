from typing import List

from ..logging_utils import LoggerManager, Timer
from ..geometry import Discontinuity
from ..pointcloud import PointCloud


class PlaneDetectionAlgorithm:
    """
    功能简介:
        所有平面/结构面检测算法的抽象基类, 统一接口与日志记录方式.
    """

    def __init__(self, name: str):
        self.name = name
        self.logger = LoggerManager.GetLogger(self.__class__.__name__)

    def DetectPlanes(self, point_cloud: PointCloud) -> List[Discontinuity]:
        """
        功能简介:
            统一封装的对外检测接口, 内部带计时与日志.
        """
        with Timer(f"{self.name}.DetectPlanes", self.logger):
            discontinuities = self._DetectImpl(point_cloud)
        self.logger.info(f"{self.name} detected {len(discontinuities)} discontinuities.")
        return discontinuities

    def _DetectImpl(self, point_cloud: PointCloud) -> List[Discontinuity]:
        raise NotImplementedError
