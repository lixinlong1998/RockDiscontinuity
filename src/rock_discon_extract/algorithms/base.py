import numpy as np
from typing import List, Optional, Dict, Tuple

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

    def DisconGeometry(self, point_cloud: PointCloud, discontinuities: List[Discontinuity]) -> List[Discontinuity]:
        with Timer(f"{self.name}.DetectPlanes", self.logger):
            for disc_id, disc in enumerate(discontinuities):
                if not disc.segments:
                    continue
                # 计算disc几何信息
                disc.ComputeGeometry(point_cloud)
        self.logger.info(f"{self.name} {len(discontinuities)} discontinuities geometry computed.")
        return discontinuities

    def get_parameters(self) -> Dict:
        """返回当前算法对象的参数字典(用于导出/复现实验)。"""
        params: Dict = {
            "algo_name": self.name,
            "algo_type": self.__class__.__name__,
        }
        # 约定: 只导出“公开成员变量”(不以下划线开头)，避免把日志器/缓存/中间大对象写进参数文件
        for k, v in self.__dict__.items():
            if k in ("logger", "name"):
                continue
            if k.startswith("_"):
                continue
            if callable(v):
                continue
            params[k] = _ToJsonableValue(v)
        return params


class PlaneClusterInfo:
    """
    功能简介:
        描述法向聚类结果中单个簇的统计信息, 包括簇中心法向、样本数量以及簇置信度等.

    实现思路:
        - 在聚类完成后, 对每个簇统计:
            * cluster_id: 簇编号
            * center_normal: 簇中心的单位法向 (统一上半球)
            * num_members: 簇内被分配样本的数量(不含噪声)
            * confidence: 根据隶属度和质量权重计算的簇整体置信度
            * is_manual_seed: 该簇是否来源于人工给定的初始中心
        - 这些信息可用于后续的统计分析与可视化, 也可以写入日志中.

    输入参数:
        cluster_id: int
            簇编号.
        center_normal: np.ndarray
            单位法向数组, 形状 (3,).
        num_members: int
            被分配到该簇的非噪声结构面数量.
        confidence: float
            簇整体置信度指标.
        is_manual_seed: bool
            是否至少部分来源于人工种子簇.

    输出:
        PlaneClusterInfo 实例, 由 ClusterDiscontinuities 返回.
    """

    def __init__(
            self,
            cluster_id: int,
            center_normal: np.ndarray,
            num_members: int,
            confidence: float,
            is_manual_seed: bool
    ):
        self.cluster_id = cluster_id
        self.center_normal = center_normal
        self.num_members = num_members
        self.confidence = confidence
        self.is_manual_seed = is_manual_seed


class ClusteringAlgorithm:
    """
    功能简介:
        所有平面/结构面聚类算法的抽象基类, 统一接口与日志记录方式.
    """

    def __init__(self, name: str):
        self.name = name
        self.logger = LoggerManager.GetLogger(self.__class__.__name__)

    def ClusterPlanes(self,
                      discontinuities: List[Discontinuity],
                      manual_dip_dirs: Optional[List[Tuple[float, float]]] = None,
                      ) -> Tuple[List[Discontinuity], List[PlaneClusterInfo]]:
        """
        功能简介:
            统一封装的对外检测接口, 内部带计时与日志.
        """
        with Timer(f"{self.name}.ClusterPlanes", self.logger):
            discontinuities, clusters = self._ClusterImpl(discontinuities, manual_dip_dirs)
            clustered_discon_number = len([disc.cluster_id for disc in discontinuities if disc.cluster_id != -1])
        self.logger.info(
            f"{self.name} Clustered {clustered_discon_number} of {len(discontinuities)} discontinuities into {clustered_discon_number} groups.")
        return discontinuities, clusters

    def _ClusterImpl(self,
                     discontinuities: List[Discontinuity],
                     manual_dip_dirs: Optional[List[Tuple[float, float]]] = None,
                     ) -> Tuple[List[Discontinuity], List[PlaneClusterInfo]]:
        raise NotImplementedError

    def get_parameters(self) -> Dict:
        """返回当前算法对象的参数字典(用于导出/复现实验)。"""
        params: Dict = {
            "algo_name": self.name,
            "algo_type": self.__class__.__name__,
        }
        for k, v in self.__dict__.items():
            if k in ("logger", "name"):
                continue
            if k.startswith("_"):
                continue
            if callable(v):
                continue
            params[k] = _ToJsonableValue(v)
        return params


def _ToJsonableValue(value):
    """将常见的 Python / numpy 类型转换为可 JSON 序列化的对象。
    - 对于过大的数组/列表，仅记录 shape/len，避免输出文件爆炸。
    """
    try:
        import numpy as _np
    except Exception:
        _np = None

    # 基础标量
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    # numpy 标量
    if _np is not None and isinstance(value, _np.generic):
        return value.item()

    # numpy 数组
    if _np is not None and isinstance(value, _np.ndarray):
        if value.size <= 50:
            return value.tolist()
        return {"__type__": "ndarray", "shape": list(value.shape), "dtype": str(value.dtype), "size": int(value.size)}

    # 容器
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            out[str(k)] = _ToJsonableValue(v)
        return out

    if isinstance(value, (list, tuple)):
        if len(value) <= 50:
            return [_ToJsonableValue(v) for v in value]
        return {"__type__": type(value).__name__, "len": len(value)}

    if isinstance(value, set):
        value_list = list(value)
        if len(value_list) <= 50:
            return [_ToJsonableValue(v) for v in value_list]
        return {"__type__": "set", "len": len(value_list)}

    # pathlib.Path / os.PathLike
    try:
        import os as _os
        if isinstance(value, _os.PathLike):
            return str(value)
    except Exception:
        pass

    # 兜底: 使用字符串表示
    return str(value)
