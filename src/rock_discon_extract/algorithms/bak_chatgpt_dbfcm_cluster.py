from typing import List, Optional, Dict, Tuple
import math
import numpy as np
from collections import deque

from .geometry import Discontinuity  # 依赖当前项目的几何定义
from .logging_utils import LoggerManager, Timer


class ClusterConfig:
    """
    功能简介:
        聚类算法的参数配置类, 用于控制 DBSCAN、FCM 以及质量权重计算方式等.

    实现思路:
        - 将所有可调参数集中到一个配置类中, 方便在外部统一设置与修改;
        - 支持两种质量权重模式: "piecewise" 与 "continuous";
        - DBSCAN 相关参数 (eps, min_samples_ratio) 用于第一层初始聚类;
        - FCM 相关参数 (m, max_iter, tol 等) 用于后续带权模糊聚类;
        - 提供可选的可视化开关, 由外部根据环境决定是否绘图.

    输入参数:
        rmse_weight: float
            在质量优先度 q 计算中, RMSE 项的权重 (默认 0.4).
        length_weight: float
            在质量优先度 q 计算中, 迹长项的权重 (默认 0.6).
        first_layer_ratio: float
            第一层高质量样本所占比例 (0-1), 默认 0.6 表示前 60%.
        dbscan_eps_deg: float
            DBSCAN 的角度阈值 eps, 单位为度.
        dbscan_min_samples_ratio: float
            DBSCAN 中最小样本数占当前样本总数的比例 (0-1), 实际值会向上取整并至少为 3.
        quality_mode: str
            质量权重模式, 可选 "piecewise" 或 "continuous".
        quality_gamma: float
            在 continuous 模式下的指数参数 γ, γ>1 时高质量样本权重更高.
        quality_min_weight: float
            最低质量权重 s_min, 用于两个模式中的权重下界.
        low_quality_ratio: float
            全体样本中作为 "低质量" 样本的比例 (0-1), 用于二级 RMSE 调权 (默认 0.2).
        rmse_gamma: float
            二级 RMSE 调权参数 λ, 控制 β_i = exp(-λ * RMSEnorm) 的衰减强度.
        fcm_m: float
            Fuzzy C-Means 的模糊指数 m (一般取 2.0).
        fcm_max_iter: int
            FCM 最大迭代次数.
        fcm_tol_deg: float
            FCM 收敛判据: 簇中心最大变化角度(度) 小于该值则停止.
        noise_membership_threshold: float
            最大隶属度 μ_max 低于该阈值的样本视为噪声/未分配.
        merge_angle_eps_deg: float
            簇中心之间若夹角小于该阈值(度), 则认为两簇过近, 在后处理阶段合并.
        max_merge_loops: int
            簇合并-微调过程的最大循环次数.
        enable_plot: bool
            是否在聚类完成后进行可视化绘图 (若环境允许).

    输出:
        无直接输出, ClusterConfig 实例将被 ClusterDiscontinuities 使用.
    """

    def __init__(
            self,
            rmse_weight: float = 0.4,
            length_weight: float = 0.6,
            first_layer_ratio: float = 0.6,
            dbscan_eps_deg: float = 10.0,
            dbscan_min_samples_ratio: float = 0.05,
            quality_mode: str = "piecewise",
            quality_gamma: float = 1.5,
            quality_min_weight: float = 0.1,
            low_quality_ratio: float = 0.2,
            rmse_gamma: float = 1.0,
            fcm_m: float = 2.0,
            fcm_max_iter: int = 100,
            fcm_tol_deg: float = 1e-2,
            noise_membership_threshold: float = 0.4,
            merge_angle_eps_deg: float = 5.0,
            max_merge_loops: int = 3,
            enable_plot: bool = False
    ):
        self.rmse_weight = rmse_weight
        self.length_weight = length_weight
        self.first_layer_ratio = first_layer_ratio
        self.dbscan_eps_deg = dbscan_eps_deg
        self.dbscan_min_samples_ratio = dbscan_min_samples_ratio
        self.quality_mode = quality_mode
        self.quality_gamma = quality_gamma
        self.quality_min_weight = quality_min_weight
        self.low_quality_ratio = low_quality_ratio
        self.rmse_gamma = rmse_gamma
        self.fcm_m = fcm_m
        self.fcm_max_iter = fcm_max_iter
        self.fcm_tol_deg = fcm_tol_deg
        self.noise_membership_threshold = noise_membership_threshold
        self.merge_angle_eps_deg = merge_angle_eps_deg
        self.max_merge_loops = max_merge_loops
        self.enable_plot = enable_plot


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


def ClusterDiscontinuities(
        discontinuities: List[Discontinuity],
        config: Optional[ClusterConfig] = None,
        manual_dip_dirs: Optional[List[Tuple[float, float]]] = None
) -> List[PlaneClusterInfo]:
    """
    功能简介:
        对一组结构面 Discontinuity 对象进行基于法向的多层级聚类:
            1) 计算 RMSE 与迹长的质量优先度 q;
            2) 选取高质量样本使用 DBSCAN 获得初始簇中心;
            3) 合并人工给定的 dip/dir 初始簇;
            4) 使用带样本质量权重的 Fuzzy C-Means 在单位球面上细化聚类;
            5) 根据最大隶属度与阈值区分簇与噪声, 并计算簇置信度.

    实现思路(概要):
        - 提取每个结构面的法向、拟合 RMSE、迹长, 将法向统一到上半球并归一化;
        - 对 RMSE 与迹长分别归一化, 按配置的权重计算质量优先度 q, 并排序;
        - 前 first_layer_ratio 部分作为第一层, 使用基于角度距离的 DBSCAN 聚类得到初始簇中心;
        - 将人工 dip/dir 转成单位法向, 如与已有中心夹角大于 eps, 则加入为新簇中心;
        - 按 q 将样本分层, 使用两种可选的质量权重模式, 再结合低质量样本的 RMSE 做二级调权, 得到综合权重 α_i;
        - 以初始簇中心和所有样本法向为输入, 运行带权 FCM, 在球面上迭代更新隶属度与簇中心;
        - 对过近簇中心进行合并, 并再次短迭代微调;
        - 根据最大隶属度 μ_max 与阈值判定噪声及主簇, 计算每个簇的置信度;
        - 将 cluster_id, cluster_membership, cluster_quality_weight, cluster_quality_q, cluster_is_noise,
          cluster_confidence 等结果写回各个 Discontinuity 实例的属性。

    输入参数:
        discontinuities: List[Discontinuity]
            待聚类的结构面对象列表, 需已包含 plane.normal, plane.rmse, trace_length 等信息。
        config: Optional[ClusterConfig]
            聚类参数配置; 若为 None, 则使用 ClusterConfig 的默认参数。
        manual_dip_dirs: Optional[List[Tuple[float, float]]]
            人工给定的结构面 dip/dir 列表, 每项为 (dip_deg, dip_dir_deg), 用于作为额外初始簇中心。

    输出:
        cluster_infos: List[PlaneClusterInfo]
            聚类后每个簇的统计信息列表, 可用于后续可视化与分析。
    """
    logger = LoggerManager().GetLogger("PlaneCluster")
    if config is None:
        config = ClusterConfig()

    if not discontinuities:
        logger.warning("ClusterDiscontinuities: 输入的结构面列表为空, 不执行聚类.")
        return []

    # ----------------------
    # Step 0: 提取有效结构面与基础数据
    # ----------------------
    with Timer(logger, "Step0_ExtractData"):
        normals, rmses, lengths, valid_indices = _ExtractDiscontinuityFeatures(discontinuities, logger)
        n_valid = len(valid_indices)
        if n_valid == 0:
            logger.warning("ClusterDiscontinuities: 没有找到包含有效 plane.normal 与 trace_length 的结构面.")
            return []

        # 将法向统一到上半球并归一化
        normals = _NormalizeNormalsToUpperHemisphere(normals)

    # ----------------------
    # Step 1: 计算质量优先度 q
    # ----------------------
    with Timer(logger, "Step1_ComputeQualityQ"):
        q_values, rmse_norm, length_norm = _ComputeQualityScores(
            rmses,
            lengths,
            config.rmse_weight,
            config.length_weight
        )
        # 将 q 写回 Discontinuity
        for idx_arr, disc_idx in enumerate(valid_indices):
            disc = discontinuities[disc_idx]
            setattr(disc, "cluster_quality_q", float(q_values[idx_arr]))

    # ----------------------
    # Step 2: 第一层样本 + DBSCAN 初始簇
    # ----------------------
    with Timer(logger, "Step2_DBSCAN_Init"):
        first_layer_mask, cluster_labels_first, cluster_centers, is_manual_seed = _RunDbscanOnNormals(
            normals,
            q_values,
            config,
            logger
        )

        # 将第一层的 DBSCAN 簇结果写入 plane_cluster_id (仅作为参考/调试)
        for local_idx, disc_idx in enumerate(valid_indices):
            disc = discontinuities[disc_idx]
            setattr(disc, "plane_cluster_id", int(cluster_labels_first[local_idx]))

        # 若人工给定 dip/dir, 加入额外簇中心
        if manual_dip_dirs:
            cluster_centers, is_manual_seed = _AppendManualSeeds(
                cluster_centers,
                is_manual_seed,
                manual_dip_dirs,
                config.dbscan_eps_deg,
                logger
            )

        n_clusters_init = cluster_centers.shape[0]
        if n_clusters_init == 0:
            logger.warning("ClusterDiscontinuities: DBSCAN + 人工初始簇总数为 0, 将不执行 FCM, 所有结构面标记为噪声.")
            _MarkAllAsNoise(discontinuities, valid_indices)
            return []

    # ----------------------
    # Step 3: 计算样本综合质量权重 α_i
    # ----------------------
    with Timer(logger, "Step3_ComputeSampleWeights"):
        alpha = _ComputeSampleWeights(
            q_values=q_values,
            rmse_norm=rmse_norm,
            first_layer_mask=first_layer_mask,
            config=config
        )
        # 写回 Discontinuity 的质量权重
        for local_idx, disc_idx in enumerate(valid_indices):
            disc = discontinuities[disc_idx]
            setattr(disc, "cluster_quality_weight", float(alpha[local_idx]))

    # ----------------------
    # Step 4: 带权 FCM 聚类 (含簇合并)
    # ----------------------
    with Timer(logger, "Step4_WeightedFCM"):
        centers_fcm, membership = _RunWeightedFcmOnSphere(
            normals=normals,
            alpha=alpha,
            centers_init=cluster_centers,
            config=config,
            logger=logger
        )

        centers_fcm, membership, is_manual_seed_final = _MergeCloseClusters(
            normals=normals,
            alpha=alpha,
            centers=centers_fcm,
            membership=membership,
            is_manual_seed=is_manual_seed,
            config=config,
            logger=logger
        )

    # ----------------------
    # Step 5: 根据隶属度分配簇/噪声, 计算簇置信度
    # ----------------------
    with Timer(logger, "Step5_AssignLabelsAndConfidence"):
        cluster_infos, labels_final, membership_max, cluster_conf = _AssignLabelsAndConfidence(
            normals=normals,
            alpha=alpha,
            centers=centers_fcm,
            membership=membership,
            is_manual_seed=is_manual_seed_final,
            config=config,
            logger=logger
        )

        # 将结果写回 Discontinuity 属性
        for local_idx, disc_idx in enumerate(valid_indices):
            disc = discontinuities[disc_idx]
            cid = int(labels_final[local_idx])
            mu_max = float(membership_max[local_idx])
            is_noise = (cid < 0)
            conf_val = 0.0
            if not is_noise and cid in cluster_conf:
                conf_val = float(cluster_conf[cid])

            setattr(disc, "cluster_id", cid)
            setattr(disc, "cluster_membership", mu_max)
            setattr(disc, "cluster_is_noise", bool(is_noise))
            setattr(disc, "cluster_confidence", conf_val)

    # ----------------------
    # Step 6: 可视化 (可选)
    # ----------------------
    if config.enable_plot:
        with Timer(logger, "Step6_PlotClusters"):
            try:
                _PlotClustersOnStereonet(
                    normals=normals,
                    labels=labels_final,
                    centers=centers_fcm
                )
            except Exception as e:
                logger.warning(f"PlaneCluster: 绘图失败: {e}")

    return cluster_infos


# ======================================================================
# 内部工具函数实现
# ======================================================================

def _ExtractDiscontinuityFeatures(
        discontinuities: List[Discontinuity],
        logger
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """
    功能简介:
        从 Discontinuity 列表中提取用于聚类的基础特征:
            - 平面法向向量
            - 拟合 RMSE
            - 迹长(优先使用 Discontinuity.trace_length, 其次 Segment.trace_length)

    实现思路:
        - 遍历所有结构面, 对于 plane.normal/plane.rmse 缺失或非法的对象直接跳过;
        - 若 trace_length <= 0, 则尝试从 segments 中选取最大 trace_length 作为替代;
        - 收集有效结构面的法向、RMSE、迹长, 并记录其在原列表中的索引。

    输入参数:
        discontinuities: List[Discontinuity]
            结构面对象列表.
        logger:
            日志记录器, 用于输出警告信息.

    输出:
        normals: np.ndarray
            形状 (N, 3), 有效结构面的原始法向(未归一化/未上半球统一).
        rmses: np.ndarray
            形状 (N,), 对应的 RMSE 数组.
        lengths: np.ndarray
            形状 (N,), 对应的迹长数组.
        valid_indices: List[int]
            有效结构面在原列表中的索引列表.
    """
    normals_list = []
    rmses_list = []
    lengths_list = []
    valid_indices: List[int] = []

    for idx, disc in enumerate(discontinuities):
        plane = getattr(disc, "plane", None)
        if plane is None or plane.normal is None:
            continue

        nx, ny, nz = plane.normal
        vec = np.array([nx, ny, nz], dtype=np.float64)
        if not np.all(np.isfinite(vec)):
            continue
        if np.linalg.norm(vec) < 1e-12:
            continue

        rmse = float(getattr(plane, "rmse", 0.0))
        # 迹长优先使用 Discontinuity.trace_length
        trace_len = float(getattr(disc, "trace_length", 0.0))
        if trace_len <= 0.0:
            # 从 segments 中尝试获得最大 trace_length
            segments = getattr(disc, "segments", [])
            max_seg_len = 0.0
            for seg in segments:
                seg_len = float(getattr(seg, "trace_length", 0.0))
                if seg_len > max_seg_len:
                    max_seg_len = seg_len
            trace_len = max_seg_len

        normals_list.append(vec)
        rmses_list.append(rmse)
        lengths_list.append(trace_len)
        valid_indices.append(idx)

    if not normals_list:
        logger.warning("PlaneCluster: 未找到包含有效法向的结构面.")
        return (np.zeros((0, 3), dtype=np.float64),
                np.zeros((0,), dtype=np.float64),
                np.zeros((0,), dtype=np.float64),
                [])

    normals = np.vstack(normals_list).astype(np.float64)
    rmses = np.asarray(rmses_list, dtype=np.float64)
    lengths = np.asarray(lengths_list, dtype=np.float64)
    return normals, rmses, lengths, valid_indices


def _NormalizeNormalsToUpperHemisphere(normals: np.ndarray) -> np.ndarray:
    """
    功能简介:
        将所有法向向量归一化为单位向量, 并统一到上半球 (nz >= 0).

    实现思路:
        - 对每个法向除以其模长, 得到单位法向;
        - 若单位法向的 z 分量 < 0, 则取其相反方向, 保证 nz >= 0;
        - 返回处理后的单位法向数组。

    输入参数:
        normals: np.ndarray
            形状 (N, 3), 原始法向向量数组.

    输出:
        normals_unit: np.ndarray
            形状 (N, 3), 统一到上半球的单位法向数组.
    """
    n = normals.shape[0]
    normals_unit = np.zeros_like(normals, dtype=np.float64)
    for i in range(n):
        v = normals[i]
        norm = float(np.linalg.norm(v))
        if norm < 1e-12:
            continue
        v_unit = v / norm
        if v_unit[2] < 0.0:
            v_unit = -v_unit
        normals_unit[i] = v_unit
    return normals_unit


def _ComputeQualityScores(
        rmses: np.ndarray,
        lengths: np.ndarray,
        rmse_weight: float,
        length_weight: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    功能简介:
        基于 RMSE 与迹长计算质量优先度 q, 同时返回归一化后的 RMSEnorm 与 lengthnorm。

    实现思路:
        - 对 RMSE 与迹长分别做线性归一化;
        - 对 RMSE 使用 (1 - RMSEnorm), 对迹长使用 lengthnorm, 按给定权重做加权平均;
        - 得到 q 值, q 越大代表结构面质量越高 (RMSE 小且迹长长)。

    输入参数:
        rmses: np.ndarray
            形状 (N,), 每个结构面的拟合 RMSE.
        lengths: np.ndarray
            形状 (N,), 每个结构面的迹长.
        rmse_weight: float
            在 q 计算中的 RMSE 项权重.
        length_weight: float
            在 q 计算中的迹长项权重.

    输出:
        q_values: np.ndarray
            形状 (N,), 每个结构面的质量优先度 q.
        rmse_norm: np.ndarray
            形状 (N,), 归一化后的 RMSEnorm.
        length_norm: np.ndarray
            形状 (N,), 归一化后的 lengthnorm.
    """
    eps = 1e-12
    rmses = np.asarray(rmses, dtype=np.float64)
    lengths = np.asarray(lengths, dtype=np.float64)

    # RMSE 归一化
    rmse_min = float(np.nanmin(rmses))
    rmse_max = float(np.nanmax(rmses))
    if rmse_max - rmse_min < eps:
        rmse_norm = np.zeros_like(rmses, dtype=np.float64)
    else:
        rmse_norm = (rmses - rmse_min) / (rmse_max - rmse_min + eps)

    # 迹长归一化
    length_min = float(np.nanmin(lengths))
    length_max = float(np.nanmax(lengths))
    if length_max - length_min < eps:
        length_norm = np.zeros_like(lengths, dtype=np.float64)
    else:
        length_norm = (lengths - length_min) / (length_max - length_min + eps)

    # 计算 q
    w_r = float(rmse_weight)
    w_l = float(length_weight)
    denom = w_r + w_l
    if denom < eps:
        denom = 1.0
    q_values = (w_r * (1.0 - rmse_norm) + w_l * length_norm) / denom
    return q_values, rmse_norm, length_norm


def _AngularDistanceRad(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    功能简介:
        计算两个单位向量之间的夹角距离 (弧度).

    实现思路:
        - 使用点积计算 cos(theta) = |v1·v2|;
        - 对 cos 值裁剪到 [-1, 1] 范围内, 再做 arccos;
        - 返回弧度制的夹角。

    输入参数:
        v1: np.ndarray
            形状 (3,), 单位向量 1.
        v2: np.ndarray
            形状 (3,), 单位向量 2.

    输出:
        theta: float
            两向量夹角, 单位为弧度.
    """
    dot = float(np.dot(v1, v2))
    dot = max(min(dot, 1.0), -1.0)
    # 统一采用绝对值, 保证 n/-n 一致
    dot = abs(dot)
    return math.acos(dot)


def _RunDbscanOnNormals(
        normals: np.ndarray,
        q_values: np.ndarray,
        config: ClusterConfig,
        logger
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    功能简介:
        在法向单位球面上对高质量结构面样本执行 DBSCAN 聚类, 得到初始簇中心。

    实现思路:
        1) 按 q 降序排序, 选取前 first_layer_ratio 比例样本作为第一层;
        2) 在第一层样本上使用自实现 DBSCAN (基于角度距离) 聚类;
        3) 对每个簇计算中心法向 (归一化向量和), 并统计其簇标签;
        4) 返回:
            - first_layer_mask: 全体样本中的布尔标记数组
            - labels_first: 全体样本的第一层 DBSCAN 标签 (非第一层样本统一为 -2)
            - cluster_centers: 初始簇中心数组 (K,3)
            - is_manual_seed: 每个初始簇是否为人工种子 (这里全部 False, 后续再追加)

    输入参数:
        normals: np.ndarray
            形状 (N,3), 所有结构面的单位法向 (已统一上半球).
        q_values: np.ndarray
            形状 (N,), 对应的质量优先度 q.
        config: ClusterConfig
            聚类参数配置.
        logger:
            日志记录器.

    输出:
        first_layer_mask: np.ndarray
            形状 (N,), 布尔数组, True 表示属于第一层高质量样本.
        labels_first: np.ndarray
            形状 (N,), 第一层样本的 DBSCAN 标签, 非第一层样本为 -2.
        cluster_centers: np.ndarray
            形状 (K,3), 初始簇中心法向数组.
        is_manual_seed: np.ndarray
            形状 (K,), 布尔数组, 初始时全为 False, 供后续人工簇扩展使用.
    """
    N = normals.shape[0]
    if N == 0:
        return (np.zeros((0,), dtype=bool),
                np.zeros((0,), dtype=int),
                np.zeros((0, 3), dtype=np.float64),
                np.zeros((0,), dtype=bool))

    # 1) 选取第一层样本
    order_desc = np.argsort(-q_values)  # q 值降序
    first_count = max(1, int(round(config.first_layer_ratio * N)))
    first_count = min(first_count, N)
    first_indices = order_desc[:first_count]
    first_layer_mask = np.zeros((N,), dtype=bool)
    first_layer_mask[first_indices] = True

    # DBSCAN 参数
    eps_rad = math.radians(config.dbscan_eps_deg)
    min_samples = max(3, int(round(config.dbscan_min_samples_ratio * first_count)))
    if min_samples > first_count:
        min_samples = max(3, first_count)

    labels_first = np.full((N,), -2, dtype=int)  # 非第一层标记为 -2
    if first_count < min_samples:
        # 第一层样本不足以执行 DBSCAN
        logger.warning("PlaneCluster: 第一层样本数不足以执行 DBSCAN, 将跳过初始聚类.")
        return first_layer_mask, labels_first, np.zeros((0, 3), dtype=np.float64), np.zeros((0,), dtype=bool)

    # 2) 在第一层做自实现 DBSCAN
    # 映射: first_idx_array -> 原始 idx
    normals_first = normals[first_indices]

    # 预计算邻居列表
    neighbors = []
    for i in range(first_count):
        ni = normals_first[i]
        nbrs_i = []
        for j in range(first_count):
            nj = normals_first[j]
            theta = _AngularDistanceRad(ni, nj)
            if theta <= eps_rad:
                nbrs_i.append(j)
        neighbors.append(nbrs_i)

    visited = np.zeros((first_count,), dtype=bool)
    labels_local = np.full((first_count,), -1, dtype=int)  # -1 表示噪声
    cluster_id = 0

    for i in range(first_count):
        if visited[i]:
            continue
        visited[i] = True
        nbrs = neighbors[i]
        if len(nbrs) < min_samples:
            labels_local[i] = -1  # 噪声
            continue

        # 创建新簇
        labels_local[i] = cluster_id
        queue = deque(nbrs)
        while queue:
            j = queue.popleft()
            if not visited[j]:
                visited[j] = True
                nbrs_j = neighbors[j]
                if len(nbrs_j) >= min_samples:
                    for nb in nbrs_j:
                        if nb not in queue:
                            queue.append(nb)
            if labels_local[j] == -1:
                labels_local[j] = cluster_id
        cluster_id += 1

    # 将第一层标签映射回全体样本
    labels_first[first_indices] = labels_local

    # 3) 计算簇中心
    unique_labels = sorted([cid for cid in np.unique(labels_local) if cid >= 0])
    centers = []
    for cid in unique_labels:
        mask_c = (labels_local == cid)
        if not np.any(mask_c):
            continue
        vec_sum = normals_first[mask_c].sum(axis=0)
        norm = float(np.linalg.norm(vec_sum))
        if norm < 1e-12:
            continue
        center = vec_sum / norm
        centers.append(center)

    if not centers:
        logger.warning("PlaneCluster: DBSCAN 未找到任何簇(均为噪声).")
        return first_layer_mask, labels_first, np.zeros((0, 3), dtype=np.float64), np.zeros((0,), dtype=bool)

    cluster_centers = np.vstack(centers).astype(np.float64)
    is_manual_seed = np.zeros((cluster_centers.shape[0],), dtype=bool)
    logger.info(f"PlaneCluster: DBSCAN 初始得到 {cluster_centers.shape[0]} 个簇中心.")
    return first_layer_mask, labels_first, cluster_centers, is_manual_seed


def _ConvertDipDirToNormal(dip_deg: float, dip_dir_deg: float) -> np.ndarray:
    """
    功能简介:
        将结构面的 dip(倾角) 与 dip_direction(倾向) 转换为单位法向向量,
        并统一到上半球 (nz >= 0).

    实现思路:
        - 假定坐标系: y 轴为北, x 轴为东, z 轴向上;
        - Dip_direction 为从北顺时针的方位角(度), Dip 为与水平面的夹角(度);
        - 可以用常用地质公式构造法向:
            n_x = sin(dip) * sin(dip_dir)
            n_y = sin(dip) * cos(dip_dir)
            n_z = cos(dip)
        - 归一化后若 nz<0 则取反, 保证上半球.

    输入参数:
        dip_deg: float
            倾角(度).
        dip_dir_deg: float
            倾向(度, 从北顺时针).

    输出:
        normal: np.ndarray
            形状 (3,), 上半球单位法向.
    """
    dip_rad = math.radians(dip_deg)
    dir_rad = math.radians(dip_dir_deg)
    nx = math.sin(dip_rad) * math.sin(dir_rad)
    ny = math.sin(dip_rad) * math.cos(dir_rad)
    nz = math.cos(dip_rad)
    v = np.array([nx, ny, nz], dtype=np.float64)
    norm = float(np.linalg.norm(v))
    if norm < 1e-12:
        v = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        v = v / norm
    if v[2] < 0.0:
        v = -v
    return v


def _AppendManualSeeds(
        cluster_centers: np.ndarray,
        is_manual_seed: np.ndarray,
        manual_dip_dirs: List[Tuple[float, float]],
        eps_deg: float,
        logger
) -> Tuple[np.ndarray, np.ndarray]:
    """
    功能简介:
        将人工提供的 dip/dir 转换为簇中心法向, 并在与现有中心夹角大于 eps 时追加为新的簇中心。

    实现思路:
        - 将每个 (dip, dip_dir) 转为单位法向 normal_man;
        - 对每个 normal_man 计算其与所有现有中心的最小夹角;
        - 若最小夹角大于 eps, 则将该 normal_man 追加到 cluster_centers;
        - 同时在 is_manual_seed 中记录该簇为人工种子.

    输入参数:
        cluster_centers: np.ndarray
            形状 (K,3), 当前簇中心.
        is_manual_seed: np.ndarray
            形状 (K,), 当前簇是否为人工种子标记.
        manual_dip_dirs: List[Tuple[float, float]]
            人工 dip/dir 列表.
        eps_deg: float
            角度阈值(度), 小于该阈值视为“重复簇”不新增.
        logger:
            日志记录器.

    输出:
        new_centers: np.ndarray
            更新后的所有簇中心.
        new_is_manual_seed: np.ndarray
            更新后的人工簇标记数组.
    """
    eps_rad = math.radians(eps_deg)

    centers = cluster_centers.copy()
    manual_flags = is_manual_seed.copy()

    for dip_deg, dip_dir_deg in manual_dip_dirs:
        n_man = _ConvertDipDirToNormal(dip_deg, dip_dir_deg)
        if centers.shape[0] > 0:
            angles = np.array(
                [_AngularDistanceRad(n_man, c) for c in centers],
                dtype=np.float64
            )
            min_angle = float(np.min(angles))
        else:
            min_angle = float("inf")

        if min_angle > eps_rad:
            # 追加新簇中心
            centers = np.vstack([centers, n_man]) if centers.size else n_man.reshape(1, 3)
            manual_flags = np.concatenate([manual_flags, np.array([True], dtype=bool)])
            logger.info(f"PlaneCluster: 追加人工簇中心 dip={dip_deg}, dip_dir={dip_dir_deg}, "
                        f"与已有中心最小夹角={math.degrees(min_angle):.2f}°")
        else:
            logger.info(f"PlaneCluster: 忽略人工簇中心 dip={dip_deg}, dip_dir={dip_dir_deg}, "
                        f"因与已有中心夹角过小({math.degrees(min_angle):.2f}°)")

    return centers, manual_flags


def _ComputeSampleWeights(
        q_values: np.ndarray,
        rmse_norm: np.ndarray,
        first_layer_mask: np.ndarray,
        config: ClusterConfig
) -> np.ndarray:
    """
    功能简介:
        根据质量优先度 q 与 RMSEnorm, 结合两种模式(piecewise/continuous),
        计算 FCM 中使用的综合样本质量权重 α_i。

    实现思路:
        - 第一层(高质量)样本基础权重 s_i = 1.0;
        - 剩余样本在其内部按 q 降序计算百分位 p_remain:
            * piecewise 模式: 使用分段常数映射 p_remain -> s_i;
            * continuous 模式: 对全体样本 q 做缩放, 使用 s_i = s_min + (1-s_min)*q_scaled^γ;
        - 对全体样本再识别全局最低 low_quality_ratio 部分, 对这些样本附加二级权重
          β_i = exp(-λ*RMSEnorm_i), 其余 β_i = 1;
        - 综合权重 α_i = s_i * β_i。

    输入参数:
        q_values: np.ndarray
            形状 (N,), 每个样本的质量优先度 q.
        rmse_norm: np.ndarray
            形状 (N,), 归一化 RMSEnorm.
        first_layer_mask: np.ndarray
            形状 (N,), 布尔数组, 第一层样本为 True.
        config: ClusterConfig
            配置参数.

    输出:
        alpha: np.ndarray
            形状 (N,), 每个样本的综合质量权重 α_i.
    """
    N = q_values.shape[0]
    s = np.ones((N,), dtype=np.float64)

    # 对剩余样本根据配置计算基础权重 s_i
    remain_mask = ~first_layer_mask
    remain_indices = np.where(remain_mask)[0]
    n_remain = remain_indices.shape[0]
    if n_remain > 0:
        if config.quality_mode.lower() == "piecewise":
            # 在剩余样本内按 q 降序排序, 计算百分位
            q_remain = q_values[remain_indices]
            order_desc = np.argsort(-q_remain)
            ranks = np.empty_like(order_desc)
            ranks[order_desc] = np.arange(n_remain)
            p_remain = (ranks.astype(np.float64) + 1.0) / float(n_remain)
            # 映射百分位到权重
            s_remain = np.zeros_like(p_remain, dtype=np.float64)
            for i, p in enumerate(p_remain):
                if p < 0.2:
                    s_remain[i] = 0.8
                elif p < 0.4:
                    s_remain[i] = 0.6
                elif p < 0.6:
                    s_remain[i] = 0.4
                elif p < 0.8:
                    s_remain[i] = 0.2
                else:
                    s_remain[i] = config.quality_min_weight
            s[remain_indices] = s_remain
        else:
            # continuous 模式: 对全体 q 做缩放到 [0,1]
            q_min = float(np.min(q_values))
            q_max = float(np.max(q_values))
            denom = q_max - q_min if (q_max - q_min) > 1e-12 else 1.0
            q_scaled = (q_values - q_min) / denom
            s = config.quality_min_weight + (1.0 - config.quality_min_weight) * np.power(
                q_scaled, config.quality_gamma
            )
            # 第一层重新强制为 1.0
            s[first_layer_mask] = 1.0

    # 二级: 针对全体最低质量 low_quality_ratio 部分做 RMSE 调权
    N_low = max(0, int(round(config.low_quality_ratio * N)))
    beta = np.ones((N,), dtype=np.float64)
    if N_low > 0:
        order_asc = np.argsort(q_values)  # q 从小到大
        low_indices = order_asc[:N_low]
        beta_low = np.exp(-config.rmse_gamma * rmse_norm[low_indices])
        beta[low_indices] = beta_low

    alpha = s * beta
    return alpha


def _RunWeightedFcmOnSphere(
        normals: np.ndarray,
        alpha: np.ndarray,
        centers_init: np.ndarray,
        config: ClusterConfig,
        logger
) -> Tuple[np.ndarray, np.ndarray]:
    """
    功能简介:
        在单位球面上运行带样本权重的 Fuzzy C-Means 聚类, 使用角度距离作为度量,
        并返回最终簇中心与样本隶属度矩阵。

    实现思路:
        - 使用给定的初始簇中心 centers_init;
        - 迭代过程:
            1) 根据当前中心计算每个样本到各簇的角度距离 d_ik;
            2) 根据标准 FCM 更新公式计算隶属度 μ_ik (处理 d_ik=0 的情况);
            3) 使用 α_i 与 μ_ik^m 对法向加权求和, 更新簇中心并归一化到单位球面;
            4) 若所有簇中心变化角度的最大值 < fcm_tol_deg, 则停止迭代;
        - 若某个簇在迭代中出现数值退化 (中心向量模长极小), 则保持之前的中心不变, 并在日志中提醒。

    输入参数:
        normals: np.ndarray
            形状 (N,3), 所有结构面的单位法向(已统一上半球).
        alpha: np.ndarray
            形状 (N,), 每个样本的综合质量权重 α_i.
        centers_init: np.ndarray
            形状 (K,3), 初始簇中心法向数组.
        config: ClusterConfig
            FCM 参数配置.
        logger:
            日志记录器.

    输出:
        centers: np.ndarray
            形状 (K,3), 收敛后的簇中心法向数组.
        membership: np.ndarray
            形状 (N,K), 收敛后的隶属度矩阵 μ_ik.
    """
    normals = np.asarray(normals, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    centers = np.asarray(centers_init, dtype=np.float64)

    N = normals.shape[0]
    K = centers.shape[0]
    if N == 0 or K == 0:
        return centers, np.zeros((N, K), dtype=np.float64)

    m = config.fcm_m
    max_iter = config.fcm_max_iter
    tol_rad = math.radians(config.fcm_tol_deg)
    eps = 1e-12

    # 初始化隶属度矩阵 (可用距离公式直接初始化)
    membership = np.zeros((N, K), dtype=np.float64)

    for it in range(max_iter):
        # 1) 计算距离矩阵 d_ik (弧度)
        d = np.zeros((N, K), dtype=np.float64)
        for i in range(N):
            for k in range(K):
                d_ik = _AngularDistanceRad(normals[i], centers[k])
                # 避免完全 0 的情况影响换算
                d[i, k] = max(d_ik, eps)

        # 2) 更新隶属度 μ_ik
        for i in range(N):
            # 若某个簇距离近似为 0, 则将该簇隶属度置为 1, 其他簇为 0
            zero_mask = d[i, :] <= eps
            if np.any(zero_mask):
                membership[i, :] = 0.0
                membership[i, zero_mask] = 1.0 / float(np.sum(zero_mask))
                continue

            # 标准 FCM 更新公式
            for k in range(K):
                denom = 0.0
                for j in range(K):
                    ratio = d[i, k] / d[i, j]
                    denom += math.pow(ratio, 2.0 / (m - 1.0))
                membership[i, k] = 1.0 / denom

        # 3) 根据 α_i 与 μ_ik^m 更新簇中心
        centers_old = centers.copy()
        for k in range(K):
            # 计算该簇的加权向量和
            weights = alpha * np.power(membership[:, k], m)
            w_sum = float(np.sum(weights))
            if w_sum < eps:
                # 若该簇几乎没有支持样本, 则不更新中心
                logger.warning(f"PlaneCluster: 第 {it} 次迭代时簇 {k} 权重和过小, 保持原中心不变.")
                continue
            vec_sum = np.sum(normals * weights[:, None], axis=0)
            norm_vec = float(np.linalg.norm(vec_sum))
            if norm_vec < eps:
                logger.warning(f"PlaneCluster: 第 {it} 次迭代时簇 {k} 更新中心向量模长过小, 保持原中心不变.")
                continue
            centers[k] = vec_sum / norm_vec

        # 4) 检查收敛: 所有簇中心最大变化夹角
        max_angle = 0.0
        for k in range(K):
            angle_k = _AngularDistanceRad(centers_old[k], centers[k])
            if angle_k > max_angle:
                max_angle = angle_k

        if max_angle < tol_rad:
            logger.info(f"PlaneCluster: FCM 在第 {it} 次迭代时收敛, 最大中心变化角度={math.degrees(max_angle):.4f}°")
            break

    return centers, membership


def _MergeCloseClusters(
        normals: np.ndarray,
        alpha: np.ndarray,
        centers: np.ndarray,
        membership: np.ndarray,
        is_manual_seed: np.ndarray,
        config: ClusterConfig,
        logger
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    功能简介:
        对 FCM 得到的簇中心进行后处理, 若两个簇中心之间的夹角小于 merge_angle_eps_deg,
        则将其合并为一个簇, 并适当调整隶属度矩阵, 可重复多轮。

    实现思路:
        - 循环最多 max_merge_loops 轮:
            1) 计算所有簇中心的两两夹角矩阵;
            2) 找出夹角最小的一对簇 (p, q);
            3) 若最小夹角 < merge_angle_eps_deg, 则:
                * 新簇中心 = 对 p、q 使用 α_i*(μ_ip^m + μ_iq^m) 加权的向量和并归一化;
                * 对隶属度矩阵: 新 μ_i = μ_ip + μ_iq, 删除旧的 q 列, 再对每个样本归一化;
                * is_manual_seed[p] = is_manual_seed[p] or is_manual_seed[q];
            4) 否则跳出循环;
        - 合并后可选再做少量 FCM 微调 (本实现中为简化只进行一次短迭代或不再迭代)。

    输入参数:
        normals: np.ndarray
            形状 (N,3), 单位法向数组.
        alpha: np.ndarray
            形状 (N,), 样本综合权重.
        centers: np.ndarray
            形状 (K,3), FCM 结果的簇中心.
        membership: np.ndarray
            形状 (N,K), FCM 结果的隶属度矩阵.
        is_manual_seed: np.ndarray
            形状 (K,), 每个簇是否来源于人工种子.
        config: ClusterConfig
            聚类配置.
        logger:
            日志记录器.

    输出:
        centers_new: np.ndarray
            合并后的簇中心数组.
        membership_new: np.ndarray
            合并后的隶属度矩阵.
        is_manual_seed_new: np.ndarray
            合并后的人工种子标记数组.
    """
    centers_cur = centers.copy()
    membership_cur = membership.copy()
    manual_flags = is_manual_seed.copy()

    N, K = membership_cur.shape
    if K <= 1:
        return centers_cur, membership_cur, manual_flags

    eps_rad = math.radians(config.merge_angle_eps_deg)
    m = config.fcm_m
    eps = 1e-12

    for loop in range(config.max_merge_loops):
        K = centers_cur.shape[0]
        if K <= 1:
            break

        # 计算两两夹角
        angle_mat = np.full((K, K), np.inf, dtype=np.float64)
        for p in range(K):
            for q in range(p + 1, K):
                angle = _AngularDistanceRad(centers_cur[p], centers_cur[q])
                angle_mat[p, q] = angle
                angle_mat[q, p] = angle

        # 找到最小夹角
        min_angle = float(np.min(angle_mat))
        if min_angle >= eps_rad:
            logger.info(f"PlaneCluster: 簇合并停止, 最小中心夹角={math.degrees(min_angle):.3f}° >= 阈值.")
            break

        # 找到需要合并的一对 (p, q)
        idx_flat = int(np.argmin(angle_mat))
        p = idx_flat // K
        q = idx_flat % K
        if p == q:
            break

        logger.info(f"PlaneCluster: 合并簇 {p} 与簇 {q}, 中心夹角={math.degrees(min_angle):.3f}°")

        # 计算新簇中心: 对 p, q 的加权向量和
        weights_p = alpha * np.power(membership_cur[:, p], m)
        weights_q = alpha * np.power(membership_cur[:, q], m)
        weights_sum = weights_p + weights_q
        w_total = float(np.sum(weights_sum))
        if w_total < eps:
            # 权重过小, 直接用 centers_cur[p] 作为新中心
            new_center = centers_cur[p].copy()
        else:
            vec_sum = np.sum(normals * weights_sum[:, None], axis=0)
            norm_vec = float(np.linalg.norm(vec_sum))
            if norm_vec < eps:
                new_center = centers_cur[p].copy()
            else:
                new_center = vec_sum / norm_vec

        # 新隶属度: μ_new = μ_p + μ_q
        mu_new_col = membership_cur[:, p] + membership_cur[:, q]

        # 构造新的中心与隶属度矩阵
        centers_list = []
        manual_list = []
        membership_list = []

        for k in range(K):
            if k == p:
                continue
            if k == q:
                continue
            centers_list.append(centers_cur[k])
            manual_list.append(manual_flags[k])
            membership_list.append(membership_cur[:, k])

        centers_list.append(new_center)
        manual_list.append(manual_flags[p] or manual_flags[q])
        membership_list.append(mu_new_col)

        centers_cur = np.vstack(centers_list).astype(np.float64)
        manual_flags = np.array(manual_list, dtype=bool)
        membership_cur = np.vstack(membership_list).T  # (N, K_new)

        # 对每个样本重新归一化隶属度
        for i in range(N):
            s_row = float(np.sum(membership_cur[i, :]))
            if s_row < eps:
                # 若某行全为 0, 则平均分配
                membership_cur[i, :] = 1.0 / membership_cur.shape[1]
            else:
                membership_cur[i, :] /= s_row

    return centers_cur, membership_cur, manual_flags


def _AssignLabelsAndConfidence(
        normals: np.ndarray,
        alpha: np.ndarray,
        centers: np.ndarray,
        membership: np.ndarray,
        is_manual_seed: np.ndarray,
        config: ClusterConfig,
        logger
) -> Tuple[List[PlaneClusterInfo], np.ndarray, np.ndarray, Dict[int, float]]:
    """
    功能简介:
        根据最终隶属度矩阵与簇中心, 为每个样本分配聚类标签/噪声, 并计算每个簇的整体置信度。

    实现思路:
        - 对每个样本 i:
            * 计算 μ_max_i = max_k μ_ik, k_best = argmax μ_ik;
            * 若 μ_max_i < noise_membership_threshold, 则 label_i = -1 (噪声);
            * 否则 label_i = k_best;
        - 对每个簇 k:
            * 收集所有 label_i = k 的样本集合 S_k;
            * 计算簇置信度 conf_k = sum(α_i * μ_max_i) / sum(α_i), i ∈ S_k;
        - 构造 PlaneClusterInfo 列表并返回, 同时返回 labels、μ_max 以及簇置信度字典。

    输入参数:
        normals: np.ndarray
            形状 (N,3), 单位法向数组 (目前不直接使用, 可扩展用于更多置信度指标).
        alpha: np.ndarray
            形状 (N,), 样本综合权重.
        centers: np.ndarray
            形状 (K,3), 簇中心法向.
        membership: np.ndarray
            形状 (N,K), 隶属度矩阵.
        is_manual_seed: np.ndarray
            形状 (K,), 每簇是否来自人工种子.
        config: ClusterConfig
            聚类配置.
        logger:
            日志记录器.

    输出:
        cluster_infos: List[PlaneClusterInfo]
            每个簇的统计信息列表.
        labels: np.ndarray
            形状 (N,), 每个样本的聚类标签, -1 表示噪声.
        membership_max: np.ndarray
            形状 (N,), 每个样本的最大隶属度 μ_max_i.
        cluster_confidence: Dict[int, float]
            每个簇的置信度, 键为簇 id, 值为 conf_k.
    """
    N, K = membership.shape
    labels = np.full((N,), -1, dtype=int)
    membership_max = np.zeros((N,), dtype=np.float64)

    # 样本级标签
    for i in range(N):
        if K == 0:
            labels[i] = -1
            membership_max[i] = 0.0
            continue
        mu_row = membership[i, :]
        k_best = int(np.argmax(mu_row))
        mu_max = float(mu_row[k_best])
        membership_max[i] = mu_max
        if mu_max < config.noise_membership_threshold:
            labels[i] = -1
        else:
            labels[i] = k_best

    # 簇级置信度
    cluster_conf: Dict[int, float] = {}
    cluster_infos: List[PlaneClusterInfo] = []

    for k in range(K):
        idx_k = np.where(labels == k)[0]
        if idx_k.size == 0:
            cluster_conf[k] = 0.0
            info = PlaneClusterInfo(
                cluster_id=k,
                center_normal=centers[k],
                num_members=0,
                confidence=0.0,
                is_manual_seed=bool(is_manual_seed[k])
            )
            cluster_infos.append(info)
            continue

        # 使用 α_i * μ_max_i 计算加权平均置信度
        alpha_k = alpha[idx_k]
        mu_max_k = membership_max[idx_k]
        num = float(np.sum(alpha_k * mu_max_k))
        den = float(np.sum(alpha_k)) if np.sum(alpha_k) > 1e-12 else 1.0
        conf_k = num / den
        cluster_conf[k] = conf_k

        info = PlaneClusterInfo(
            cluster_id=k,
            center_normal=centers[k],
            num_members=idx_k.size,
            confidence=conf_k,
            is_manual_seed=bool(is_manual_seed[k])
        )
        cluster_infos.append(info)

    logger.info("PlaneCluster: 聚类完成, 有效簇数=%d, 噪声样本数=%d",
                len(cluster_infos), int(np.sum(labels < 0)))
    return cluster_infos, labels, membership_max, cluster_conf


def _MarkAllAsNoise(discontinuities: List[Discontinuity], valid_indices: List[int]) -> None:
    """
    功能简介:
        在无法执行聚类或没有簇中心的情况下, 将所有有效结构面标记为噪声。

    实现思路:
        - 对 valid_indices 中的每个结构面:
            * cluster_id = -1
            * cluster_membership = 0.0
            * cluster_is_noise = True
            * cluster_confidence = 0.0

    输入参数:
        discontinuities: List[Discontinuity]
            结构面列表.
        valid_indices: List[int]
            有效结构面索引列表.

    输出:
        无 (直接修改 Discontinuity 实例属性).
    """
    for disc_idx in valid_indices:
        disc = discontinuities[disc_idx]
        setattr(disc, "cluster_id", -1)
        setattr(disc, "cluster_membership", 0.0)
        setattr(disc, "cluster_is_noise", True)
        setattr(disc, "cluster_confidence", 0.0)


def _PlotClustersOnStereonet(
        normals: np.ndarray,
        labels: np.ndarray,
        centers: np.ndarray
) -> None:
    """
    功能简介:
        在简化的平面投影上对法向聚类结果进行可视化 (示意性),
        便于人工检视聚类效果。

    实现思路:
        - 使用 matplotlib 绘制散点图:
            * 将单位法向投影到平面 (nx, ny) 或 (dip_dir, dip) 空间中;
            * 不同簇使用不同颜色, 噪声点用灰色;
            * 簇中心使用较大标记绘制。

        注意:
            - 本函数仅在 config.enable_plot=True 且 matplotlib 可用时调用;
            - 若 matplotlib 不可用, 将抛出 ImportError, 调用方需捕获处理。

    输入参数:
        normals: np.ndarray
            形状 (N,3), 单位法向数组.
        labels: np.ndarray
            形状 (N,), 聚类标签.
        centers: np.ndarray
            形状 (K,3), 簇中心法向数组.

    输出:
        无, 仅在屏幕上绘制图形.
    """
    import matplotlib.pyplot as plt

    N = normals.shape[0]
    if N == 0:
        return

    # 简单投影: 直接使用 nx, ny 平面
    x = normals[:, 0]
    y = normals[:, 1]
    unique_labels = np.unique(labels)

    plt.figure(figsize=(6, 6))
    for lab in unique_labels:
        mask = labels == lab
        if lab < 0:
            plt.scatter(x[mask], y[mask], c="lightgray", s=10, label="noise")
        else:
            plt.scatter(x[mask], y[mask], s=15, label=f"cluster {lab}")

    # 绘制簇中心
    if centers is not None and centers.shape[0] > 0:
        cx = centers[:, 0]
        cy = centers[:, 1]
        plt.scatter(cx, cy, c="red", s=80, marker="x", label="centers")

    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)
    plt.xlabel("nx")
    plt.ylabel("ny")
    plt.legend()
    plt.title("Plane Normal Clusters (projection to nx-ny)")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.show()
