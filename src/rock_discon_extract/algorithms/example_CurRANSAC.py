# -*- coding: utf-8 -*-
"""
Numba==0.60.0 示例：曲率加权的平面 RANSAC（方案A）
- 目标：在点云 coords(N,3) + curvature(N,) 上，快速找到最优平面
- 核心：评分 S = sum_i w_i * exp(-dist_i^2/(2*sigma^2))
- w_i 由 curvature 映射：w_i = exp(-(kappa_i/kappa0)^2)，并做下限截断 w_min
- 最后对硬内点(dist<th)做加权TLS重估平面参数

运行方式：
    python demo_numba_weighted_ransac_plane.py
"""

import time
import math
from dataclasses import dataclass
from typing import Tuple, Dict, Any

import numpy as np
from numba import njit


# -------------------------
# 参数区（按你工程习惯：脚本内变量传参，不用 argparse）
# -------------------------
SEED = 42

# RANSAC
NUM_ITER = 1000                  # RANSAC 迭代次数（演示用）
SUBSAMPLE_RATIO = 1.0            # 评分时是否对子采样点（1.0 = 全量评分；大N时可用0.5加速）
DISTANCE_THRESHOLD = 0.03        # 硬内点阈值（m）
SIGMA_D = DISTANCE_THRESHOLD / 2 # 软评分的sigma（可调）

# 曲率权重
KAPPA0_PERCENTILE = 40           # kappa0 用曲率分位数（例如40%分位）
W_MIN = 0.05                     # 权重下限，避免数值退化

# 生成测试数据（仅示例）
N_SYN = 20000                    # 点数（示例）
OUTLIER_RATIO = 0.25             # 外点比例
NOISE_STD = 0.005                # 平面噪声（m）


# -------------------------
# 计时器与日志
# -------------------------
class SimpleTimer:
    """[耗时] 简单计时器"""
    def __init__(self, name: str):
        self.name = name
        self.t0 = None

    def __enter__(self):
        self.t0 = time.perf_counter()
        print(f"[log] {self.name} ...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        t1 = time.perf_counter()
        print(f"[耗时] {self.name}: {(t1 - self.t0):.4f} s")


@dataclass
class WeightedRansacConfig:
    """RANSAC 配置"""
    num_iter: int
    distance_threshold: float
    sigma_d: float
    subsample_ratio: float
    w_min: float


# -------------------------
# Numba JIT: 核心数值函数
# -------------------------
@njit(cache=True)
def _NormalizeVec3(x: float, y: float, z: float) -> Tuple[float, float, float, float]:
    """单位化向量并返回范数"""
    n = math.sqrt(x * x + y * y + z * z) + 1e-12
    return x / n, y / n, z / n, n


@njit(cache=True)
def _FitPlaneFrom3Points(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Tuple[float, float, float, float, int]:
    """
    由3点拟合平面 ax+by+cz+d=0（单位法向）
    返回 (a,b,c,d,is_valid)
    """
    v1x = p2[0] - p1[0]
    v1y = p2[1] - p1[1]
    v1z = p2[2] - p1[2]
    v2x = p3[0] - p1[0]
    v2y = p3[1] - p1[1]
    v2z = p3[2] - p1[2]

    # 叉乘得到法向
    nx = v1y * v2z - v1z * v2y
    ny = v1z * v2x - v1x * v2z
    nz = v1x * v2y - v1y * v2x

    a, b, c, nlen = _NormalizeVec3(nx, ny, nz)
    if nlen < 1e-10:
        return 0.0, 0.0, 0.0, 0.0, 0

    d = -(a * p1[0] + b * p1[1] + c * p1[2])
    return a, b, c, d, 1


@njit(cache=True)
def _WeightedSoftScorePlane(
    coords: np.ndarray,          # (N,3)
    weights: np.ndarray,         # (N,)
    a: float, b: float, c: float, d: float,
    sigma_d: float,
    subsample_step: int
) -> float:
    """
    计算曲率加权软评分:
        S = sum_i w_i * exp(-dist_i^2 / (2*sigma_d^2))

    subsample_step:
        =1 表示全量评分；
        >1 表示每隔 subsample_step 取1个点评分（加速用）。
    """
    inv_2sig2 = 1.0 / (2.0 * sigma_d * sigma_d + 1e-12)
    s = 0.0
    n = coords.shape[0]
    for i in range(0, n, subsample_step):
        x = coords[i, 0]
        y = coords[i, 1]
        z = coords[i, 2]
        dist = abs(a * x + b * y + c * z + d)  # 正交距离（法向已单位化）
        s += weights[i] * math.exp(-dist * dist * inv_2sig2)
    return s


@njit(cache=True)
def _WeightedRansacPlaneJit(
    coords: np.ndarray,          # (N,3)
    weights: np.ndarray,         # (N,)
    triplets: np.ndarray,        # (K,3) int64
    sigma_d: float,
    subsample_step: int
) -> Tuple[float, float, float, float, float]:
    """
    在给定 triplets（预生成随机三元组）上做 RANSAC：
        - 每个 triplet 拟合平面
        - 用曲率加权软评分选最优
    返回：
        best_a, best_b, best_c, best_d, best_score
    """
    best_score = -1e30
    best_a = 0.0
    best_b = 0.0
    best_c = 0.0
    best_d = 0.0

    K = triplets.shape[0]
    for k in range(K):
        i1 = triplets[k, 0]
        i2 = triplets[k, 1]
        i3 = triplets[k, 2]

        a, b, c, d, ok = _FitPlaneFrom3Points(coords[i1], coords[i2], coords[i3])
        if ok == 0:
            continue

        s = _WeightedSoftScorePlane(coords, weights, a, b, c, d, sigma_d, subsample_step)
        if s > best_score:
            best_score = s
            best_a, best_b, best_c, best_d = a, b, c, d

    return best_a, best_b, best_c, best_d, best_score


@njit(cache=True)
def _CollectInliers(
    coords: np.ndarray, a: float, b: float, c: float, d: float, dist_th: float
) -> np.ndarray:
    """
    收集硬内点索引：dist < dist_th
    返回 int64 数组（长度不定，使用两遍扫描）
    """
    n = coords.shape[0]
    # 先计数
    cnt = 0
    for i in range(n):
        dist = abs(a * coords[i, 0] + b * coords[i, 1] + c * coords[i, 2] + d)
        if dist < dist_th:
            cnt += 1
    out = np.empty((cnt,), dtype=np.int64)
    # 再填充
    j = 0
    for i in range(n):
        dist = abs(a * coords[i, 0] + b * coords[i, 1] + c * coords[i, 2] + d)
        if dist < dist_th:
            out[j] = i
            j += 1
    return out


@njit(cache=True)
def _WeightedTlsRefitPlane(
    coords: np.ndarray,          # (N,3)
    weights: np.ndarray,         # (N,)
    inliers: np.ndarray          # (M,) int64
) -> Tuple[float, float, float, float]:
    """
    加权TLS（加权PCA）重估平面：
        - 加权质心
        - 加权协方差
        - 最小特征向量为法向
    输出单位法向平面 (a,b,c,d)
    """
    m = inliers.shape[0]
    if m < 3:
        return 0.0, 0.0, 0.0, 0.0

    # 加权质心
    sw = 0.0
    cx = 0.0
    cy = 0.0
    cz = 0.0
    for t in range(m):
        i = inliers[t]
        w = weights[i]
        sw += w
        cx += w * coords[i, 0]
        cy += w * coords[i, 1]
        cz += w * coords[i, 2]
    if sw < 1e-12:
        return 0.0, 0.0, 0.0, 0.0
    cx /= sw
    cy /= sw
    cz /= sw

    # 加权协方差（3x3）
    c00 = c01 = c02 = 0.0
    c10 = c11 = c12 = 0.0
    c20 = c21 = c22 = 0.0
    for t in range(m):
        i = inliers[t]
        w = weights[i]
        x = coords[i, 0] - cx
        y = coords[i, 1] - cy
        z = coords[i, 2] - cz
        c00 += w * x * x
        c01 += w * x * y
        c02 += w * x * z
        c10 += w * y * x
        c11 += w * y * y
        c12 += w * y * z
        c20 += w * z * x
        c21 += w * z * y
        c22 += w * z * z

    # 归一化
    c00 /= sw; c01 /= sw; c02 /= sw
    c10 /= sw; c11 /= sw; c12 /= sw
    c20 /= sw; c21 /= sw; c22 /= sw

    # 对称矩阵特征分解：这里用 numpy 的 eigh 不在 numba nopython 内可用，
    # 因此采用一个简化策略：在 JIT 内返回协方差，由 Python 做 eig。
    # ——为了示例能“纯JIT”，这里用一个近似：用幂迭代找最大特征向量，再取正交补求最小特征向量不稳。
    # 所以：推荐工程实现时“JIT 内返回加权质心 + 协方差”，Python 外做 np.linalg.eigh。
    #
    # 为了保证示例稳定，这里直接返回占位（工程请用下方 Python 外重估函数）。
    return 0.0, 0.0, 0.0, 0.0


def WeightedTlsRefitPlane_Py(coords: np.ndarray, weights: np.ndarray, inliers: np.ndarray) -> Tuple[float, float, float, float]:
    """
    功能简介:
        用 Python/NumPy 做加权TLS重估（稳定可靠，且这一步只做一次，不是性能瓶颈）
    思路:
        - 加权质心
        - 加权协方差
        - np.linalg.eigh 得最小特征向量
    """
    pts = coords[inliers]
    w = weights[inliers].astype(np.float64)
    sw = float(np.sum(w)) + 1e-12
    mu = (pts * w[:, None]).sum(axis=0) / sw
    X = pts - mu[None, :]
    C = (X.T * w[None, :]) @ X / sw  # 3x3
    eigvals, eigvecs = np.linalg.eigh(C)
    n = eigvecs[:, 0]
    n = n / (np.linalg.norm(n) + 1e-12)
    d = -float(np.dot(n, mu))
    return float(n[0]), float(n[1]), float(n[2]), float(d)


def BuildCurvatureWeights(curvature: np.ndarray, kappa0_percentile: float, w_min: float) -> Dict[str, Any]:
    """
    功能简介:
        将曲率转成权重 w_i = exp(-(kappa/kappa0)^2)，并做下限截断
    输入:
        curvature: (N,)
    输出:
        dict: {weights, kappa0}
    """
    curv = curvature.astype(np.float64)
    # 若存在 NaN，则用中位数填充（示例做法；工程可用更稳健策略）
    if np.isnan(curv).any():
        med = np.nanmedian(curv)
        curv = np.where(np.isnan(curv), med, curv)

    kappa0 = float(np.percentile(curv, kappa0_percentile))
    kappa0 = max(kappa0, 1e-12)
    w = np.exp(- (curv / kappa0) ** 2)
    w = np.maximum(w, w_min).astype(np.float64)
    return {"weights": w, "kappa0": kappa0}


class WeightedPlaneRansac:
    """
    面向对象封装：曲率加权 RANSAC 平面拟合（Numba加速）
    """
    def __init__(self, cfg: WeightedRansacConfig):
        self.cfg = cfg

    def Fit(self, coords: np.ndarray, curvature: np.ndarray) -> Dict[str, Any]:
        """
        功能简介:
            对 coords 拟合平面，返回最优 plane 与内点索引，并输出关键调试量
        输入:
            coords: np.ndarray (N,3) float
            curvature: np.ndarray (N,) float
        输出:
            dict: plane(a,b,c,d), score, inliers, debug
        """
        coords = np.asarray(coords, dtype=np.float64)
        assert coords.ndim == 2 and coords.shape[1] == 3

        # 1) 曲率权重
        w_pack = BuildCurvatureWeights(curvature, KAPPA0_PERCENTILE, self.cfg.w_min)
        weights = w_pack["weights"]

        # 2) 预生成 triplets（Python外做随机，JIT内只读取）
        N = coords.shape[0]
        rng = np.random.default_rng(SEED)
        triplets = np.empty((self.cfg.num_iter, 3), dtype=np.int64)
        for k in range(self.cfg.num_iter):
            # 不放回抽样3点
            i = rng.integers(0, N)
            j = rng.integers(0, N)
            l = rng.integers(0, N)
            # 简单去重（示例；工程可更严谨）
            while j == i:
                j = rng.integers(0, N)
            while l == i or l == j:
                l = rng.integers(0, N)
            triplets[k, 0] = i
            triplets[k, 1] = j
            triplets[k, 2] = l

        # 3) subsample_step（评分可选子采样）
        if self.cfg.subsample_ratio >= 0.999:
            subsample_step = 1
        else:
            subsample_step = int(round(1.0 / max(self.cfg.subsample_ratio, 1e-6)))
            subsample_step = max(subsample_step, 1)

        # 4) Numba: 选最优模型
        with SimpleTimer("Numba Weighted RANSAC (proposal + scoring)"):
            a, b, c, d, score = _WeightedRansacPlaneJit(
                coords=coords,
                weights=weights,
                triplets=triplets,
                sigma_d=float(self.cfg.sigma_d),
                subsample_step=subsample_step
            )

        # 5) 收集硬内点
        with SimpleTimer("Collect hard inliers"):
            inliers = _CollectInliers(coords, a, b, c, d, float(self.cfg.distance_threshold))

        # 6) 加权TLS重估（Python外用 eigh，稳定可靠）
        with SimpleTimer("Weighted TLS refit (numpy eigh)"):
            ra, rb, rc, rd = WeightedTlsRefitPlane_Py(coords, weights, inliers)

        debug = {
            "N": int(N),
            "kappa0": float(w_pack["kappa0"]),
            "score": float(score),
            "inliers": int(inliers.size),
            "subsample_step": int(subsample_step),
        }

        return {
            "plane_initial": (float(a), float(b), float(c), float(d)),
            "plane_refit": (float(ra), float(rb), float(rc), float(rd)),
            "score": float(score),
            "inliers": inliers,
            "weights": weights,
            "debug": debug,
        }


def MakeSyntheticPlanePoints(n: int, outlier_ratio: float, noise_std: float, seed: int = 0):
    """
    生成一个示例数据：大部分点在一个平面附近，小部分为外点；
    曲率：示例中让外点曲率偏大、平面点曲率偏小（仅用于演示）
    """
    rng = np.random.default_rng(seed)

    # 真平面：n·x + d = 0
    n_true = np.array([0.2, -0.3, 0.93], dtype=np.float64)
    n_true = n_true / (np.linalg.norm(n_true) + 1e-12)
    d_true = -1.2

    n_in = int(round(n * (1.0 - outlier_ratio)))
    n_out = n - n_in

    # 在平面上采样两个自由度 u,v
    # 构造平面基
    ref = np.array([0.0, 0.0, 1.0]) if abs(n_true[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(ref, n_true); u /= (np.linalg.norm(u) + 1e-12)
    v = np.cross(n_true, u); v /= (np.linalg.norm(v) + 1e-12)

    uv = rng.uniform(-2.0, 2.0, size=(n_in, 2))
    pts_plane = uv[:, 0:1] * u[None, :] + uv[:, 1:2] * v[None, :]
    # 平面偏移到满足 d
    # 对任一点 x，加一个沿法向的偏移 t，使 n·(x+t n)+d=0 => t=-(n·x+d)
    t = -(pts_plane @ n_true + d_true)
    pts_plane = pts_plane + t[:, None] * n_true[None, :]

    # 加噪声（沿法向）
    pts_plane = pts_plane + rng.normal(0.0, noise_std, size=(n_in, 1)) * n_true[None, :]

    # 外点：空间随机
    pts_out = rng.uniform(-2.5, 2.5, size=(n_out, 3))

    coords = np.vstack([pts_plane, pts_out]).astype(np.float64)

    # 曲率：示例构造（平面点小，外点大）
    curvature_in = rng.uniform(0.001, 0.02, size=(n_in,))
    curvature_out = rng.uniform(0.03, 0.15, size=(n_out,))
    curvature = np.hstack([curvature_in, curvature_out]).astype(np.float64)

    return coords, curvature, (n_true, d_true)


def main():
    # 生成示例数据
    with SimpleTimer("Generate synthetic data"):
        coords, curvature, gt = MakeSyntheticPlanePoints(N_SYN, OUTLIER_RATIO, NOISE_STD, seed=SEED)

    # 配置并拟合
    cfg = WeightedRansacConfig(
        num_iter=NUM_ITER,
        distance_threshold=DISTANCE_THRESHOLD,
        sigma_d=SIGMA_D,
        subsample_ratio=SUBSAMPLE_RATIO,
        w_min=W_MIN,
    )
    model = WeightedPlaneRansac(cfg)

    # 首次调用会触发 numba 编译（耗时较长属正常）
    print("[log] 第一次运行会触发 numba JIT 编译，耗时会明显高于后续运行。")
    out = model.Fit(coords, curvature)

    print("\n[关键变量打印] debug：")
    for k, v in out["debug"].items():
        print(f"  - {k}: {v}")

    print("\n[关键变量打印] plane_initial(a,b,c,d)：", out["plane_initial"])
    print("[关键变量打印] plane_refit(a,b,c,d)  ：", out["plane_refit"])

    # 简单验证：看看重估法向与真法向夹角（仅示例）
    n_true, d_true = gt
    n_est = np.array(out["plane_refit"][:3], dtype=np.float64)
    n_est /= (np.linalg.norm(n_est) + 1e-12)
    cosang = abs(float(np.dot(n_true, n_est)))
    ang = math.degrees(math.acos(max(-1.0, min(1.0, cosang))))
    print(f"\n[log] refit normal vs GT normal angle = {ang:.3f} deg")


if __name__ == "__main__":
    for i in range(10):
        main()
