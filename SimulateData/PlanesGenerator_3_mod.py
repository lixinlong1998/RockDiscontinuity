# -*- coding: utf-8 -*-
"""
三面(三棱锥 P-ABC 侧面)点云生成器 —— V3 (基于你给出的几何公式重写)

功能简介:
    - 以等边三角形 ABC 为底(边长 a), 顶点 P=(0,0,h) 构成三棱锥 P-ABC
    - 给定二面角 θ (20/40/60/80/100°), 由公式求 h, 只在三个侧面三角形上采样(不含底面)
    - 每个侧面以“过 P 的中线”为纵轴、以中线的中点为原点建立二维坐标系; 规则采样+三角形裁剪
    - 形变/噪声与 V1 一致: 弯曲(二次曲面)+网格随机起伏(K×step)+正弦波起伏+高斯噪声(绝对σ)
    - 支持面积比例控制: 在 PA 上取 A′=s·PA, 用三角形 PA′B 与 PA′C 取代 PAB 与 PCA,
      使其面积为 PBC 的 {20,40,60,80,100}%。
    - 工程化: logging+计时, 统一命名模板, 真值平面方程, 清单 CSV, 断点续跑

注意:
    - h(θ) 公式按你提供的图实现, 仅在 θ<120° 时有意义
"""

from typing import Optional, Tuple, Dict, List
import numpy as np
import os
import csv
import time
import logging

# ========================
# 常量 & 日志
# ========================
CM = 0.05  # 采样默认步长 5 cm
DEFAULT_EXPORT_DIR = r"E:\Database\_RockPoints\PlanesInCube"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("plane3_pyramid")


# ========================
# 工具函数
# ========================
def TimeIt(func):
    """装饰器: 记录函数耗时(秒)"""

    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        out = func(*args, **kwargs)
        t1 = time.perf_counter()
        logger.info(f"[耗时] {func.__name__}: {(t1 - t0):.4f}s")
        return out

    return wrapper


def Normalize(v: np.ndarray) -> np.ndarray:
    """单位化向量"""
    n = np.linalg.norm(v)
    return v if n < 1e-12 else (v / n)


def OrthonormalBasisFromNV(n: np.ndarray, v_axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    由面法向 n 和期望的纵轴 v_axis 构造 (e_u, e_v, n) 正交基
    e_v 与 v_axis 同向(在面内), e_u = n × e_v
    """
    n = Normalize(n)
    v_axis = Normalize(v_axis - np.dot(v_axis, n) * n)  # 投到面内
    if np.linalg.norm(v_axis) < 1e-9:
        # 退化时任选一条
        v_axis = Normalize(np.array([1.0, 0.0, 0.0]) - n[0] * n)
    e_v = v_axis
    e_u = Normalize(np.cross(n, e_v))
    return e_u, e_v, n


def PlaneEquationFromTriangle(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> Dict:
    """
    由三点 p,q,r 求平面方程 ax+by+cz+d=0 (单位法向)

    输入:
        p,q,r: np.ndarray shape(3,)
    输出:
        dict: {a,b,c,d}
    """
    n = Normalize(np.cross(q - p, r - p))
    d = -float(np.dot(n, p))
    return dict(a=float(n[0]), b=float(n[1]), c=float(n[2]), d=float(d))


def HFromTheta(a: float, theta_deg: float) -> float:
    """
    由二面角 θ 求三棱锥高 h (按用户提供公式, 仅 θ<120°)
        cosθ = (a^2 - 6 h^2) / (a^2 + 12 h^2)
        h = (a/√6) * sqrt((1 - cosθ) / (1 + 2 cosθ))
    """
    c = np.cos(np.deg2rad(theta_deg))
    if c <= -0.5:
        # θ>=120° 趋向无穷大, 此处限制
        c = -0.499999
    h = (a / np.sqrt(6.0)) * np.sqrt(max(0.0, (1.0 - c) / (1.0 + 2.0 * c)))
    return float(h)


def EquilateralBase(a: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    在 z=0 平面, 以 O=(0,0,0) 为中心构造等边三角形 ABC (边长 a)
    A=(R,0,0), B=(R cos120°, R sin120°, 0), C=(R cos240°, R sin240°, 0)
    """
    R = a / np.sqrt(3.0)
    A = np.array([R, 0.0, 0.0], dtype=float)
    B = np.array([R * np.cos(np.deg2rad(120.0)), R * np.sin(np.deg2rad(120.0)), 0.0], dtype=float)
    C = np.array([R * np.cos(np.deg2rad(240.0)), R * np.sin(np.deg2rad(240.0)), 0.0], dtype=float)
    return A, B, C


def TriangleUVFromBasis(p: np.ndarray, q: np.ndarray, r: np.ndarray, o_face: np.ndarray,
                        e_u: np.ndarray, e_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将三角形顶点 p,q,r 投影到 2D 面局部坐标 (e_u, e_v), 原点 o_face
    返回 2D 顶点: P2, Q2, R2 (shape(2,))
    """
    P2 = np.array([np.dot(p - o_face, e_u), np.dot(p - o_face, e_v)], dtype=float)
    Q2 = np.array([np.dot(q - o_face, e_u), np.dot(q - o_face, e_v)], dtype=float)
    R2 = np.array([np.dot(r - o_face, e_u), np.dot(r - o_face, e_v)], dtype=float)
    return P2, Q2, R2


def PointInTriUV(u: np.ndarray, v: np.ndarray, P2: np.ndarray, Q2: np.ndarray, R2: np.ndarray) -> np.ndarray:
    """
    2D 点集 (u,v) 的三角形内测 (含边)
    用有向面积法/重心坐标, 返回布尔掩码
    """

    def cross2(a, b):
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    X = np.stack([u, v], axis=-1)
    v0 = R2 - P2
    v1 = Q2 - P2
    v2 = X - P2[None, :]

    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    # 逐点计算
    dot02 = v2[..., 0] * v0[0] + v2[..., 1] * v0[1]
    dot11 = np.dot(v1, v1)
    dot12 = v2[..., 0] * v1[0] + v2[..., 1] * v1[1]

    inv_denom = 1.0 / max(dot00 * dot11 - dot01 * dot01, 1e-12)
    u_b = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v_b = (dot00 * dot12 - dot01 * dot02) * inv_denom
    w_b = 1.0 - u_b - v_b
    mask = (u_b >= -1e-9) & (v_b >= -1e-9) & (w_b >= -1e-9)
    return mask


# ========================
# 形变场（与 V1 保持一致的接口）
# ========================
def BendDisplacement(u: np.ndarray, v: np.ndarray, u_half: float, v_half: float, kappa: float,
                     amp: float) -> np.ndarray:
    """二次曲面弯曲: dn = kappa * amp * ((u/u_half)^2 + (v/v_half)^2)"""
    uu = (u / max(u_half, 1e-9)) ** 2
    vv = (v / max(v_half, 1e-9)) ** 2
    return kappa * amp * (uu + vv)


def GridRandomDisplacement(u: np.ndarray, v: np.ndarray, u_half: float, v_half: float,
                           grid_n_u: int, grid_n_v: int, amp: float, rng: np.random.Generator) -> np.ndarray:
    """控制网格随机起伏(双线性插值)"""
    u_coords = np.linspace(-u_half, u_half, grid_n_u)
    v_coords = np.linspace(-v_half, v_half, grid_n_v)
    ctrl = rng.uniform(-amp, amp, size=(grid_n_u, grid_n_v))

    def interp_1d(x, x_coords):
        x = np.clip(x, x_coords[0], x_coords[-1])
        idx = np.searchsorted(x_coords, x) - 1
        idx = np.clip(idx, 0, len(x_coords) - 2)
        x0 = x_coords[idx];
        x1 = x_coords[idx + 1]
        t = np.where(np.abs(x1 - x0) < 1e-12, 0.0, (x - x0) / (x1 - x0))
        return idx, t

    iu, tu = interp_1d(u, u_coords)
    iv, tv = interp_1d(v, v_coords)

    c00 = ctrl[iu, iv];
    c10 = ctrl[iu + 1, iv];
    c01 = ctrl[iu, iv + 1];
    c11 = ctrl[iu + 1, iv + 1]
    c0 = c00 * (1 - tu) + c10 * tu
    c1 = c01 * (1 - tu) + c11 * tu
    return c0 * (1 - tv) + c1 * tv


def WaveDisplacement(u: np.ndarray, v: np.ndarray, u_half: float, v_half: float,
                     grid_n_u: int, grid_n_v: int, amp: float) -> np.ndarray:
    """正弦波起伏(频率与控制格一致)"""
    fu = max(grid_n_u - 1, 1) / max(2 * u_half, 1e-9)
    fv = max(grid_n_v - 1, 1) / max(2 * v_half, 1e-9)
    return amp * np.sin(2 * np.pi * fu * (u + u_half)) * np.sin(2 * np.pi * fv * (v + v_half))


# ========================
# 三角面采样器
# ========================
class TriangleFaceSampler:
    """
    三角面在其本地 2D 坐标系上的规则采样 + (弯曲/网格随机/正弦)形变 + 噪声

    实现思路:
        1) 以面三点 (P,T,Q) 求面法向 n; 底边中点 M=(T+Q)/2
        2) 本地基: e_v(沿 P→M), e_u=n×e_v, 原点 o_face=中线中点=(P+M)/2
        3) 将三点投到 (e_u,e_v), 得到 P2,T2,Q2; 用其外接对称盒构造规则网格 (step)
        4) 用三角形掩码裁剪; 仅保留在三角形内部(含边)的 (u,v)
        5) 形变: dn=Bend+Grid+Wave(沿 n), 叠加绝对高斯噪声(若设)
        6) 映射回 3D 并裁到 cube
    """

    def __init__(self, cube_size: float, seed: int = 0):
        self.cube_size = float(cube_size)
        self.rng = np.random.default_rng(seed)

    @TimeIt
    def Sample(self,
               P: np.ndarray, T: np.ndarray, Q: np.ndarray,
               step: float,
               bend_kappa: float,
               grid_size_k: Optional[int],
               bend_amp: float,
               grid_amp: float,
               wave_amp: float,
               apply_grid_warp: bool,
               apply_wave_warp: bool,
               noise_level: float,
               noise_sigma_abs: Optional[float]) -> np.ndarray:
        """
        输入:
            P,T,Q: 三角面三顶点(3,)
            step: 采样间距(米)
            bend_kappa: 弯曲度[0,1]
            grid_size_k: 控制网格尺寸 = K×step; None 表示关闭格/波两类形变
            bend_amp/grid_amp/wave_amp: 各形变法向幅值(米)
            apply_grid_warp/apply_wave_warp: 是否启用对应形变
            noise_level: 遗留(相对 1cm), 当 noise_sigma_abs is None 时生效
            noise_sigma_abs: 绝对噪声 σ(米)
        输出:
            pts: (M,3) float32
        """
        # 面法向与本地基
        n = Normalize(np.cross(T - P, Q - P))
        M = 0.5 * (T + Q)
        e_u, e_v, n = OrthonormalBasisFromNV(n, v_axis=(M - P))
        o_face = 0.5 * (P + M)

        # 顶点在局部 2D 的坐标
        P2, T2, Q2 = TriangleUVFromBasis(P, T, Q, o_face, e_u, e_v)

        # 对称包围盒
        u_half = float(max(abs(P2[0]), abs(T2[0]), abs(Q2[0]))) + 1e-9
        v_half = float(max(abs(P2[1]), abs(T2[1]), abs(Q2[1]))) + 1e-9

        # 规则网格
        u_vals = np.arange(-u_half, u_half + 1e-9, step)
        v_vals = np.arange(-v_half, v_half + 1e-9, step)
        U, V = np.meshgrid(u_vals, v_vals, indexing="xy")

        # 三角形掩码
        mask = PointInTriUV(U, V, P2, T2, Q2)
        U = U[mask];
        V = V[mask]

        # 先映射到 3D 基面
        pts = (o_face[None, :]
               + U[:, None] * e_u[None, :]
               + V[:, None] * e_v[None, :])

        # 控制网格分辨率
        grid_n_u = grid_n_v = None
        if grid_size_k is not None and grid_size_k >= 1:
            grid_size = max(grid_size_k * step, 1e-9)
            grid_n_u = max(2, int(np.floor((2 * u_half) / grid_size)) + 1)
            grid_n_v = max(2, int(np.floor((2 * v_half) / grid_size)) + 1)

        # 形变: 法向位移
        if bend_kappa > 0.0:
            dn = BendDisplacement(U, V, u_half, v_half, bend_kappa, bend_amp)
            pts = pts + dn[:, None] * n[None, :]
        if apply_grid_warp and (grid_n_u is not None):
            dn = GridRandomDisplacement(U, V, u_half, v_half, grid_n_u, grid_n_v, grid_amp, self.rng)
            pts = pts + dn[:, None] * n[None, :]
        if apply_wave_warp and (grid_n_u is not None):
            dn = WaveDisplacement(U, V, u_half, v_half, grid_n_u, grid_n_v, wave_amp)
            pts = pts + dn[:, None] * n[None, :]

        # 噪声
        sigma = None
        if (noise_sigma_abs is not None) and (noise_sigma_abs > 0.0):
            sigma = float(noise_sigma_abs)
        elif noise_level > 0.0:
            sigma = noise_level * CM
        if sigma is not None and sigma > 0.0:
            noise = self.rng.normal(0.0, sigma, size=pts.shape)
            pts = pts + noise

        # 裁剪到 cube
        # pts = np.clip(pts, 0.0, self.cube_size)
        return pts.astype(np.float32)


# ========================
# 生成器
# ========================
class PyramidPointCloudGenerator:
    """
    三棱锥 P-ABC 侧面点云生成器
    """

    def __init__(self, cube_size: float = 10.0, seed: int = 0):
        self.cube_size = float(cube_size)
        self.rng = np.random.default_rng(seed)
        self.sampler = TriangleFaceSampler(cube_size=cube_size, seed=seed)

    @TimeIt
    def Build(self,
              angle_deg: int,
              a: float,
              pa_prime_percent: int,
              step: float = CM,
              bend_kappa: float = 0.0,
              bend_amp: float = 0.2,
              grid_amp: float = 0.05,
              wave_amp: float = 0.05,
              grid_size_k: Optional[int] = None,
              apply_grid_warp: bool = False,
              apply_wave_warp: bool = False,
              noise_level: float = 0.0,
              noise_sigma_abs: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        生成一个三棱锥侧面点云样本

        输入:
            angle_deg: 指定二面角 θ(度), {20,40,60,80,100}
            a: 底边等边三角形边长(米)
            pa_prime_percent: A' 在 PA 上的百分比 {20,40,60,80,100}
            step/bend_kappa/bend_amp/grid_amp/wave_amp/grid_size_k/apply_grid_warp/apply_wave_warp:
                与 V1/V2 口径一致
            noise_level / noise_sigma_abs: 噪声控制(优先绝对 σ)

        输出:
            points: (N,3) float32
            labels: (N,) int (0:P A'B, 1:P B C, 2:P A'C)
            meta: 元信息(真值平面/实际角/参数等)
        """
        # 1) 构造顶点
        h = HFromTheta(a, angle_deg)  # 按给定 θ 求 h
        A, B, C = EquilateralBase(a)
        P = np.array([0.0, 0.0, h], dtype=float)

        # 2) A' 在 PA 上
        s = float(pa_prime_percent) / 100.0  # A' = s * PA
        A_prime = P + s * (A - P)

        # 三个侧面三角形 (按标签顺序):
        tri0 = (P, A_prime, B)  # 0
        tri1 = (P, B, C)  # 1
        tri2 = (P, A_prime, C)  # 2

        # 3) 逐面采样
        pts_list = []
        lbl_list = []
        for face_id, (X, Y, Z) in enumerate([tri0, tri1, tri2]):
            pts_face = self.sampler.Sample(
                P=X, T=Y, Q=Z,
                step=step,
                bend_kappa=bend_kappa,
                grid_size_k=grid_size_k,
                bend_amp=bend_amp,
                grid_amp=grid_amp,
                wave_amp=wave_amp,
                apply_grid_warp=apply_grid_warp,
                apply_wave_warp=apply_wave_warp,
                noise_level=noise_level,
                noise_sigma_abs=noise_sigma_abs
            )
            lbl_face = np.full((pts_face.shape[0],), face_id, dtype=int)
            pts_list.append(pts_face)
            lbl_list.append(lbl_face)

        points = np.vstack(pts_list)
        labels = np.concatenate(lbl_list)

        # 4) 真值平面与角度
        gt_planes = []
        for pid, (X, Y, Z) in enumerate([tri0, tri1, tri2]):
            gt_planes.append(dict(plane_id=pid, **PlaneEquationFromTriangle(X, Y, Z)))
        # 三对二面角(应相等): 0-1、1-2、2-0
        n0 = Normalize(np.cross(tri0[1] - tri0[0], tri0[2] - tri0[0]))
        n1 = Normalize(np.cross(tri1[1] - tri1[0], tri1[2] - tri1[0]))
        n2 = Normalize(np.cross(tri2[1] - tri2[0], tri2[2] - tri2[0]))

        def angle_deg_between(na, nb):
            c = np.clip(np.dot(na, nb), -1.0, 1.0)
            return float(np.degrees(np.arccos(c)))

        deg01 = angle_deg_between(n0, n1)
        deg12 = angle_deg_between(n1, n2)
        deg20 = angle_deg_between(n2, n0)
        deg_actual = float((deg01 + deg12 + deg20) / 3.0)

        meta = dict(
            plane_count=3,
            angle_deg=int(angle_deg),
            deg_actual=deg_actual,
            base_a=a,
            apex_h=h,
            pa_prime_percent=int(pa_prime_percent),
            area_ratio=(int(pa_prime_percent), 100, int(pa_prime_percent)),  # 对应 tri0,tri1,tri2
            step=step,
            bend_kappa=bend_kappa,
            grid_size_k=grid_size_k,
            apply_grid_warp=apply_grid_warp,
            apply_wave_warp=apply_wave_warp,
            noise_level=noise_level,
            noise_sigma_abs=noise_sigma_abs,
            n_points=int(points.shape[0]),
            gt_planes=gt_planes,
        )
        logger.info(
            f"[完成] 点数: {points.shape[0]} | 目标角: {angle_deg}°, 实际角均值: {deg_actual:.4f}° (各对: {deg01:.3f},{deg12:.3f},{deg20:.3f})")
        return points, labels, meta


# ========================
# 导出 & 命名模板
# ========================
def PercentListToStr(vals: Tuple[int, int, int]) -> str:
    return "_".join([f"{int(v)}%" for v in vals])


def MakeFileName(angle_deg: int,
                 area_ratio_percent: Tuple[int, int, int],
                 noise_sigma_abs_m: float,
                 bend_percent: int,
                 grid_size_k: int,
                 wave_percent: int) -> str:
    """
    Plane3_Ang{θ}_Ara{p1%}_{p2%}_{p3%}_Gno{mm}mm_Ben{ben}_Grid{K}_Sin{sin}.ply
    """
    gno_mm = int(round(noise_sigma_abs_m * 1000.0))
    ara = PercentListToStr(area_ratio_percent)
    return (f"Plane3_Ang{int(angle_deg)}_Ara{ara}_"
            f"Gno{gno_mm}mm_Ben{bend_percent}_Grid{grid_size_k}_Sin{wave_percent}.ply")


def SaveAsPLY(path: str, pts: np.ndarray, lbl: Optional[np.ndarray] = None):
    """最简 ASCII PLY 导出"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = pts.shape[0]
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if lbl is not None:
            f.write("property int label\n")
        f.write("end_header\n")
        if lbl is None:
            for i in range(n):
                f.write(f"{pts[i, 0]:.6f} {pts[i, 1]:.6f} {pts[i, 2]:.6f}\n")
        else:
            for i in range(n):
                f.write(f"{pts[i, 0]:.6f} {pts[i, 1]:.6f} {pts[i, 2]:.6f} {int(lbl[i])}\n")
    logger.info(f"[导出] PLY: {path}")


# ========================
# 批量
# ========================
if __name__ == "__main__":
    gen = PyramidPointCloudGenerator(cube_size=10.0, seed=7)

    # 形变基准(可按需改)
    bend_amp_base = 0.2
    grid_amp_base = 0.05
    wave_amp_base = 0.05

    # 批量参数
    angle_list = [20, 40, 60, 80, 100]  # 二面夹角θ
    a_edge = 4.0  # 底边 a (米)
    pa_prime_percent_list = [20, 40, 60, 80, 100]  # A' 比例
    noise_sigma_abs_mm_list = [0, 20, 40, 60, 80, 100]  # 绝对噪声 σ(毫米)
    bend_percent_list = [0, 20, 40, 60, 80, 100]  # 弯曲度百分比 0..100
    grid_size_k_list = [0, 2, 4, 6, 8, 10]  # 控制网格尺寸 = K × step
    wave_percent_list = [0, 20, 40, 60, 80, 100]  # 正弦起伏百分比 0..100

    batch_dir = os.path.join(DEFAULT_EXPORT_DIR, "batch_plane3")
    os.makedirs(batch_dir, exist_ok=True)
    manifest_path = os.path.join(batch_dir, "manifest_plane3.csv")
    write_header = not os.path.exists(manifest_path)

    with open(manifest_path, "a", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        if write_header:
            writer.writerow([
                "filename",
                "plane_count",
                "angle_deg",
                "deg_actual",
                "base_a",
                "apex_h",
                "area_ratio_percent_1",  # tri0 = P A' B
                "area_ratio_percent_2",  # tri1 = P B C (100%)
                "area_ratio_percent_3",  # tri2 = P A' C
                "noise_sigma_abs_m",
                "bend_percent",
                "grid_size_k",
                "wave_percent",
                "bend_amp",
                "grid_amp",
                "wave_amp_abs",
                "n_points",
                # 真值平面
                "A1", "B1", "C1", "D1",
                "A2", "B2", "C2", "D2",
                "A3", "B3", "C3", "D3"
            ])

        total_new, skipped = 0, 0
        for ang in angle_list:
            for p_pct in pa_prime_percent_list:
                for gno_mm in noise_sigma_abs_mm_list:
                    noise_sigma_abs = gno_mm / 1000.0
                    for ben in bend_percent_list:
                        bend_kappa = ben / 100.0
                        for grid_k in grid_size_k_list:
                            for sinp in wave_percent_list:
                                wave_amp = (sinp / 100.0) * wave_amp_base
                                apply_wave = (sinp > 0)

                                fname = MakeFileName(
                                    angle_deg=ang,
                                    area_ratio_percent=(p_pct, 100, p_pct),
                                    noise_sigma_abs_m=noise_sigma_abs,
                                    bend_percent=ben,
                                    grid_size_k=grid_k,
                                    wave_percent=sinp
                                )
                                fpath = os.path.join(batch_dir, fname)
                                if os.path.exists(fpath):
                                    skipped += 1
                                    continue

                                pts, lbl, meta = gen.Build(
                                    angle_deg=ang,
                                    a=a_edge,
                                    pa_prime_percent=p_pct,
                                    step=CM,
                                    bend_kappa=bend_kappa,
                                    bend_amp=bend_amp_base,
                                    grid_amp=grid_amp_base,
                                    wave_amp=wave_amp,
                                    grid_size_k=grid_k,
                                    apply_grid_warp=True,
                                    apply_wave_warp=apply_wave,
                                    noise_sigma_abs=noise_sigma_abs
                                )

                                SaveAsPLY(fpath, pts, lbl)

                                # 真值
                                A1 = B1 = C1 = D1 = A2 = B2 = C2 = D2 = A3 = B3 = C3 = D3 = float("nan")
                                if "gt_planes" in meta and len(meta["gt_planes"]) == 3:
                                    A1 = meta["gt_planes"][0]["a"];
                                    B1 = meta["gt_planes"][0]["b"];
                                    C1 = meta["gt_planes"][0]["c"];
                                    D1 = meta["gt_planes"][0]["d"]
                                    A2 = meta["gt_planes"][1]["a"];
                                    B2 = meta["gt_planes"][1]["b"];
                                    C2 = meta["gt_planes"][1]["c"];
                                    D2 = meta["gt_planes"][1]["d"]
                                    A3 = meta["gt_planes"][2]["a"];
                                    B3 = meta["gt_planes"][2]["b"];
                                    C3 = meta["gt_planes"][2]["c"];
                                    D3 = meta["gt_planes"][2]["d"]

                                writer.writerow([
                                    fname,
                                    3,
                                    meta.get("angle_deg"),
                                    f"{meta.get('deg_actual', float('nan')):.6f}",
                                    meta.get("base_a", a_edge),
                                    meta.get("apex_h", float("nan")),
                                    p_pct, 100, p_pct,
                                    meta.get("noise_sigma_abs", 0.0),
                                    ben,
                                    grid_k,
                                    sinp,
                                    bend_amp_base,
                                    grid_amp_base,
                                    wave_amp,
                                    meta.get("n_points", pts.shape[0]),
                                    A1, B1, C1, D1,
                                    A2, B2, C2, D2,
                                    A3, B3, C3, D3
                                ])
                                total_new += 1

        logger.info(f"[批量完成] 新生成: {total_new} 个, 跳过: {skipped} 个。清单: {manifest_path}")
