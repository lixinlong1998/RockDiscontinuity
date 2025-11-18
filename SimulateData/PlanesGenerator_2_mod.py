# -*- coding: utf-8 -*-
"""
两平面“书本式”二面角点云生成器 (V2 基于 V1 能力增强版)

功能简介:
    - 在 cube(默认 10 m) 中生成共享一条公共棱(铰链线)的两平面 ΠA、ΠB
    - 二面角 = 指定夹角 θ ∈ {20,40,60,80,100,120,140,160} 度
    - 以公共棱为界, 对每个平面裁去一半(书本状), 只保留各自一侧的半平面点
    - 面积占比: A 是 B 的 {20,40,60,80,100}% (通过线性尺度 √r 控制)
    - 形变与噪声: 与 V1 一致
        * 弯曲(二次面): bend_kappa ∈ [0,1], 幅值 bend_amp(米)
        * 流形 I(随机格): grid_size_k ∈ {2,4,6,8,10} → 控制网格单元尺寸 = K×step
        * 流形 II(正弦波): wave_amp(米), 频率与控制格一致
        * 高斯噪声: 优先使用 noise_sigma_abs(米), 独立于 step
    - 工程化:
        * logging + TimeIt 计时, 断点续跑
        * 清单 CSV: 记录文件名/参数/真值平面(A1..D1, A2..D2)/实际夹角等
        * 统一命名模板, 比例以百分比输出, 仅使用下划线

实现思路:
    1) 构造共棱几何:
        - 设公共棱方向 h = (0,0,1), 过点 p0 = (cube/2, cube/2, cube/2)
        - 令 nA = e1 = (1,0,0), nB = cosθ·e1 + sinθ·e2, e2=(0,1,0)
        - ΠA: nA·(x - p0)=0, ΠB: nB·(x - p0)=0
    2) 对每个补丁(矩形)在自身局部坐标系采样 -> 叠加形变 -> 加噪 -> 点集
    3) 书本裁剪:
        - A 面保留 nB·(x - p0) ≥ 0, B 面保留 nA·(x - p0) ≥ 0
    4) 统一导出与清单记录, 并裁到 [0,cube]

输入/输出约定:
    - 代码内直接定义批量参数(不使用 argparse)
    - 导出到 ./_exports/batch_plane2/

作者: 你当前项目内使用
"""

from typing import Optional, Tuple, Dict, List
import numpy as np
import os
import csv
import time
import logging

# ========================
# 全局常量与日志
# ========================
CM = 0.05  # 采样默认步长 5 cm
DEFAULT_EXPORT_DIR = r"E:\Database\_RockPoints\PlanesInCube"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("plane2_book")


# ========================
# 工具: 计时与向量
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


def OrthonormalBasis(n: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """由法向生成正交基 t1,t2,n"""
    n = Normalize(n)
    if abs(n[2]) < 0.9:
        t1 = Normalize(np.cross(n, np.array([0.0, 0.0, 1.0])))
    else:
        t1 = Normalize(np.cross(n, np.array([0.0, 1.0, 0.0])))
    t2 = Normalize(np.cross(n, t1))
    return t1, t2, n


# ========================
# 补丁规格与构造器
# ========================
class PatchSpec:
    """
    面片规格

    属性:
        center: np.ndarray shape(3,) 面中心
        normal: np.ndarray shape(3,) 面法向(未必单位化)
        size_u, size_v: float 矩形面两向尺寸(米)
        is_triangle: bool 是否按三角形掩码裁剪(本文件用矩形, False)
    """

    def __init__(self, center: np.ndarray, normal: np.ndarray, size_u: float, size_v: float, is_triangle: bool = False):
        self.center = np.asarray(center, dtype=float)
        self.normal = np.asarray(normal, dtype=float)
        self.size_u = float(size_u)
        self.size_v = float(size_v)
        self.is_triangle = bool(is_triangle)


class PlanePatchBuilder:
    """
    两平面共棱构造器

    实现思路(详):
        - 公共棱方向 h=(0,0,1), 过点 p0=cube/2
        - 目标夹角 θ: 根据 θ 构造 nA, nB (均 ⟂ h)
        - 为便于“书本裁剪”, 两补丁中心选在 p0 附近; 采样后再按对方平面符号裁半
    """

    def __init__(self, cube_size: float, rng: Optional[np.random.Generator] = None):
        self.cube_size = float(cube_size)
        self.rng = np.random.default_rng(0) if rng is None else rng

    def BuildTwoPlanesBook(self, angle_deg: int, size_b: float, ratio_a_over_b: float) -> Tuple[List[PatchSpec], Dict]:
        """
        构造“共棱 + 指定夹角”的两矩形补丁

        输入:
            angle_deg: int 目标二面角(20..160)
            size_b: float B 面基准边长(米), B 面矩形 size_u = size_v = size_b
            ratio_a_over_b: float 面积占比 r=A/B ∈ {0.2,0.4,0.6,0.8,1.0}
        输出:
            patches: [PatchSpec_A, PatchSpec_B]
            geom_meta: dict 包含 nA,nB,p0,h 等几何真值
        """
        theta = np.deg2rad(float(angle_deg))
        # 公共棱几何
        h = np.array([0.0, 0.0, 1.0], dtype=float)  # 铰链方向
        h = Normalize(h)
        cube = self.cube_size
        p0 = np.array([cube / 2.0, cube / 2.0, cube / 2.0], dtype=float)

        # XY 平面上取 nA, nB
        e1 = np.array([1.0, 0.0, 0.0], dtype=float)
        e2 = np.array([0.0, 1.0, 0.0], dtype=float)
        nA = Normalize(e1)
        nB = Normalize(np.cos(theta) * e1 + np.sin(theta) * e2)

        # 面尺寸: A 是 B 的 r 倍(面积), 线性尺度 √r 倍
        scale_a = np.sqrt(max(ratio_a_over_b, 1e-9))
        size_a = size_b * scale_a

        # 让补丁中心在 p0, 便于对称裁剪
        patchA = PatchSpec(center=p0.copy(), normal=nA, size_u=size_a, size_v=size_a, is_triangle=False)
        patchB = PatchSpec(center=p0.copy(), normal=nB, size_u=size_b, size_v=size_b, is_triangle=False)

        meta = dict(p0=p0, h=h, nA=nA, nB=nB, theta_deg=float(angle_deg))
        return [patchA, patchB], meta


# ========================
# 形变场
# ========================
def BendDisplacement(u: np.ndarray, v: np.ndarray, u_half: float, v_half: float, kappa: float,
                     amp: float) -> np.ndarray:
    """二次曲面型弯曲: dn = kappa * amp * ((u/u_half)^2 + (v/v_half)^2)"""
    uu = (u / max(u_half, 1e-9)) ** 2
    vv = (v / max(v_half, 1e-9)) ** 2
    dn = kappa * amp * (uu + vv)
    return dn


def GridRandomDisplacement(u: np.ndarray, v: np.ndarray, u_half: float, v_half: float,
                           grid_n_u: int, grid_n_v: int, amp: float, rng: np.random.Generator) -> np.ndarray:
    """
    控制网格随机起伏(双线性插值)
    """
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
    dn = c0 * (1 - tv) + c1 * tv
    return dn


def WaveDisplacement(u: np.ndarray, v: np.ndarray, u_half: float, v_half: float,
                     grid_n_u: int, grid_n_v: int, amp: float) -> np.ndarray:
    """正弦波起伏, 频率与控制格一致"""
    fu = max(grid_n_u - 1, 1) / max(2 * u_half, 1e-9)
    fv = max(grid_n_v - 1, 1) / max(2 * v_half, 1e-9)
    dn = amp * np.sin(2 * np.pi * fu * (u + u_half)) * np.sin(2 * np.pi * fv * (v + v_half))
    return dn


# ========================
# 真值平面方程
# ========================
def PlaneEquationFrom(n: np.ndarray, p0: np.ndarray) -> Dict:
    """
    由法向 n 和过点 p0 计算平面 ax+by+cz+d=0 (单位法向)
    """
    nn = Normalize(n)
    d = -float(np.dot(nn, p0))
    return dict(a=float(nn[0]), b=float(nn[1]), c=float(nn[2]), d=float(d))


# ========================
# 采样器与构建
# ========================
class PlanePointCloudGenerator:
    """
    两平面“书本式”点云生成器

    关键函数:
        SamplePatch(...)  —— 规则采样 + (弯曲/格点/正弦)形变 + 噪声 + 夹 cube
        Build(...)        —— 构造两补丁, 采样, 书本裁剪, 合并导出元信息
    """

    def __init__(self, cube_size: float = 10.0, seed: int = 0):
        self.cube_size = float(cube_size)
        self.rng = np.random.default_rng(seed)

    @TimeIt
    def SamplePatch(self,
                    patch: PatchSpec,
                    step: float = CM,
                    bend_kappa: float = 0.0,
                    grid_size_k: Optional[int] = None,
                    bend_amp: float = 0.2,
                    grid_amp: float = 0.05,
                    wave_amp: float = 0.05,
                    apply_grid_warp: bool = False,
                    apply_wave_warp: bool = False,
                    noise_level: float = 0.0,
                    noise_sigma_abs: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        采样单个补丁并施加形变与噪声

        实现思路(详):
            - 在局部 (u,v) 规则网格上生成点, 映射到 3D: x = center + u*t1 + v*t2
            - 形变以法向 n 方向叠加位移 dn (弯曲/格点/正弦)
            - 高斯噪声优先使用 noise_sigma_abs(米), 否则兼容 legacy: noise_level*CM
            - 最后将点夹到 [0,cube]

        输入:
            patch: PatchSpec 面片规格
            step: float 采样步长(米)
            bend_kappa: float 弯曲度(0..1)
            grid_size_k: Optional[int] 控制网格单元尺寸 = K×step
            bend_amp/grid_amp/wave_amp: float 形变幅值(米)
            apply_grid_warp/apply_wave_warp: bool 是否启用对应形变
            noise_level: float 兼容旧逻辑(相对 1cm 的比例)
            noise_sigma_abs: Optional[float] 绝对噪声σ(米)

        输出:
            pts: (M,3) float32
            lbl: (M,)  int 占位标签(由外层赋值)
        """
        t1, t2, n = OrthonormalBasis(patch.normal)

        # 局部规则网格
        u_half = patch.size_u / 2.0
        v_half = patch.size_v / 2.0
        u_vals = np.arange(-u_half, u_half + 1e-9, step)
        v_vals = np.arange(-v_half, v_half + 1e-9, step)
        U, V = np.meshgrid(u_vals, v_vals, indexing='xy')

        base = patch.center
        pts = (base[None, None, :]
               + U[..., None] * t1[None, None, :]
               + V[..., None] * t2[None, None, :]).reshape(-1, 3)
        U = U.reshape(-1);
        V = V.reshape(-1)

        # 控制网格分辨率由 K×step 折算
        grid_n_u = grid_n_v = None
        if grid_size_k is not None and grid_size_k >= 1:
            grid_size = max(grid_size_k * step, 1e-9)
            grid_n_u = max(2, int(np.floor((2 * u_half) / grid_size)) + 1)
            grid_n_v = max(2, int(np.floor((2 * v_half) / grid_size)) + 1)

        # 弯曲
        if bend_kappa > 0.0:
            dn = BendDisplacement(U, V, u_half, v_half, bend_kappa, bend_amp)
            pts = pts + dn[:, None] * n[None, :]

        # 格点随机
        if apply_grid_warp and (grid_n_u is not None):
            dn = GridRandomDisplacement(U, V, u_half, v_half, grid_n_u, grid_n_v, grid_amp, self.rng)
            pts = pts + dn[:, None] * n[None, :]

        # 正弦波
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

        # 裁到 cube
        pts = np.clip(pts, 0.0, self.cube_size)
        lbl = np.zeros((pts.shape[0],), dtype=int)
        return pts.astype(np.float32), lbl

    @TimeIt
    def Build(self,
              angle_deg: int,
              ratio_a_percent_of_b: int,
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
        生成“书本式”二面角的两平面点云

        输入:
            angle_deg: int 目标夹角, 20..160
            ratio_a_percent_of_b: int A 是 B 的百分比(20/40/60/80/100)
            其余参数同 SamplePatch

        输出:
            points: (N,3) float32
            labels: (N,) int (A 面=0, B 面=1)
            meta: Dict 包含 gt_planes、实际夹角、参数等
        """
        # 构造两补丁(共棱)
        builder = PlanePatchBuilder(self.cube_size, rng=self.rng)
        ratio = max(1, int(ratio_a_percent_of_b))
        r = float(ratio) / 100.0  # A/B
        size_b = 4.0  # B 面基准边长(米); 可按需调整
        patches, geom = builder.BuildTwoPlanesBook(angle_deg=angle_deg, size_b=size_b, ratio_a_over_b=r)
        patchA, patchB = patches[0], patches[1]
        p0 = geom["p0"];
        nA = geom["nA"];
        nB = geom["nB"]

        # 真值平面方程(用于评测)
        gtA = PlaneEquationFrom(nA, p0)
        gtB = PlaneEquationFrom(nB, p0)
        deg_actual = float(np.degrees(np.arccos(np.clip(np.dot(Normalize(nA), Normalize(nB)), -1.0, 1.0))))

        # 采样两面
        ptsA, lblA = self.SamplePatch(
            patchA, step=step, bend_kappa=bend_kappa,
            grid_size_k=grid_size_k, bend_amp=bend_amp, grid_amp=grid_amp, wave_amp=wave_amp,
            apply_grid_warp=apply_grid_warp, apply_wave_warp=apply_wave_warp,
            noise_level=noise_level, noise_sigma_abs=noise_sigma_abs
        );
        lblA[:] = 0

        ptsB, lblB = self.SamplePatch(
            patchB, step=step, bend_kappa=bend_kappa,
            grid_size_k=grid_size_k, bend_amp=bend_amp, grid_amp=grid_amp, wave_amp=wave_amp,
            apply_grid_warp=apply_grid_warp, apply_wave_warp=apply_wave_warp,
            noise_level=noise_level, noise_sigma_abs=noise_sigma_abs
        );
        lblB[:] = 1

        # 书本裁剪: A 面按 ΠB 的符号裁半, B 面按 ΠA 的符号裁半
        sA_keep = np.dot(ptsA - p0[None, :], nB) >= 0.0
        sB_keep = np.dot(ptsB - p0[None, :], nA) >= 0.0
        ptsA = ptsA[sA_keep];
        lblA = lblA[sA_keep]
        ptsB = ptsB[sB_keep];
        lblB = lblB[sB_keep]

        # 合并
        points = np.vstack([ptsA, ptsB])
        labels = np.concatenate([lblA, lblB])

        meta = dict(
            plane_count=2,
            angle_deg=int(angle_deg),
            deg_actual=deg_actual,
            area_ratio=(int(round(100 * r)), 100),  # A%, B%
            step=step,
            bend_kappa=bend_kappa,
            grid_size_k=grid_size_k,
            apply_grid_warp=apply_grid_warp,
            apply_wave_warp=apply_wave_warp,
            noise_level=noise_level,
            noise_sigma_abs=noise_sigma_abs,
            n_points=int(points.shape[0]),
            gt_planes=[
                dict(plane_id=0, **gtA),
                dict(plane_id=1, **gtB),
            ],
        )
        logger.info(f"[完成] 点数: {points.shape[0]} | 目标角: {angle_deg}°, 实际角: {deg_actual:.3f}°")
        return points.astype(np.float32), labels, meta


# ========================
# 命名模板 & 导出
# ========================
def PercentListFromRatio(ratio: Tuple[int, int]) -> str:
    """(A%,B%) -> 'A%_B%'"""
    return "_".join([f"{int(r)}%" for r in ratio])


def MakeFileName(plane_count: int,
                 angle_deg: int,
                 area_ratio_percent: Tuple[int, int],
                 noise_sigma_abs_m: float,
                 bend_percent: int,
                 grid_size_k: int,
                 wave_percent: int) -> str:
    """
    Plane{plane_count}_Ang{angle}_Ara{A%}_{B%}_Gno{mm}mm_Ben{ben}_Grid{K}_Sin{sin}.ply
    """
    gno_mm = int(round(noise_sigma_abs_m * 1000.0))
    ara = PercentListFromRatio(area_ratio_percent)
    return (f"Plane{plane_count}_Ang{int(angle_deg)}_Ara{ara}_"
            f"Gno{gno_mm}mm_Ben{bend_percent}_Grid{grid_size_k}_Sin{wave_percent}.ply")


def SaveAsPLY(path: str, pts: np.ndarray, lbl: Optional[np.ndarray] = None):
    """最简 PLY (ASCII) 导出"""
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
# 批量主程序
# ========================
if __name__ == "__main__":
    gen = PlanePointCloudGenerator(cube_size=10.0, seed=123)

    # 形变基准
    bend_amp_base = 0.2
    grid_amp_base = 0.05
    wave_amp_base = 0.05

    # 批量参数（你可自由增删）
    angle_list = [20, 40, 60, 80, 100, 120, 140, 160]  # 夹角
    ratio_a_percent_list = [20, 40, 60, 80, 100]  # A 是 B 的百分比
    noise_sigma_abs_mm_list = [0, 20, 40, 60, 80, 100]  # 绝对噪声 σ(毫米)
    bend_percent_list = [0, 20, 40, 60, 80, 100]  # 弯曲度百分比 0..100
    grid_size_k_list = [0, 2, 4, 6, 8, 10]  # 控制网格尺寸 = K × step
    wave_percent_list = [0, 20, 40, 60, 80, 100]  # 正弦起伏百分比 0..100

    batch_dir = os.path.join(DEFAULT_EXPORT_DIR, "batch_plane2")
    os.makedirs(batch_dir, exist_ok=True)
    manifest_path = os.path.join(batch_dir, "manifest_plane2.csv")
    write_header = not os.path.exists(manifest_path)

    with open(manifest_path, "a", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        if write_header:
            writer.writerow([
                "filename",
                "plane_count",
                "angle_deg",
                "deg_actual",
                "area_ratio_percent_A",
                "area_ratio_percent_B",
                "noise_sigma_abs_m",
                "bend_percent",
                "grid_size_k",
                "wave_percent",
                "bend_amp",
                "grid_amp",
                "wave_amp_abs",
                "n_points",
                # 真值平面参数:
                "A1", "B1", "C1", "D1",
                "A2", "B2", "C2", "D2"
            ])

        total_new = 0
        skipped = 0

        for ang in angle_list:
            for a_pct in ratio_a_percent_list:
                for gno_mm in noise_sigma_abs_mm_list:
                    noise_sigma_abs = gno_mm / 1000.0
                    for ben in bend_percent_list:
                        bend_kappa = ben / 100.0
                        for grid_k in grid_size_k_list:
                            for sinp in wave_percent_list:
                                wave_amp = (sinp / 100.0) * wave_amp_base
                                apply_wave = (sinp > 0)

                                fname = MakeFileName(
                                    plane_count=2,
                                    angle_deg=ang,
                                    area_ratio_percent=(a_pct, 100),
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
                                    ratio_a_percent_of_b=a_pct,
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
                                A1 = B1 = C1 = D1 = A2 = B2 = C2 = D2 = float("nan")
                                if "gt_planes" in meta and len(meta["gt_planes"]) == 2:
                                    A1 = meta["gt_planes"][0]["a"];
                                    B1 = meta["gt_planes"][0]["b"];
                                    C1 = meta["gt_planes"][0]["c"];
                                    D1 = meta["gt_planes"][0]["d"]
                                    A2 = meta["gt_planes"][1]["a"];
                                    B2 = meta["gt_planes"][1]["b"];
                                    C2 = meta["gt_planes"][1]["c"];
                                    D2 = meta["gt_planes"][1]["d"]

                                writer.writerow([
                                    fname,
                                    2,
                                    meta.get("angle_deg"),
                                    f"{meta.get('deg_actual', float('nan')):.6f}",
                                    int(meta["area_ratio"][0]),
                                    int(meta["area_ratio"][1]),
                                    meta.get("noise_sigma_abs", 0.0),
                                    ben,
                                    grid_k,
                                    sinp,
                                    bend_amp_base,
                                    grid_amp_base,
                                    wave_amp,
                                    meta.get("n_points", pts.shape[0]),
                                    A1, B1, C1, D1,
                                    A2, B2, C2, D2
                                ])

                                total_new += 1

        logger.info(f"[批量完成] 新生成: {total_new} 个, 跳过: {skipped} 个。清单: {manifest_path}")
