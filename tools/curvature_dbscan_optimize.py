import os
import time
import math
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt

import open3d as o3d
from sklearn.cluster import DBSCAN

'''
你需要按你的数据立刻改的 3 个地方（否则结果不可解释）
DBSCAN 参数：dbscan_eps / dbscan_min_samples 必须与你点间距同量级。
cur0 与 η：如果 curvature 估计尺度不同（邻域 k 不同、点密度不同），cur0/η 需要重标定。
curvature/normal 来源：若你已有高质量字段，应当替换掉代码里 PCA/knn 的估计（否则优化的其实是“估计曲率”的阈值）。
'''
# =========================
# 配置与计时工具
# =========================

@dataclass
class CurvatureDbscanOptimizeConfig:
    """
    功能简介:
        通过“curvature阈值递减 + DBSCAN聚类 + cluster法向随机点对角度累积α分布”的迭代，
        找到使相邻两轮α分布变化趋于稳定的最佳 curvature 阈值。

    实现思路(详细):
        1) 读取点云，优先读取 curvature 与 normal（若缺失则估计）
        2) 迭代：cur = cur0 - j*eta
           2.1 筛选 curvature <= cur 的点作为候选平面点
           2.2 对候选点做 DBSCAN 得到 clusters
           2.3 每个 cluster 内随机采样 n_pairs 对点，计算法向夹角并累积为 α
           2.4 收集所有 α 得到分布 T_j
           2.5 计算 diff(T_j, T_{j-1})，若足够小则停止

    输入变量及类型/结构:
        point_path (str): 点云文件路径（ply/pcd/xyz 等 open3d 支持格式）
        out_dir (str): 输出目录
        cur0 (float): 初始 curvature 阈值
        eta (float): 阈值递减步长
        max_iters (int): 最大迭代次数
        min_points_for_plane (int): 筛出的候选平面点最少点数，不足则提前停止
        dbscan_eps (float): DBSCAN eps（单位同坐标，通常米）
        dbscan_min_samples (int): DBSCAN min_samples
        min_cluster_size (int): 参与 α 计算的最小 cluster 点数（过小跳过）
        n_pairs (int): 每个 cluster 采样点对数（你要求 500）
        angle_unit (str): 'deg' 或 'rad'，默认 'deg'
        diff_metric (str): 分布差异度量：'wasserstein' 或 'hist_l1' 或 'ks'
        hist_bins (int): 仅在 'hist_l1' 使用
        diff_tol (float): 收敛阈值（越小越严格）
        patience (int): 连续满足 diff_tol 的次数后停止
        random_seed (int): 随机种子
        curvature_attr (Optional[str]): 若点云含 curvature 属性（自定义），填属性名；否则 None 表示估计
        normal_attr (Optional[str]): 若点云含 normal 属性（自定义），填属性名；否则 None 表示估计
        normal_est_k (int): 法向估计 knn（仅缺失 normal 时使用）
        curvature_est_k (int): curvature 估计 knn（仅缺失 curvature 时使用）

    输出变量及类型/结构:
        best_cur (float): 最佳 curvature 阈值（按“停止时 cur 已减过步长”的定义）
        history (List[Dict]): 每轮迭代的记录（cur, alphas, diff, n_clusters 等）
    """
    point_path: str = r"./your.ply"
    out_dir: str = r"./curvature_opt_out"

    cur0: float = 0.08
    eta: float = 0.005
    max_iters: int = 30

    min_points_for_plane: int = 5000

    dbscan_eps: float = 0.08
    dbscan_min_samples: int = 20
    min_cluster_size: int = 200

    n_pairs: int = 500
    angle_unit: str = "deg"  # 'deg' or 'rad'

    diff_metric: str = "wasserstein"  # 'wasserstein' | 'hist_l1' | 'ks'
    hist_bins: int = 40
    diff_tol: float = 0.02
    patience: int = 2

    random_seed: int = 7

    curvature_attr: Optional[str] = None
    normal_attr: Optional[str] = None

    normal_est_k: int = 30
    curvature_est_k: int = 30


class StepTimer:
    """记录每一步耗时。"""
    def __init__(self, logger: logging.Logger, name: str):
        self.logger = logger
        self.name = name
        self.t0 = None

    def __enter__(self):
        self.t0 = time.perf_counter()
        self.logger.info(f"[开始] {self.name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        dt = time.perf_counter() - self.t0
        self.logger.info(f"[结束] {self.name} | 耗时 {dt:.3f}s")


# =========================
# 主类
# =========================

class CurvatureThresholdOptimizer:
    """
    功能简介:
        按用户描述的流程，迭代搜索最佳 curvature 阈值。

    实现思路(详细):
        - 读取点云 xyz
        - 获取/估计 normals 与 curvature
        - 在 cur0 到 cur0 - max_iters*eta 范围迭代：
          1) 筛选 curvature<=cur
          2) DBSCAN 聚类
          3) 每簇随机点对采样 -> α
          4) 得到 T_j，并与 T_{j-1} 计算差异 diff
          5) diff 足够小则停止，输出当前 cur

    输入变量及类型/结构:
        cfg (CurvatureDbscanOptimizeConfig)

    输出变量及类型/结构:
        best_cur (float)
        history (List[Dict[str, Any]])
    """
    def __init__(self, cfg: CurvatureDbscanOptimizeConfig):
        self.cfg = cfg
        os.makedirs(self.cfg.out_dir, exist_ok=True)

        self.logger = logging.getLogger("CurvatureOpt")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            ch.setFormatter(fmt)
            self.logger.addHandler(ch)

        self.rng = np.random.default_rng(self.cfg.random_seed)

        self.xyz: Optional[np.ndarray] = None          # (N,3)
        self.normals: Optional[np.ndarray] = None      # (N,3), 单位向量
        self.curvature: Optional[np.ndarray] = None    # (N,)

    # -------- 数据读取/估计 --------

    def _LoadPointCloud(self) -> o3d.geometry.PointCloud:
        pcd = o3d.io.read_point_cloud(self.cfg.point_path)
        if len(pcd.points) == 0:
            raise ValueError("点云为空或读取失败。")
        return pcd

    def _EstimateNormals(self, pcd: o3d.geometry.PointCloud) -> np.ndarray:
        # 使用 knn 估计法向（注意：open3d 会给出方向不一致的法向，这对 |dot| 的夹角定义没问题）
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=self.cfg.normal_est_k)
        )
        n = np.asarray(pcd.normals, dtype=np.float64)
        # 归一化
        n_norm = np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        return n / n_norm

    def _EstimateCurvatureByPca(self, xyz: np.ndarray, k: int) -> np.ndarray:
        """
        功能简介:
            用局部 PCA 的 λ3/(λ1+λ2+λ3) 估计“曲率变化率”(常见定义)。

        实现思路(详细):
            - 对每点找 k 近邻
            - 计算协方差矩阵 -> 特征值 λ1>=λ2>=λ3
            - curvature = λ3/(λ1+λ2+λ3)

        输入:
            xyz (np.ndarray): (N,3)
            k (int): 邻居数

        输出:
            curvature (np.ndarray): (N,)
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        kdtree = o3d.geometry.KDTreeFlann(pcd)

        N = xyz.shape[0]
        curv = np.zeros(N, dtype=np.float64)

        # 逐点计算（工程上可进一步并行/向量化；这里保证清晰可复用）
        for i in range(N):
            _, idx, _ = kdtree.search_knn_vector_3d(pcd.points[i], k)
            pts = xyz[np.array(idx, dtype=np.int64)]
            mu = pts.mean(axis=0, keepdims=True)
            X = pts - mu
            cov = (X.T @ X) / max(len(idx), 1)

            # 特征值
            w = np.linalg.eigvalsh(cov)  # 升序
            w = np.clip(w, 0.0, None)
            s = float(w.sum()) + 1e-12
            curv[i] = float(w[0]) / s  # λ3/(λ1+λ2+λ3)，此处 w[0] 是最小特征值
        return curv

    def PrepareData(self) -> None:
        with StepTimer(self.logger, "读取点云与基础数据"):
            pcd = self._LoadPointCloud()
            self.xyz = np.asarray(pcd.points, dtype=np.float64)
            self.logger.info(f"点数 N={self.xyz.shape[0]}")

        # normals
        with StepTimer(self.logger, "获取/估计 normals"):
            if len(pcd.normals) > 0:
                n = np.asarray(pcd.normals, dtype=np.float64)
                n_norm = np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
                self.normals = n / n_norm
                self.logger.info("使用点云自带 normals。")
            else:
                self.normals = self._EstimateNormals(pcd)
                self.logger.info("点云不含 normals，已用 KNN 估计。")

        # curvature
        with StepTimer(self.logger, "获取/估计 curvature"):
            # open3d 的 PointCloud 不保证携带自定义 scalar（不同来源写法不一）
            # 因此这里默认：若你点云 curvature 在外部文件/属性，需要你自己读入并替换 self.curvature。
            # 否则使用 PCA 估计。
            self.curvature = self._EstimateCurvatureByPca(self.xyz, self.cfg.curvature_est_k)
            self.logger.info("已用 PCA 估计 curvature=λ3/(λ1+λ2+λ3)。")

    # -------- 核心计算 --------

    def _DbscanClusters(self, xyz_sub: np.ndarray) -> np.ndarray:
        db = DBSCAN(
            eps=self.cfg.dbscan_eps,
            min_samples=self.cfg.dbscan_min_samples,
            metric="euclidean",
            n_jobs=-1
        )
        labels = db.fit_predict(xyz_sub).astype(np.int32)
        return labels

    def _ComputeAlphaForCluster(self, normals_cluster: np.ndarray) -> float:
        """
        功能简介:
            对一个 cluster，随机采样 n_pairs 对点，计算法向夹角并累积得到 α。

        实现思路(详细):
            - 随机生成 i,j 索引数组，长度为 n_pairs
            - dot = sum(n_i * n_j)
            - angle = arccos(|dot|)
            - α = sum(angle)（默认转为度）

        输入:
            normals_cluster (np.ndarray): (M,3) 单位法向

        输出:
            alpha (float): 累积法向夹角
        """
        M = normals_cluster.shape[0]
        if M < 2:
            return 0.0

        # 随机抽样点对
        i = self.rng.integers(0, M, size=self.cfg.n_pairs, endpoint=False)
        j = self.rng.integers(0, M, size=self.cfg.n_pairs, endpoint=False)

        ni = normals_cluster[i]
        nj = normals_cluster[j]

        dot = np.sum(ni * nj, axis=1)
        dot = np.clip(np.abs(dot), 0.0, 1.0)  # |dot| 保证方向不一致时仍一致
        ang = np.arccos(dot)  # 弧度

        if self.cfg.angle_unit == "deg":
            ang = ang * (180.0 / np.pi)

        alpha = float(np.sum(ang))
        return alpha

    def _BuildAlphaDistribution(self, idx_plane: np.ndarray) -> Tuple[List[float], Dict[str, Any]]:
        """
        功能简介:
            对筛选出的“平面点”做 DBSCAN，并对每个 cluster 计算 α，得到分布 T_j。

        输入:
            idx_plane (np.ndarray): (K,) 在全局点云中的索引

        输出:
            alphas (List[float]): 所有有效 cluster 的 α 列表
            info (Dict): 统计信息（簇数、噪声点数等）
        """
        xyz_sub = self.xyz[idx_plane]
        labels = self._DbscanClusters(xyz_sub)

        alphas: List[float] = []
        valid_cluster_ids = [cid for cid in np.unique(labels) if cid != -1]

        noise_count = int(np.sum(labels == -1))
        used_clusters = 0

        for cid in valid_cluster_ids:
            loc = np.where(labels == cid)[0]
            if loc.size < self.cfg.min_cluster_size:
                continue
            normals_cluster = self.normals[idx_plane[loc]]
            alpha = self._ComputeAlphaForCluster(normals_cluster)
            alphas.append(alpha)
            used_clusters += 1

        info = {
            "n_plane_points": int(idx_plane.size),
            "n_clusters_raw": int(len(valid_cluster_ids)),
            "n_clusters_used": int(used_clusters),
            "n_noise": noise_count,
        }
        return alphas, info

    # -------- 分布差异度量 --------

    def _DiffDistributions(self, a: List[float], b: List[float]) -> float:
        """
        功能简介:
            计算两轮 α 分布差异 diff(Tj, Tj-1)。

        实现思路:
            - wasserstein: 1D Earth Mover's Distance（不依赖 bins）
            - hist_l1: 统一 bins 的直方图 L1 距离
            - ks: Kolmogorov-Smirnov 统计量（不依赖 bins）

        输入:
            a, b: α 列表

        输出:
            diff (float)
        """
        if len(a) == 0 or len(b) == 0:
            return float("inf")

        aa = np.asarray(a, dtype=np.float64)
        bb = np.asarray(b, dtype=np.float64)

        metric = self.cfg.diff_metric.lower()

        if metric == "wasserstein":
            # 1D Wasserstein distance：实现一个轻量版本（排序后积分）
            aa = np.sort(aa)
            bb = np.sort(bb)
            n = aa.size
            m = bb.size
            # 对齐到同一分位点
            q = np.linspace(0.0, 1.0, num=max(n, m), endpoint=True)
            aa_q = np.quantile(aa, q)
            bb_q = np.quantile(bb, q)
            return float(np.mean(np.abs(aa_q - bb_q)))

        if metric == "hist_l1":
            lo = float(min(aa.min(), bb.min()))
            hi = float(max(aa.max(), bb.max()))
            if math.isclose(lo, hi):
                return 0.0
            ha, _ = np.histogram(aa, bins=self.cfg.hist_bins, range=(lo, hi), density=True)
            hb, _ = np.histogram(bb, bins=self.cfg.hist_bins, range=(lo, hi), density=True)
            return float(np.sum(np.abs(ha - hb)) / self.cfg.hist_bins)

        if metric == "ks":
            # KS：最大 CDF 差
            aa = np.sort(aa)
            bb = np.sort(bb)
            grid = np.unique(np.concatenate([aa, bb]))
            cdf_a = np.searchsorted(aa, grid, side="right") / aa.size
            cdf_b = np.searchsorted(bb, grid, side="right") / bb.size
            return float(np.max(np.abs(cdf_a - cdf_b)))

        raise ValueError("diff_metric 必须是 'wasserstein'/'hist_l1'/'ks' 之一。")

    # -------- 可视化输出 --------

    def _PlotHistory(self, history: List[Dict[str, Any]]) -> None:
        # 1) diff vs iter
        iters = [h["iter"] for h in history]
        curs = [h["cur"] for h in history]
        diffs = [h["diff"] for h in history]

        plt.figure()
        plt.plot(iters, diffs, marker="o")
        plt.xlabel("iteration")
        plt.ylabel("diff(Tj, Tj-1)")
        plt.title(f"Distribution diff metric = {self.cfg.diff_metric}")
        plt.tight_layout()
        plt.savefig(os.path.join(self.cfg.out_dir, "diff_vs_iter.png"), dpi=200)
        plt.close()

        # 2) cur vs iter
        plt.figure()
        plt.plot(iters, curs, marker="o")
        plt.xlabel("iteration")
        plt.ylabel("curvature threshold (cur)")
        plt.title("cur threshold vs iteration")
        plt.tight_layout()
        plt.savefig(os.path.join(self.cfg.out_dir, "cur_vs_iter.png"), dpi=200)
        plt.close()

        # 3) α 分布的直方图叠加（取最后两轮）
        if len(history) >= 2:
            a1 = history[-2]["alphas"]
            a2 = history[-1]["alphas"]
            if len(a1) > 0 and len(a2) > 0:
                aa = np.asarray(a1, dtype=np.float64)
                bb = np.asarray(a2, dtype=np.float64)
                lo = float(min(aa.min(), bb.min()))
                hi = float(max(aa.max(), bb.max()))
                if not math.isclose(lo, hi):
                    plt.figure()
                    plt.hist(aa, bins=self.cfg.hist_bins, range=(lo, hi), density=True, alpha=0.5, label="T_{j-1}")
                    plt.hist(bb, bins=self.cfg.hist_bins, range=(lo, hi), density=True, alpha=0.5, label="T_{j}")
                    plt.xlabel("alpha (sum of normal angles)")
                    plt.ylabel("density")
                    plt.title("Alpha distributions (last two iterations)")
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.cfg.out_dir, "alpha_dist_last_two.png"), dpi=200)
                    plt.close()

    def _SaveHistory(self, history: List[Dict[str, Any]]) -> None:
        # 保存为 txt，避免引入额外依赖
        out_txt = os.path.join(self.cfg.out_dir, "history.txt")
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(f"point_path={self.cfg.point_path}\n")
            f.write(f"cur0={self.cfg.cur0}, eta={self.cfg.eta}, max_iters={self.cfg.max_iters}\n")
            f.write(f"dbscan_eps={self.cfg.dbscan_eps}, dbscan_min_samples={self.cfg.dbscan_min_samples}\n")
            f.write(f"n_pairs={self.cfg.n_pairs}, angle_unit={self.cfg.angle_unit}\n")
            f.write(f"diff_metric={self.cfg.diff_metric}, diff_tol={self.cfg.diff_tol}, patience={self.cfg.patience}\n\n")

            for h in history:
                f.write(
                    f"iter={h['iter']:03d} | cur={h['cur']:.6f} | diff={h['diff']:.6f} | "
                    f"plane_pts={h['info']['n_plane_points']} | clusters_used={h['info']['n_clusters_used']} | noise={h['info']['n_noise']}\n"
                )
                if len(h["alphas"]) > 0:
                    a = np.asarray(h["alphas"], dtype=np.float64)
                    f.write(f"  alpha: n={a.size}, mean={a.mean():.3f}, std={a.std():.3f}, min={a.min():.3f}, max={a.max():.3f}\n")
                else:
                    f.write("  alpha: n=0\n")
                f.write("\n")

    # -------- 主流程 --------

    def Run(self) -> Tuple[float, List[Dict[str, Any]]]:
        if self.xyz is None or self.normals is None or self.curvature is None:
            self.PrepareData()

        history: List[Dict[str, Any]] = []
        stable_count = 0

        prev_alphas: Optional[List[float]] = None
        best_cur: float = self.cfg.cur0  # 默认

        for it in range(self.cfg.max_iters):
            cur = self.cfg.cur0 - it * self.cfg.eta

            with StepTimer(self.logger, f"迭代 it={it}, cur={cur:.6f}"):
                idx_plane = np.where(self.curvature <= cur)[0]

                # 候选平面点过少则停止
                if idx_plane.size < self.cfg.min_points_for_plane:
                    self.logger.warning(
                        f"候选平面点不足({idx_plane.size} < {self.cfg.min_points_for_plane})，提前停止。"
                    )
                    break

                alphas, info = self._BuildAlphaDistribution(idx_plane)

                # 计算 diff
                if prev_alphas is None:
                    diff = float("inf")
                else:
                    diff = self._DiffDistributions(alphas, prev_alphas)

                record = {
                    "iter": it,
                    "cur": float(cur),
                    "alphas": alphas,
                    "info": info,
                    "diff": float(diff),
                }
                history.append(record)

                # 收敛检查（从第二轮开始才有意义）
                if prev_alphas is not None and np.isfinite(diff) and diff < self.cfg.diff_tol:
                    stable_count += 1
                    self.logger.info(f"diff<{self.cfg.diff_tol}，stable_count={stable_count}/{self.cfg.patience}")
                else:
                    stable_count = 0

                # 按你的定义：停止时“当前cur已减过步长”作为最佳
                if stable_count >= self.cfg.patience:
                    best_cur = float(cur)
                    self.logger.info(f"满足收敛条件，停止；best_cur={best_cur:.6f}")
                    break

                prev_alphas = alphas
                best_cur = float(cur)  # 若未收敛，继续推进，best_cur随迭代更新为当前cur

        with StepTimer(self.logger, "保存与可视化输出"):
            self._SaveHistory(history)
            self._PlotHistory(history)

        return best_cur, history


# =========================
# 直接运行示例
# =========================
if __name__ == "__main__":
    cfg = CurvatureDbscanOptimizeConfig(
        point_path=r"D:\Research\20250313_RockFractureSeg\Code\RockDiscontinuity\data\rock_data\Rock_GLS4_part1_localize_0.05m.ply",
        out_dir=r"D:\Research\20250313_RockFractureSeg\Code\RockDiscontinuity\result/curvature_opt_out",

        cur0=0.03,
        eta=0.001,
        max_iters=30,

        min_points_for_plane=5000,

        dbscan_eps=0.05,
        dbscan_min_samples=20,
        min_cluster_size=100,

        n_pairs=500,
        angle_unit="deg",

        diff_metric="wasserstein",  # 推荐：不依赖 bins
        diff_tol=0.02,
        patience=2,

        random_seed=7,

        normal_est_k=30,
        curvature_est_k=30
    )

    opt = CurvatureThresholdOptimizer(cfg)
    best_cur, history = opt.Run()
    print(f"best_cur={best_cur:.6f}, iters={len(history)}")
