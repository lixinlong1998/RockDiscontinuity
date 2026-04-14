# -*- coding: utf-8 -*-
"""
书本状二面体两块 1m×1m 正方形平面（共享一条边），采样为带噪点云。
以公共边中点为核心点，在半径 R 球内随机采样 N 对点并计算法向差，统计分布与指标。
新增：用最小生成树（MST）统一法向指向：
- 用公共边“非锐角一侧”参考方向 v_out 先定向种子点法向
- 沿 MST 传递，把每个点法向翻转到与父节点同向（dot>0）

依赖：numpy, matplotlib, pandas, scikit-learn（用于kNN与PCA法向估计）
安装：pip install numpy matplotlib pandas scikit-learn
"""

import os
import json
import time
import math
import heapq
import logging
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from sklearn.neighbors import NearestNeighbors

    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


# ----------------------------
# [log] Logger
# ----------------------------
def SetupLogger(name: str = "BookDihedralSim") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger


class Timer:
    def __init__(self, logger: logging.Logger, msg: str):
        self.logger = logger
        self.msg = msg
        self.t0 = None

    def __enter__(self):
        self.t0 = time.time()
        self.logger.info(f"[Timer] START - {self.msg}")
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = time.time() - self.t0
        self.logger.info(f"[Timer] END   - {self.msg} | {dt:.3f} s")


# ----------------------------
# Config
# ----------------------------
@dataclass
class SimConfig:
    # [基本几何]
    square_size_m: float = 1.0  # 正方形边长（m）
    sample_spacing_m: float = 0.05  # 平均采样间距（m），近似用网格步长实现

    # [二面角]
    dihedral_angle_deg: float = 90.0  # 二面角 θ（度），绕公共边（y轴）旋转

    # [噪声]
    noise_sigma_m: float = 0.005  # 坐标高斯噪声标准差（m）

    # [邻域与采样]
    neighborhood_radius_m: float = 0.5  # 核心点邻域球半径 R（m）
    random_pair_samples: int = 500  # 随机点对采样次数 N
    require_distinct_points: bool = True  # 点对是否要求 i!=j

    # [法向估计]
    estimate_normals: bool = True  # True：用kNN PCA估计法向；False：直接用真值平面法向
    knn_k: int = 30  # kNN邻居数（法向估计）

    # [MST 法向统一]
    mst_orient_normals: bool = True  # 是否用 MST 统一法向
    mst_knn_graph_k: int = 12  # 构图用kNN（MST边候选）
    mst_seed_topk: int = 50  # 在距公共边最近的 topK 点中选种子（增强稳定性）
    mst_flip_global_by_vout: bool = True  # MST后若整体与v_out相反则全局翻转

    # [输出]
    workspace: str = r"D:\workspace_book_dihedral_sim"
    random_seed: int = 42


# ----------------------------
# Core simulator
# ----------------------------
class BookDihedralSimulator:
    """
    书本状二面体：
    - 平面A：z=0，(x,y) in [0,1]×[-0.5,0.5]，共享边为 x=0, z=0, y∈[-0.5,0.5]
    - 平面B：将平面A绕 y轴 旋转 θ，仍取 u∈[0,1], v∈[-0.5,0.5]，旋转后得到 (x,y,z)

    核心点：共享边中点 (0,0,0)
    """

    def __init__(self, cfg: SimConfig, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger
        self.rng = np.random.default_rng(cfg.random_seed)

        self.out_dir = None

        # data holders
        self.points = None  # (M,3) noisy points
        self.points_clean = None  # (M,3) clean points
        self.plane_id = None  # (M,) 0 for A, 1 for B
        self.truth_normals = None  # (M,3) true plane normals (unsigned)
        self.normals = None  # (M,3) normals used for pair metrics (estimated or truth, oriented)
        self.core_point = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        self.in_sphere_idx = None  # indices within radius R
        self.pair_df = None  # per-pair metrics
        self.summary_df = None  # summary metrics (1 row)
        self._summary_dict = None

    # ----------------------------
    # Geometry + sampling
    # ----------------------------
    def _GenerateSquareGridUV(self):
        """生成平面参数网格 u∈[0,L], v∈[-L/2, L/2]，步长 s。"""
        L = float(self.cfg.square_size_m)
        s = float(self.cfg.sample_spacing_m)

        # [关键变量打印]
        self.logger.info(f"[MON] square_size_m={L:.3f}, sample_spacing_m={s:.3f}")

        u = np.arange(0.0, L + 1e-9, s)
        v = np.arange(-0.5 * L, 0.5 * L + 1e-9, s)
        uu, vv = np.meshgrid(u, v, indexing="xy")
        uv = np.stack([uu.reshape(-1), vv.reshape(-1)], axis=1)  # (K,2)
        return uv

    def _RotationY(self, angle_deg: float):
        a = math.radians(angle_deg)
        c, s = math.cos(a), math.sin(a)
        R = np.array([[c, 0.0, s],
                      [0.0, 1.0, 0.0],
                      [-s, 0.0, c]], dtype=np.float64)
        return R

    def _ComputeVoutNonAcuteSide(self, dihedral_angle_deg: float) -> np.ndarray:
        """
        计算公共边（y轴）横向剖面内“非锐角一侧”的参考方向 v_out。
        这里按你前面定义：锐角为“内部”，v_out 指向“非锐角(外部)侧”。

        页内正向：
        tA = (1,0,0)
        tB = R_y(θ) * (1,0,0) = (cosθ,0,-sinθ)

        内部(锐角)方向近似：v_in ∝ tA + tB
        外部(非锐角)方向：v_out = -normalize(v_in)
        """
        theta = math.radians(float(dihedral_angle_deg))
        t_a = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        t_b = np.array([math.cos(theta), 0.0, -math.sin(theta)], dtype=np.float64)
        v_in = t_a + t_b
        nrm = np.linalg.norm(v_in)
        if nrm < 1e-9:
            # θ≈180°退化：给一个稳定方向
            v_in = np.array([0.0, 0.0, -1.0], dtype=np.float64)
            nrm = 1.0
        v_in = v_in / nrm
        v_out = -v_in
        return v_out

    def GeneratePointCloud(self):
        with Timer(self.logger, "Generate clean point cloud (two squares)"):
            uv = self._GenerateSquareGridUV()
            u = uv[:, 0]
            v = uv[:, 1]

            # Plane A (x=u, y=v, z=0)
            pts_a = np.stack([u, v, np.zeros_like(u)], axis=1)

            # Plane B: rotate the same param plane about y-axis by θ
            R = self._RotationY(self.cfg.dihedral_angle_deg)
            pts_b0 = np.stack([u, v, np.zeros_like(u)], axis=1)
            pts_b = (R @ pts_b0.T).T

            points_clean = np.vstack([pts_a, pts_b])
            plane_id = np.concatenate([np.zeros(len(pts_a), dtype=np.int32),
                                       np.ones(len(pts_b), dtype=np.int32)], axis=0)

            # True (unsigned) normals per plane
            n_a = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            n_b = (R @ n_a.reshape(3, 1)).reshape(3)
            truth_normals = np.vstack([np.tile(n_a, (len(pts_a), 1)),
                                       np.tile(n_b, (len(pts_b), 1))])

            self.points_clean = points_clean
            self.plane_id = plane_id
            self.truth_normals = truth_normals

            self.logger.info(f"[MON] Clean points total={points_clean.shape[0]}, "
                             f"PlaneA={len(pts_a)}, PlaneB={len(pts_b)}")
            self.logger.info(f"[MON] dihedral_angle_deg={self.cfg.dihedral_angle_deg:.2f}, "
                             f"nA={n_a.tolist()}, nB={n_b.tolist()}")

        with Timer(self.logger, "Add Gaussian noise to points"):
            sigma = float(self.cfg.noise_sigma_m)
            noise = self.rng.normal(loc=0.0, scale=sigma, size=self.points_clean.shape)
            self.points = self.points_clean + noise
            self.logger.info(f"[MON] noise_sigma_m={sigma:.6f} | "
                             f"noise_abs_mean={np.mean(np.linalg.norm(noise, axis=1)):.6f} m")

    # ----------------------------
    # Neighborhood selection
    # ----------------------------
    def SelectPointsInSphere(self):
        with Timer(self.logger, "Select points inside sphere neighborhood"):
            R = float(self.cfg.neighborhood_radius_m)
            d = np.linalg.norm(self.points - self.core_point.reshape(1, 3), axis=1)
            idx = np.where(d <= R)[0]
            self.in_sphere_idx = idx

            self.logger.info(f"[MON] neighborhood_radius_m={R:.3f} | "
                             f"points_in_sphere={len(idx)} / {len(self.points)}")

            if len(idx) < 10:
                raise RuntimeError("球内点数太少，无法进行稳定统计；请增大R或减小采样间距。")

    # ----------------------------
    # Normal estimation + MST orientation
    # ----------------------------
    def _EstimateNormalsKnnPca(self, pts: np.ndarray, k: int) -> np.ndarray:
        """
        用kNN PCA估计法向：
        - 对每个点找k邻居
        - 协方差最小特征向量作为法向（符号未定）
        """
        if not SKLEARN_OK:
            raise RuntimeError(
                "未检测到 scikit-learn，无法估计法向。请 pip install scikit-learn 或设 estimate_normals=False。")

        nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(pts)
        _, indices = nbrs.kneighbors(pts)  # (M,k)

        normals = np.zeros_like(pts)
        for i in range(pts.shape[0]):
            neigh = pts[indices[i]]  # (k,3)
            mu = neigh.mean(axis=0, keepdims=True)
            X = neigh - mu
            C = (X.T @ X) / max(len(neigh) - 1, 1)
            w, V = np.linalg.eigh(C)
            n = V[:, np.argmin(w)]
            n = n / (np.linalg.norm(n) + 1e-12)
            normals[i] = n
        return normals

    def _BuildKnnGraph(self, pts: np.ndarray, k: int):
        """
        构建无向kNN图（用于MST候选边）。
        返回：adj list，adj[i]=[(j, w_ij), ...]
        """
        if not SKLEARN_OK:
            raise RuntimeError("需要 scikit-learn 的 NearestNeighbors 来构图。")

        k = max(2, int(k))
        k = min(k, max(2, pts.shape[0] - 1))

        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(pts)
        dists, inds = nbrs.kneighbors(pts)  # 含自身(0)

        M = pts.shape[0]
        adj = [[] for _ in range(M)]
        for i in range(M):
            for t in range(1, inds.shape[1]):  # 跳过自身
                j = int(inds[i, t])
                w = float(dists[i, t])
                # 无向边：两边都加（允许重复，后面Prim不敏感）
                adj[i].append((j, w))
                adj[j].append((i, w))
        return adj

    def _PrimMSTParents(self, adj, root: int):
        """
        Prim 算法生成MST（在给定邻接表的连通图上）。
        返回 parent 数组：parent[root] = -1，其余为父节点索引。
        若图不连通，root所在分量会被覆盖，其余保持 -1。
        """
        M = len(adj)
        visited = [False] * M
        parent = [-1] * M
        key = [float("inf")] * M

        key[root] = 0.0
        heap = [(0.0, root)]
        while heap:
            w_u, u = heapq.heappop(heap)
            if visited[u]:
                continue
            visited[u] = True
            for v, w_uv in adj[u]:
                if not visited[v] and w_uv < key[v]:
                    key[v] = w_uv
                    parent[v] = u
                    heapq.heappush(heap, (w_uv, v))
        return parent, visited

    def _OrientNormalsByMST(self, pts: np.ndarray, normals: np.ndarray, dihedral_angle_deg: float):
        """
        用MST统一 normals 的方向：
        1) 以距公共边最近的点作为候选，取 topK 中与 v_out 投影最强者为种子
        2) 用 v_out（非锐角侧参考方向）确定种子法向符号（dot(n, v_out)>0）
        3) 沿 MST 传播：child 若与 parent 点积<0则翻转
        4) 可选：若整体与 v_out 多数相反，则全局翻转
        """
        M = pts.shape[0]
        if M < 3:
            return normals

        v_out = self._ComputeVoutNonAcuteSide(dihedral_angle_deg=dihedral_angle_deg)
        v_out = v_out / (np.linalg.norm(v_out) + 1e-12)

        # 距公共边(hinge line: x=0,z=0)距离：sqrt(x^2+z^2)
        d_edge = np.sqrt(pts[:, 0] ** 2 + pts[:, 2] ** 2)
        topk = min(int(self.cfg.mst_seed_topk), M)
        cand = np.argsort(d_edge)[:topk]

        # 在候选里选一个与 v_out |dot| 最大者作为seed（避免法向恰好近似垂直v_out导致不稳）
        dots = np.abs(normals[cand] @ v_out.reshape(3, 1)).reshape(-1)
        seed_local = int(cand[int(np.argmax(dots))])

        # 先用 v_out 定向 seed：dot<0 翻转
        if float(np.dot(normals[seed_local], v_out)) < 0.0:
            normals[seed_local] *= -1.0

        self.logger.info(f"[MON] MST seed_local={seed_local} | "
                         f"edge_dist={float(d_edge[seed_local]):.4f} | "
                         f"dot(seed,v_out)={float(np.dot(normals[seed_local], v_out)):.4f} | "
                         f"v_out={v_out.tolist()}")

        # 构图 + MST
        adj = self._BuildKnnGraph(pts, k=int(self.cfg.mst_knn_graph_k))
        parent, visited = self._PrimMSTParents(adj, root=seed_local)

        # 建children表用于遍历
        children = [[] for _ in range(M)]
        for v in range(M):
            p = parent[v]
            if p >= 0:
                children[p].append(v)

        # DFS/BFS传播翻转
        stack = [seed_local]
        orient_cnt = 0
        flip_cnt = 0
        while stack:
            u = stack.pop()
            for v in children[u]:
                # 保证 n_v 与 n_u 同向
                if float(np.dot(normals[v], normals[u])) < 0.0:
                    normals[v] *= -1.0
                    flip_cnt += 1
                orient_cnt += 1
                stack.append(v)

        self.logger.info(f"[MON] MST orient done | visited={int(np.sum(visited))}/{M} | "
                         f"propagated_edges={orient_cnt} | flips={flip_cnt}")

        # 可选：整体与 v_out 多数相反 -> 全局翻转
        if self.cfg.mst_flip_global_by_vout:
            sign = (normals @ v_out.reshape(3, 1)).reshape(-1)
            ratio_pos = float(np.mean(sign > 0.0))
            if ratio_pos < 0.5:
                normals *= -1.0
                self.logger.info(f"[MON] Global flip by v_out (ratio_pos={ratio_pos:.3f} < 0.5)")
            else:
                self.logger.info(f"[MON] Keep orientation (ratio_pos={ratio_pos:.3f} >= 0.5)")

        # 归一化
        normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)
        return normals

    def PrepareNormals(self):
        """
        准备每点法向：
        - estimate_normals=False：直接使用真值法向（但符号仍不一定统一；这里也可走MST统一）
        - estimate_normals=True：kNN-PCA估计法向（符号未定），再用MST统一
        """
        if not self.cfg.estimate_normals:
            normals = self.truth_normals.copy()
            self.logger.info("[MON] Using truth plane normals (estimate_normals=False).")
        else:
            with Timer(self.logger, "Estimate normals by kNN-PCA"):
                if not SKLEARN_OK:
                    raise RuntimeError("estimate_normals=True 但未安装 scikit-learn。")
                k = int(self.cfg.knn_k)
                k = max(10, k)
                k = min(k, max(10, self.points.shape[0] - 1))
                self.logger.info(f"[MON] estimate_normals=True | knn_k={k} | sklearn_ok={SKLEARN_OK}")
                normals = self._EstimateNormalsKnnPca(self.points, k=k)

        # 用 MST 统一法向指向：以 v_out（非锐角侧参考方向）定种子，并沿MST传播
        if self.cfg.mst_orient_normals:
            with Timer(self.logger, "Orient normals by MST propagation"):
                normals = self._OrientNormalsByMST(
                    pts=self.points,
                    normals=normals,
                    dihedral_angle_deg=self.cfg.dihedral_angle_deg
                )

        self.normals = normals

    # ----------------------------
    # Pair sampling + metrics
    # ----------------------------
    def SamplePairsAndComputeMetrics(self):
        with Timer(self.logger, "Sample random pairs and compute normal-difference metrics"):
            idx = self.in_sphere_idx
            N = int(self.cfg.random_pair_samples)
            if N <= 0:
                raise ValueError("random_pair_samples 必须 > 0")

            M = len(idx)
            self.logger.info(f"[MON] random_pair_samples={N} | candidate_points_in_sphere={M}")

            i_list = self.rng.integers(low=0, high=M, size=N, endpoint=False)
            j_list = self.rng.integers(low=0, high=M, size=N, endpoint=False)

            if self.cfg.require_distinct_points:
                same = (i_list == j_list)
                round_cnt = 0
                while np.any(same) and round_cnt < 10:
                    j_list[same] = self.rng.integers(low=0, high=M, size=int(np.sum(same)), endpoint=False)
                    same = (i_list == j_list)
                    round_cnt += 1
                self.logger.info(
                    f"[MON] require_distinct_points=True | fixed_rounds={round_cnt} | still_same={int(np.sum(same))}")

            pi = idx[i_list]
            pj = idx[j_list]

            ni = self.normals[pi]
            nj = self.normals[pj]

            diff_norm = np.linalg.norm(ni - nj, axis=1)

            dot = np.sum(ni * nj, axis=1)
            dot = np.clip(dot, -1.0, 1.0)
            angle_deg = np.degrees(np.arccos(dot))

            same_plane = (self.plane_id[pi] == self.plane_id[pj]).astype(np.int32)

            df = pd.DataFrame({
                "pair_id": np.arange(N, dtype=np.int32),
                "idx_i": pi.astype(np.int32),
                "idx_j": pj.astype(np.int32),
                "plane_i": self.plane_id[pi].astype(np.int32),
                "plane_j": self.plane_id[pj].astype(np.int32),
                "same_plane": same_plane,
                "dot": dot.astype(np.float64),
                "angle_deg": angle_deg.astype(np.float64),
                "diff_norm": diff_norm.astype(np.float64),
            })

            self.pair_df = df

            def _Summ(x: np.ndarray):
                return {
                    "mean": float(np.mean(x)),
                    "std": float(np.std(x)),
                    "median": float(np.median(x)),
                    "p05": float(np.percentile(x, 5)),
                    "p25": float(np.percentile(x, 25)),
                    "p75": float(np.percentile(x, 75)),
                    "p95": float(np.percentile(x, 95)),
                    "min": float(np.min(x)),
                    "max": float(np.max(x)),
                }

            all_ang = df["angle_deg"].to_numpy()
            all_dn = df["diff_norm"].to_numpy()

            ang_same = df.loc[df["same_plane"] == 1, "angle_deg"].to_numpy()
            ang_diff = df.loc[df["same_plane"] == 0, "angle_deg"].to_numpy()
            dn_same = df.loc[df["same_plane"] == 1, "diff_norm"].to_numpy()
            dn_diff = df.loc[df["same_plane"] == 0, "diff_norm"].to_numpy()

            summary = {
                "N_pairs": int(N),
                "R_m": float(self.cfg.neighborhood_radius_m),
                "theta_deg": float(self.cfg.dihedral_angle_deg),
                "noise_sigma_m": float(self.cfg.noise_sigma_m),
                "points_total": int(self.points.shape[0]),
                "points_in_sphere": int(len(self.in_sphere_idx)),
                "ratio_same_plane": float(np.mean(df["same_plane"].to_numpy())),
                "angle_deg_all": _Summ(all_ang),
                "diff_norm_all": _Summ(all_dn),
            }

            summary["angle_deg_same_plane"] = _Summ(ang_same) if len(ang_same) > 0 else None
            summary["diff_norm_same_plane"] = _Summ(dn_same) if len(dn_same) > 0 else None
            summary["angle_deg_diff_plane"] = _Summ(ang_diff) if len(ang_diff) > 0 else None
            summary["diff_norm_diff_plane"] = _Summ(dn_diff) if len(dn_diff) > 0 else None

            self.summary_df = pd.DataFrame([{
                "N_pairs": summary["N_pairs"],
                "points_total": summary["points_total"],
                "points_in_sphere": summary["points_in_sphere"],
                "ratio_same_plane": summary["ratio_same_plane"],
                "angle_mean_deg": summary["angle_deg_all"]["mean"],
                "angle_std_deg": summary["angle_deg_all"]["std"],
                "angle_p50_deg": summary["angle_deg_all"]["median"],
                "angle_p95_deg": summary["angle_deg_all"]["p95"],
                "diffnorm_mean": summary["diff_norm_all"]["mean"],
                "diffnorm_std": summary["diff_norm_all"]["std"],
                "diffnorm_p50": summary["diff_norm_all"]["median"],
                "diffnorm_p95": summary["diff_norm_all"]["p95"],
            }])

            self._summary_dict = summary

            self.logger.info(f"[MON] ratio_same_plane={summary['ratio_same_plane']:.3f} | "
                             f"angle_mean={summary['angle_deg_all']['mean']:.2f} deg | "
                             f"angle_p95={summary['angle_deg_all']['p95']:.2f} deg")

    # ----------------------------
    # Export
    # ----------------------------
    def _MakeOutputDir(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(self.cfg.workspace, ts + f"_{self.cfg.dihedral_angle_deg}")
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.logger.info(f"[MON] Output dir: {out_dir}")

    def ExportAll(self):
        self._MakeOutputDir()

        with Timer(self.logger, "Export parameters and tables"):
            with open(os.path.join(self.out_dir, "parameters.json"), "w", encoding="utf-8") as f:
                json.dump(asdict(self.cfg), f, ensure_ascii=False, indent=2)

            pts_df = pd.DataFrame({
                "x": self.points[:, 0],
                "y": self.points[:, 1],
                "z": self.points[:, 2],
                "plane_id": self.plane_id,
                "nx_truth": self.truth_normals[:, 0],
                "ny_truth": self.truth_normals[:, 1],
                "nz_truth": self.truth_normals[:, 2],
                "nx": self.normals[:, 0],
                "ny": self.normals[:, 1],
                "nz": self.normals[:, 2],
                "in_sphere": np.isin(np.arange(len(self.points)), self.in_sphere_idx).astype(np.int32),
            })
            pts_df.to_csv(os.path.join(self.out_dir, "point_cloud.csv"), index=False, encoding="utf-8-sig")

            self.pair_df.to_csv(os.path.join(self.out_dir, "pair_metrics.csv"), index=False, encoding="utf-8-sig")
            self.summary_df.to_csv(os.path.join(self.out_dir, "summary_table.csv"), index=False, encoding="utf-8-sig")

            with open(os.path.join(self.out_dir, "summary.json"), "w", encoding="utf-8") as f:
                json.dump(self._summary_dict, f, ensure_ascii=False, indent=2)

        with Timer(self.logger, "Export plots"):
            self._Plot3DPointCloud()
            self._PlotHistogramAndCdf(metric="angle_deg", xlabel="Angle between normals (deg)")
            self._PlotHistogramAndCdf(metric="diff_norm", xlabel="||n1 - n2||")

    def _Plot3DPointCloud(self):
        pts = self.points
        pid = self.plane_id
        ins = np.isin(np.arange(len(pts)), self.in_sphere_idx)

        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")

        a = (pid == 0)
        b = (pid == 1)

        ax.scatter(pts[a, 0], pts[a, 1], pts[a, 2], s=2, alpha=0.6, label="Plane A")
        ax.scatter(pts[b, 0], pts[b, 1], pts[b, 2], s=2, alpha=0.6, label="Plane B")
        ax.scatter(pts[ins, 0], pts[ins, 1], pts[ins, 2], s=6, alpha=0.9, label="In sphere")

        ax.scatter([self.core_point[0]], [self.core_point[1]], [self.core_point[2]], s=60, marker="x", label="Core")

        ax.set_title("Noisy point cloud (book dihedral)")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.legend(loc="upper right")

        fig.tight_layout()
        fig.savefig(os.path.join(self.out_dir, "point_cloud_3d.png"), dpi=200)
        plt.close(fig)

    def _PlotHistogramAndCdf(self, metric: str, xlabel: str):
        df = self.pair_df.copy()
        x_all = df[metric].to_numpy()
        x_same = df.loc[df["same_plane"] == 1, metric].to_numpy()
        x_diff = df.loc[df["same_plane"] == 0, metric].to_numpy()

        fig = plt.figure(figsize=(8, 5))
        bins = 30

        plt.hist(x_all, bins=bins, alpha=0.5, label="All")
        if len(x_same) > 0:
            plt.hist(x_same, bins=bins, alpha=0.5, label="Same plane")
        if len(x_diff) > 0:
            plt.hist(x_diff, bins=bins, alpha=0.5, label="Different planes")

        plt.title(f"Histogram - {metric}")
        plt.xlabel(xlabel)
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(self.out_dir, f"hist_{metric}.png"), dpi=200)
        plt.close(fig)

        fig = plt.figure(figsize=(8, 5))
        xs = np.sort(x_all)
        ys = np.linspace(0, 1, len(xs), endpoint=True)
        plt.plot(xs, ys, label="All")

        if len(x_same) > 0:
            xs2 = np.sort(x_same)
            ys2 = np.linspace(0, 1, len(xs2), endpoint=True)
            plt.plot(xs2, ys2, label="Same plane")

        if len(x_diff) > 0:
            xs3 = np.sort(x_diff)
            ys3 = np.linspace(0, 1, len(xs3), endpoint=True)
            plt.plot(xs3, ys3, label="Different planes")

        plt.title(f"CDF - {metric}")
        plt.xlabel(xlabel)
        plt.ylabel("CDF")
        plt.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(self.out_dir, f"cdf_{metric}.png"), dpi=200)
        plt.close(fig)

    # ----------------------------
    # Pipeline
    # ----------------------------
    def Run(self):
        self.logger.info("========== BookDihedralSimulator RUN ==========")
        self.logger.info(f"[MON] Config: {self.cfg}")

        self.GeneratePointCloud()
        self.SelectPointsInSphere()
        self.PrepareNormals()
        self.SamplePairsAndComputeMetrics()
        self.ExportAll()

        self.logger.info("========== DONE ==========")
        self.logger.info(f"[MON] Results saved to: {self.out_dir}")


# ----------------------------
# Main
# ----------------------------
def main():
    logger = SetupLogger()
    for angle_deg in range(0, 181, 10):
        cfg = SimConfig(
            dihedral_angle_deg=angle_deg,  # θ
            neighborhood_radius_m=0.5,  # R
            random_pair_samples=500,  # N
            noise_sigma_m=0.005,  # 噪声 (m)
            sample_spacing_m=0.05,  # 平均采样间距 (m)

            estimate_normals=True,  # kNN-PCA 法向估计
            knn_k=30,

            mst_orient_normals=True,  # 用 MST 统一法向指向
            mst_knn_graph_k=12,
            mst_seed_topk=50,
            mst_flip_global_by_vout=True,

            workspace=r"D:\workspace_book_dihedral_sim",
            random_seed=42,
        )

        os.makedirs(cfg.workspace, exist_ok=True)

        sim = BookDihedralSimulator(cfg, logger)
        sim.Run()


if __name__ == "__main__":
    main()
