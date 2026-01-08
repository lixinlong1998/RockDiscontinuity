# -*- coding: utf-8 -*-
import os
import re
import time
import math
import logging
from typing import List, Optional, Any, Tuple

import numpy as np

# =========================
# 用户配置（必须修改）
# =========================
OUTPUT_DIR = r"D:\Research\20250313_RockFractureSeg\Code\RockDiscontinuity\result\cc_pycc_export"
OUTPUT_BASE_NAME = "Rock_GLS4_part1_localize_0.05m_ransac_shapes"

# 原始点云（用于读取曲率标量场并映射到 plane 点）
ORIGINAL_CLOUD_NAME = "Rock_GLS4_part1_localize_0.05m - Cloud"

# 曲率标量场名称候选（会依次尝试）
CURVATURE_SF_CANDIDATES = ["Curvature", "curvature", "curv"]
MAP_CURVATURE_FROM_ORIGINAL = True      # 原始云有曲率就映射；没有则走估计
ESTIMATE_CURVATURE_IF_MISSING = True    # 映射失败则在 plane 内估计
CURVATURE_KNN_K = 30

# 平面法向统一到上半球（nz>=0）
FORCE_UPPER_HEMISPHERE_NORMAL = True

# 画图（CloudCompare 内置 Python 若无 matplotlib 会自动跳过）
PLOT_PLANE_SIZES = True

# 写文件性能：每次缓存多少行再写出
WRITE_BUFFER_LINES = 50000

LOG_LEVEL = logging.INFO


class StepTimer:
    """
    功能简介:
        记录每一步耗时。

    实现思路:
        保存上一次时间戳，Tick 时输出时间差。

    输入变量:
        logger: logging.Logger

    输出变量:
        无
    """
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.t0 = time.perf_counter()

    def Tick(self, step_name: str) -> float:
        t1 = time.perf_counter()
        dt = t1 - self.t0
        self.logger.info(f"[耗时] {step_name}: {dt:.3f} s")
        self.t0 = t1
        return dt


class PyccRansacExporter:
    """
    功能简介:
        在 CloudCompare Python Plugin (pycc) 中，把 Ransac Detected Shapes 下的 Plane_XXXX
        导出为单个 CSV（逐点一行），字段：
        X,Y,Z,nx,ny,nz,R,G,B,DR,DG,DB,Discontinuity_id,Cluster_id,Segment_id,A,B,C,D,RMS,Curvature,DistToPlane

    实现思路:
        1) CC = pycc.GetInstance()，从 CC.getSelectedEntities() 获取用户选中对象
        2) 递归遍历子节点，收集名称匹配 Plane_#### 的 ccPointCloud
        3) 对每个 plane：
           - pts = cloud.points() -> numpy view (N,3)
           - normals/colors 同理，若没有则填充
           - PCA 拟合平面 (A,B,C,D)，并算每点 DistToPlane 与 RMS
           - Curvature：优先从原始点云标量场映射，否则 kNN-PCA 估计，否则 0
        4) 写 points.csv（流式写）和 summary.csv，并输出面片规模图

    输入变量:
        output_dir: str
        base_name: str

    输出变量:
        points_csv_path: str
        summary_csv_path: str
    """

    def __init__(self, output_dir: str, base_name: str):
        import pycc  # noqa
        self.output_dir = output_dir
        self.base_name = base_name
        os.makedirs(self.output_dir, exist_ok=True)

        self.logger = self._BuildLogger()
        self.timer = StepTimer(self.logger)

        # 原始云缓存（曲率映射）
        self.orig_pts = None         # (N,3)
        self.orig_curv = None        # (N,)
        self.orig_kdtree = None      # scipy.spatial.cKDTree 或 sklearn NN

    def _BuildLogger(self) -> logging.Logger:
        logger = logging.getLogger("PyccRansacExporter")
        logger.setLevel(LOG_LEVEL)
        logger.handlers.clear()
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        fh = logging.FileHandler(os.path.join(self.output_dir, f"{self.base_name}.log"), encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        return logger

    # -------------------------
    # pycc：实例、选择、树遍历
    # -------------------------
    def _GetCcInstance(self):
        import pycc
        return pycc.GetInstance()

    def _GetSelectedEntities(self) -> List[Any]:
        """
        功能简介:
            获取 DB 树中当前选中的实体列表（关键：从 CC 实例获取）。

        输出变量:
            selected: List[ccHObject]
        """
        cc = self._GetCcInstance()
        if hasattr(cc, "getSelectedEntities"):
            sel = cc.getSelectedEntities()
            return list(sel) if sel is not None else []
        if hasattr(cc, "GetSelectedEntities"):
            sel = cc.GetSelectedEntities()
            return list(sel) if sel is not None else []
        raise AttributeError("CC 实例上找不到 getSelectedEntities/GetSelectedEntities")

    def _GetDbRoot(self):
        cc = self._GetCcInstance()
        if hasattr(cc, "dbRootObject"):
            return cc.dbRootObject()
        if hasattr(cc, "dbRoot"):
            return cc.dbRoot()
        raise AttributeError("CC 实例上找不到 dbRootObject/dbRoot")

    def _GetName(self, ent: Any) -> str:
        if hasattr(ent, "getName"):
            return str(ent.getName())
        if hasattr(ent, "GetName"):
            return str(ent.GetName())
        return "Unnamed"

    def _GetChildren(self, ent: Any) -> List[Any]:
        if hasattr(ent, "getChildrenNumber") and hasattr(ent, "getChild"):
            n = int(ent.getChildrenNumber())
            return [ent.getChild(i) for i in range(n)]
        if hasattr(ent, "GetChildrenNumber") and hasattr(ent, "GetChild"):
            n = int(ent.GetChildrenNumber())
            return [ent.GetChild(i) for i in range(n)]
        return []

    def _IsPointCloud(self, ent: Any) -> bool:
        # pycc 点云通常具备 points()/size()
        return hasattr(ent, "points") and hasattr(ent, "size")

    def _ExtractPlaneIndex(self, name: str) -> Optional[int]:
        m = re.search(r"plane[_\-\s]*(\d+)", name.lower())
        return int(m.group(1)) if m else None

    def _CollectPlaneClouds(self, roots: List[Any]) -> List[Any]:
        plane_clouds = []
        stack = list(roots)
        while stack:
            ent = stack.pop()
            nm = self._GetName(ent)
            if self._IsPointCloud(ent) and self._ExtractPlaneIndex(nm) is not None:
                plane_clouds.append(ent)
            for ch in self._GetChildren(ent):
                stack.append(ch)

        plane_clouds.sort(key=lambda e: self._ExtractPlaneIndex(self._GetName(e)) or 10**9)
        # 去重
        uniq, seen = [], set()
        for pc in plane_clouds:
            pid = id(pc)
            if pid not in seen:
                uniq.append(pc)
                seen.add(pid)
        return uniq

    # -------------------------
    # 点云数据读取（numpy view）
    # -------------------------
    def _GetPoints(self, cloud: Any) -> np.ndarray:
        pts = cloud.points()
        return np.asarray(pts, dtype=np.float64)

    def _GetColors(self, cloud: Any) -> Optional[np.ndarray]:
        """
        功能简介:
            尝试读取点云颜色，返回 Nx3 int32；若不存在返回 None。

        实现思路:
            1) 优先尝试 cloud.colors()（若绑定提供）
            2) 若无 colors()，但有 hasColors() 且有逐点 getPointColor(i)，则逐点读取

        输出:
            colors: (N,3) 或 None
        """
        # 批量接口
        if hasattr(cloud, "colors"):
            try:
                cs = cloud.colors()
                if cs is None:
                    return None
                cs = np.asarray(cs)
                if cs.ndim == 2 and cs.shape[1] >= 3:
                    return cs[:, :3].astype(np.int32)
            except Exception:
                pass

        # 逐点接口
        if hasattr(cloud, "hasColors") and cloud.hasColors():
            if hasattr(cloud, "getPointColor"):
                n = int(cloud.size())
                cs = np.empty((n, 3), dtype=np.int32)
                for i in range(n):
                    c = cloud.getPointColor(i)
                    cs[i, 0], cs[i, 1], cs[i, 2] = int(c[0]), int(c[1]), int(c[2])
                return cs
        return None

    # ============= 关键修复：法向读取兼容 =============
    def _GetNormals(self, cloud: Any) -> Optional[np.ndarray]:
        """
        功能简介:
            尝试读取点云法向，返回 Nx3 float64；若不存在返回 None。

        实现思路:
            pycc 版本差异较大：有的没有 cloud.normals() 批量接口。
            这里按优先级尝试：
            1) 若存在 cloud.normals() / cloud.getNormals() 等批量接口，直接转 numpy
            2) 若存在逐点接口 cloud.getPointNormal(i) / cloud.getNormal(i)，逐点读取
            3) 都没有则返回 None（外部用拟合平面法向填充）

        输出:
            normals: (N,3) 或 None
        """
        # 是否声明有法向
        if hasattr(cloud, "hasNormals"):
            try:
                if not cloud.hasNormals():
                    return None
            except Exception:
                # 有些版本 hasNormals 存在但会异常，继续尝试
                pass

        # 1) 批量接口尝试（你的版本没有 normals()，但别的版本可能有）
        for fn in ["normals", "getNormals", "Normals", "GetNormals", "normalsArray", "getNormalsArray"]:
            if hasattr(cloud, fn):
                try:
                    arr = getattr(cloud, fn)()
                    if arr is not None:
                        arr = np.asarray(arr, dtype=np.float64)
                        if arr.ndim == 2 and arr.shape[1] >= 3:
                            return arr[:, :3]
                except Exception:
                    pass

        # 2) 逐点接口尝试
        for fn in ["getPointNormal", "getNormal", "GetPointNormal", "GetNormal"]:
            if hasattr(cloud, fn):
                try:
                    n = int(cloud.size())
                    ns = np.empty((n, 3), dtype=np.float64)
                    getter = getattr(cloud, fn)
                    for i in range(n):
                        nn = getter(i)
                        ns[i, 0], ns[i, 1], ns[i, 2] = float(nn[0]), float(nn[1]), float(nn[2])
                    return ns
                except Exception:
                    pass

        # 3) 无法读取
        return None
    # ============= 关键修复结束 =============

    def _TryGetScalarFieldArray(self, cloud: Any, sf_name: str) -> Optional[np.ndarray]:
        if not hasattr(cloud, "getScalarFieldIndexByName"):
            return None
        try:
            idx = int(cloud.getScalarFieldIndexByName(sf_name))
        except Exception:
            return None
        if idx < 0:
            return None
        sf = cloud.getScalarField(idx)
        if not hasattr(sf, "asArray"):
            return None
        arr = sf.asArray()
        return np.asarray(arr, dtype=np.float64)

    def _TryGetCurvatureFromCloud(self, cloud: Any) -> Optional[np.ndarray]:
        for nm in CURVATURE_SF_CANDIDATES:
            arr = self._TryGetScalarFieldArray(cloud, nm)
            if arr is not None:
                return arr
        return None

    # -------------------------
    # 数学：PCA 平面拟合 / 距离 / RMS / 曲率估计
    # -------------------------
    def _FitPlanePca(self, pts: np.ndarray) -> Tuple[np.ndarray, float]:
        centroid = pts.mean(axis=0)
        q = pts - centroid
        cov = (q.T @ q) / max(pts.shape[0] - 1, 1)
        eig_vals, eig_vecs = np.linalg.eigh(cov)
        n = eig_vecs[:, np.argmin(eig_vals)]
        if FORCE_UPPER_HEMISPHERE_NORMAL and n[2] < 0:
            n = -n
        d = -float(np.dot(n, centroid))
        abcd = np.array([n[0], n[1], n[2], d], dtype=np.float64)

        dist = self._DistToPlane(pts, abcd)
        rms = float(np.sqrt(np.mean(dist * dist)))
        return abcd, rms

    def _DistToPlane(self, pts: np.ndarray, abcd: np.ndarray) -> np.ndarray:
        a, b, c, d = abcd
        denom = math.sqrt(a*a + b*b + c*c) + 1e-12
        return np.abs(pts @ np.array([a, b, c]) + d) / denom

    def _BuildNN(self, pts: np.ndarray):
        # 优先 scipy cKDTree；否则 sklearn NearestNeighbors；都没有则返回 None
        try:
            from scipy.spatial import cKDTree
            return ("scipy", cKDTree(pts))
        except Exception:
            pass
        try:
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
            nn.fit(pts)
            return ("sklearn", nn)
        except Exception:
            return None

    def _QueryNN1(self, nn_obj, q_pts: np.ndarray) -> Optional[np.ndarray]:
        if nn_obj is None:
            return None
        kind, obj = nn_obj
        if kind == "scipy":
            _, idx = obj.query(q_pts, k=1)
            return idx
        if kind == "sklearn":
            _, idx = obj.kneighbors(q_pts, n_neighbors=1, return_distance=True)
            return idx.reshape(-1)
        return None

    def _EstimateCurvatureKnnPca(self, pts: np.ndarray, k: int) -> np.ndarray:
        # 曲率估计依赖 kNN；这里仅用 scipy（更快）。没有 scipy 就直接填0。
        try:
            from scipy.spatial import cKDTree
        except Exception:
            self.logger.warning("scipy 不可用，无法 kNN-PCA 估计曲率，Curvature 填0")
            return np.zeros((pts.shape[0],), dtype=np.float64)

        n = pts.shape[0]
        if n == 0:
            return np.zeros((0,), dtype=np.float64)

        tree = cKDTree(pts)
        kk = min(k, n)
        _, idxs = tree.query(pts, k=kk)
        if kk <= 2:
            return np.zeros((n,), dtype=np.float64)

        curv = np.empty((n,), dtype=np.float64)
        for i in range(n):
            nb = pts[idxs[i]]
            c = nb.mean(axis=0)
            q = nb - c
            cov = (q.T @ q) / max(nb.shape[0] - 1, 1)
            eig = np.linalg.eigvalsh(cov)
            s = float(np.sum(eig)) + 1e-12
            curv[i] = float(np.min(eig)) / s
        return curv

    # -------------------------
    # 原始云曲率映射
    # -------------------------
    def _FindEntityByName(self, root: Any, target_name: str) -> Optional[Any]:
        stack = [root]
        while stack:
            ent = stack.pop()
            if self._GetName(ent) == target_name:
                return ent
            for ch in self._GetChildren(ent):
                stack.append(ch)
        return None

    def _PrepareOriginalCurvature(self) -> None:
        if not MAP_CURVATURE_FROM_ORIGINAL:
            return

        root = self._GetDbRoot()
        orig = self._FindEntityByName(root, ORIGINAL_CLOUD_NAME)
        if orig is None or not self._IsPointCloud(orig):
            self.logger.warning(f"未找到原始点云或其不是点云: {ORIGINAL_CLOUD_NAME}，将不做 Curvature 映射")
            return

        curv = self._TryGetCurvatureFromCloud(orig)
        if curv is None:
            self.logger.warning(f"原始点云没有曲率标量场（候选: {CURVATURE_SF_CANDIDATES}），将不做 Curvature 映射")
            return

        pts = np.asarray(orig.points(), dtype=np.float64)
        nn = self._BuildNN(pts)
        if nn is None:
            self.logger.warning("scipy/sklearn 都不可用，无法建立最近邻索引映射曲率")
            return

        self.orig_pts = pts
        self.orig_curv = curv
        self.orig_kdtree = nn
        self.logger.info(f"已加载原始曲率用于映射：N={pts.shape[0]}")

    def _MapCurvatureFromOriginal(self, pts_plane: np.ndarray) -> Optional[np.ndarray]:
        if self.orig_kdtree is None or self.orig_curv is None:
            return None
        idx = self._QueryNN1(self.orig_kdtree, pts_plane)
        if idx is None:
            return None
        return self.orig_curv[idx]

    # -------------------------
    # 主流程
    # -------------------------
    def Run(self) -> None:
        self.logger.info("开始：导出 Ransac Detected Shapes -> CSV（pycc）")

        selected = self._GetSelectedEntities()
        if not selected:
            raise RuntimeError("DB 树未选中任何实体：请选中 Ransac Detected Shapes 组或 Plane_XXXX")

        self._PrepareOriginalCurvature()
        self.timer.Tick("准备原始曲率映射")

        plane_clouds = self._CollectPlaneClouds(selected)
        if not plane_clouds:
            raise RuntimeError("未在选中实体的子树中找到 Plane_#### 点云（请确认选中的是 Ransac Detected Shapes 组）")

        self.logger.info(f"检测到 Plane 数量: {len(plane_clouds)}")
        self.timer.Tick("收集Plane实体")

        points_csv_path = os.path.join(self.output_dir, f"{self.base_name}_points.csv")
        summary_csv_path = os.path.join(self.output_dir, f"{self.base_name}_summary.csv")
        fig_path = os.path.join(self.output_dir, f"{self.base_name}_plane_sizes.png")

        header = [
            "X","Y","Z","nx","ny","nz","R","G","B","DR","DG","DB",
            "Discontinuity_id","Cluster_id","Segment_id",
            "A","B","C","D","RMS","Curvature","DistToPlane"
        ]

        summary_rows = []
        plane_sizes = []

        with open(points_csv_path, "w", encoding="utf-8") as f:
            f.write(",".join(header) + "\n")

            for disc_id, cloud in enumerate(plane_clouds):
                plane_name = self._GetName(cloud)

                pts = self._GetPoints(cloud)  # (N,3)
                abcd, rms = self._FitPlanePca(pts)
                dist = self._DistToPlane(pts, abcd)

                # normals（修复点：可能不存在任何法向读取接口）
                normals = self._GetNormals(cloud)
                if normals is None:
                    normals = np.repeat(abcd[:3].reshape(1, 3), pts.shape[0], axis=0)

                # colors
                colors = self._GetColors(cloud)
                if colors is None:
                    colors = np.full((pts.shape[0], 3), 255, dtype=np.int32)

                # 结构面随机色（可复现）
                rng = np.random.default_rng(seed=disc_id + 12345)
                dr, dg, db = [int(x) for x in rng.integers(0, 256, size=3)]

                # curvature
                curvature = None
                if MAP_CURVATURE_FROM_ORIGINAL:
                    curvature = self._MapCurvatureFromOriginal(pts)
                if curvature is None:
                    curvature = self._TryGetCurvatureFromCloud(cloud)
                if curvature is None:
                    if ESTIMATE_CURVATURE_IF_MISSING:
                        curvature = self._EstimateCurvatureKnnPca(pts, CURVATURE_KNN_K)
                    else:
                        curvature = np.zeros((pts.shape[0],), dtype=np.float64)

                # 常量列
                cluster_id = 0
                segment_id = 0
                a, b, c, d = abcd.tolist()

                # 分块写出（比逐行 f.write 更快）
                buf = []
                for i in range(pts.shape[0]):
                    buf.append(
                        f"{pts[i,0]:.6f},{pts[i,1]:.6f},{pts[i,2]:.6f},"
                        f"{normals[i,0]:.6f},{normals[i,1]:.6f},{normals[i,2]:.6f},"
                        f"{int(colors[i,0])},{int(colors[i,1])},{int(colors[i,2])},"
                        f"{dr},{dg},{db},"
                        f"{disc_id},{cluster_id},{segment_id},"
                        f"{a:.9f},{b:.9f},{c:.9f},{d:.9f},"
                        f"{rms:.9f},{float(curvature[i]):.9f},{float(dist[i]):.9f}\n"
                    )
                    if len(buf) >= WRITE_BUFFER_LINES:
                        f.writelines(buf)
                        buf.clear()
                if buf:
                    f.writelines(buf)

                summary_rows.append([disc_id, plane_name, pts.shape[0], dr, dg, db, a, b, c, d, rms])
                plane_sizes.append(pts.shape[0])

                if disc_id % 10 == 0:
                    self.logger.info(f"已处理 {disc_id+1}/{len(plane_clouds)}: {plane_name}, pts={pts.shape[0]}, RMS={rms:.6f}")

        self.timer.Tick("写出points CSV")

        with open(summary_csv_path, "w", encoding="utf-8") as f:
            f.write("Discontinuity_id,PlaneName,NumPoints,DR,DG,DB,A,B,C,D,RMS\n")
            for r in summary_rows:
                f.write(
                    f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]},"
                    f"{r[6]:.9f},{r[7]:.9f},{r[8]:.9f},{r[9]:.9f},{r[10]:.9f}\n"
                )

        self.timer.Tick("写出summary CSV")

        if PLOT_PLANE_SIZES:
            try:
                import matplotlib.pyplot as plt
                plt.figure()
                plt.bar(np.arange(len(plane_sizes)), plane_sizes)
                plt.xlabel("Discontinuity_id")
                plt.ylabel("NumPoints")
                plt.title("RANSAC Plane Patch Sizes")
                plt.tight_layout()
                plt.savefig(fig_path, dpi=200)
                plt.close()
                self.logger.info(f"已输出图件: {fig_path}")
            except Exception as e:
                self.logger.warning(f"matplotlib 不可用或绘图失败，跳过图件输出。原因: {e}")

        self.logger.info("完成")
        self.logger.info(f"Points CSV : {points_csv_path}")
        self.logger.info(f"Summary CSV: {summary_csv_path}")


def Main():
    exporter = PyccRansacExporter(output_dir=OUTPUT_DIR, base_name=OUTPUT_BASE_NAME)
    exporter.Run()


Main()
