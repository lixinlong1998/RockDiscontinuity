import os
import sys
import csv
import json
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt

# ========= 让 import src.* 生效：把“包含 src 文件夹的目录”加入 sys.path =========
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(PROJECT_ROOT)

from src.rock_discon_extract.results_exporter import ResultsExporter
from src.rock_discon_extract.logging_utils import LoggerManager, Timer

# =========================
# 常量：输入/输出 CSV 列顺序必须一致
# =========================
CSV_COLUMNS = [
    "X", "Y", "Z",
    "nx", "ny", "nz",
    "R", "G", "B",
    "DR", "DG", "DB",
    "Discontinuity_id", "Cluster_id", "Segment_id",
    "A", "B", "C", "D",
    "RMS", "Curvature", "DistToPlane"
]


@dataclass
class CsvVoxelExportConfig:
    """
    功能简介:
        CSV 点云（含 Curvature）-> curvature 阈值标记 -> 仅保留 label==1 -> voxel 切分导出（PLY/CSV 可选）

    实现思路:
        1) 读取 CSV（优先 pandas.read_csv，缺失则 fallback 到 numpy.genfromtxt）
        2) 生成 label：Curvature < threshold -> 1 else 0
        3) 仅保留 label==1 的点
        4) 以全量点云 min_xyz 作为 origin，保证 voxel 坐标一致
        5) np.unique 分组后逐 voxel 导出：
            - export_format="ply"：ResultsExporter._ExportToMeshlabPly
            - export_format="csv"：脚本内 _WriteVoxelCsv（列与输入一致）

    输入:
        csv_path: str
            输入 CSV 路径
        out_root_dir: str
            输出根目录
        voxel_size: float
            体素尺寸
        curvature_threshold: float
            曲率阈值：Curvature < threshold -> label=1（保留）
        color_mode: str
            仅在导出 PLY 时生效："RGB" 使用 R,G,B；"DRGB" 使用 DR,DG,DB
        export_format: str
            "ply" 或 "csv"
        min_points_per_voxel: int
            小于该点数的 voxel 不导出
        create_timestamp_subdir: bool
            输出是否加时间戳子目录
        save_hist: bool
            是否保存曲率直方图（全量/保留）
        save_labeled_csv: bool
            是否输出追加 label 的轻量 CSV（只含 X,Y,Z,Curvature,label）
    """
    csv_path: str
    out_root_dir: str
    voxel_size: float = 1.0
    curvature_threshold: float = 0.005
    color_mode: str = "RGB"  # 仅 PLY 时生效："RGB" or "DRGB"
    export_format: str = "ply"  # "ply" or "csv"
    min_points_per_voxel: int = 1
    create_timestamp_subdir: bool = True
    save_hist: bool = True
    save_labeled_csv: bool = False
    visualizer: bool = False


class CsvCurvatureVoxelExporter:
    def __init__(self, cfg: CsvVoxelExportConfig):
        self.cfg = cfg
        self.logger = LoggerManager.GetLogger(self.__class__.__name__)

    def Run(self) -> str:
        export_format = str(self.cfg.export_format).lower().strip()
        if export_format not in ("ply", "csv"):
            raise ValueError("export_format 仅支持 'ply' 或 'csv'")

        # ---------- 输出目录 ----------
        base_name = os.path.splitext(os.path.basename(self.cfg.csv_path))[0]
        if self.cfg.create_timestamp_subdir:
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_dir = os.path.join(self.cfg.out_root_dir, f"{ts}_CsvCurvVoxel_{base_name}")
        else:
            out_dir = os.path.join(self.cfg.out_root_dir, f"CsvCurvVoxel_{base_name}")
        os.makedirs(out_dir, exist_ok=True)
        LoggerManager.CreatLogFile(out_dir)

        self._SaveConfig(out_dir)

        # ---------- 读取 CSV（全字段，便于导出 CSV 时保持一致） ----------
        with Timer("Step1_ReadCsvAllColumns", self.logger):
            table, points, curvature = self._ReadCsvAllColumns(self.cfg.csv_path)

            n_all = points.shape[0]
            if n_all == 0:
                raise RuntimeError("CSV 中未读取到任何点。")
            self.logger.info(f"读取点数: N={n_all}")

        # ---------- 生成 label，并仅保留 label==1 ----------
        with Timer("Step2_LabelAndFilter", self.logger):
            label = (curvature < float(self.cfg.curvature_threshold)).astype(np.uint8)
            keep_mask = label == 1
            keep_idx = np.where(keep_mask)[0]

            n_keep = keep_idx.size
            self.logger.info(
                f"curvature_threshold={self.cfg.curvature_threshold}, "
                f"label==1(保留)={n_keep}/{n_all} ({n_keep / max(n_all, 1):.3f})"
            )

            pts_keep = points[keep_idx]
            curv_keep = curvature[keep_idx]

        # 可选：输出全量轻量标注 CSV
        if self.cfg.save_labeled_csv:
            with Timer("Step2b_SaveLabeledCsv", self.logger):
                out_labeled = os.path.join(out_dir, f"{base_name}_labeled_light.csv")
                self._SaveLightLabeledCsv(out_labeled, points, curvature, label)

        # ---------- voxel 分割（origin 用全量点的 min_xyz） ----------
        with Timer("Step3_VoxelizeAndGroup", self.logger):
            origin = points.min(axis=0)
            voxel_ijk = self._ComputeVoxelIjk(pts_keep, origin, self.cfg.voxel_size)

            uniq_vox, inverse, counts = np.unique(
                voxel_ijk, axis=0, return_inverse=True, return_counts=True
            )

            order = np.argsort(inverse)
            inverse_sorted = inverse[order]
            split_pos = np.flatnonzero(np.diff(inverse_sorted)) + 1
            groups = np.split(order, split_pos)

            self.logger.info(f"保留点 voxel 数: {uniq_vox.shape[0]}")

        # ---------- 导出 voxel ----------
        voxel_dir = os.path.join(out_dir, f"voxels_label1_{export_format}")
        os.makedirs(voxel_dir, exist_ok=True)

        voxel_index_csv = os.path.join(out_dir, f"{base_name}_voxel_index_label1_{export_format}.csv")
        with Timer("Step4_ExportVoxels", self.logger):
            exported, voxel_csv_paths_list = self._ExportVoxels(
                export_format=export_format,
                table=table,
                keep_idx=keep_idx,  # keep 子集对应原表的行号
                voxel_index_csv=voxel_index_csv,
                voxel_dir=voxel_dir,
                base_name=base_name,
                uniq_vox=uniq_vox,
                groups=groups,
                pts_keep=pts_keep,
                counts=counts,
                min_points_per_voxel=self.cfg.min_points_per_voxel
            )
            self.logger.info(f"导出 voxel 数: {exported}")

        # ---------- 曲率直方图 ----------
        if self.cfg.save_hist:
            with Timer("Step5_SaveHist", self.logger):
                self._SaveCurvatureHists(out_dir, base_name, curvature, curv_keep)

        self.logger.info(f"完成输出: {out_dir}")

        # ============ 绘制 dipdir_rose 与 stereonet_kde ============
        if self.cfg.visualizer:
            with Timer("Step6_SaveFigures", self.logger):
                from src.rock_discon_extract.visualizer import ResultsVisualizer
                # 1. 构造 (out_dir, point_path) 列表，这里 point_path 即 voxel CSV 路径
                paths_list = [(voxel_dir, csv_file_path)  # 你的每个 voxel CSV 完整路径
                              for csv_file_path in voxel_csv_paths_list]
                # 2. 创建可视化器实例
                viz = ResultsVisualizer(paths_list=paths_list)
                # 3. 输出单一结果图件
                viz.ExportAllSingleAnalysis(plots_name=["points_stereonet_kde"], output_formats=("png",), show=False)
        self.logger.info(f"完成输出: {out_dir}")

        return out_dir

    # =========================================================
    # 读取 CSV：全字段（用于 CSV 输出保持与输入一致）
    # =========================================================
    def _ReadCsvAllColumns(
            self,
            csv_path: str
    ) -> Tuple[Union["object", np.ndarray], np.ndarray, np.ndarray]:
        """
        功能简介:
            读取输入 CSV 的全部规定字段，并返回：
            - table：pandas.DataFrame 或 numpy structured array（用于导出 CSV）
            - points: (N,3)
            - curvature: (N,)

        实现思路:
            - 优先 pandas.read_csv(usecols=CSV_COLUMNS)（更快更稳）
            - pandas 不可用则 fallback numpy.genfromtxt(names=True)

        输出:
            table, points, curvature
        """
        # --- 优先 pandas ---
        try:
            import pandas as pd  # type: ignore
            df = pd.read_csv(csv_path, usecols=CSV_COLUMNS)

            # 强制列顺序（确保与输入一致）
            df = df[CSV_COLUMNS]

            points = df[["X", "Y", "Z"]].to_numpy(dtype=np.float64)
            curvature = df["Curvature"].to_numpy(dtype=np.float64)

            return df, points, curvature

        except Exception as e:
            self.logger.warning(f"pandas 读取失败，fallback 到 numpy.genfromtxt。原因: {repr(e)}")

        # --- fallback: numpy.genfromtxt ---
        data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")

        # 校验列齐全
        names = set(data.dtype.names or [])
        missing = [c for c in CSV_COLUMNS if c not in names]
        if missing:
            raise KeyError(f"CSV 缺少列: {missing}")

        # points / curvature
        x = np.asarray(data["X"], dtype=np.float64)
        y = np.asarray(data["Y"], dtype=np.float64)
        z = np.asarray(data["Z"], dtype=np.float64)
        points = np.stack([x, y, z], axis=1)

        curvature = np.asarray(data["Curvature"], dtype=np.float64)

        return data, points, curvature

    # =========================================================
    # 保存配置/可视化
    # =========================================================
    def _SaveConfig(self, out_dir: str) -> None:
        cfg_path = os.path.join(out_dir, "config.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(self.cfg.__dict__, f, ensure_ascii=False, indent=2)

    def _SaveLightLabeledCsv(
            self,
            out_csv: str,
            points: np.ndarray,
            curvature: np.ndarray,
            label: np.ndarray
    ) -> None:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["X", "Y", "Z", "Curvature", "Label"])
            for p, c, lb in zip(points, curvature, label):
                writer.writerow([float(p[0]), float(p[1]), float(p[2]), float(c), int(lb)])

    def _SaveCurvatureHists(
            self,
            out_dir: str,
            base_name: str,
            curv_all: np.ndarray,
            curv_keep: np.ndarray
    ) -> None:
        plt.figure()
        plt.hist(curv_all, bins=120)
        plt.xlabel("Curvature")
        plt.ylabel("Count")
        plt.title("Curvature Histogram (All)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{base_name}_curvature_hist_all.png"), dpi=200)
        plt.close()

        plt.figure()
        plt.hist(curv_keep, bins=120)
        plt.xlabel("Curvature")
        plt.ylabel("Count")
        plt.title("Curvature Histogram (Label==1, Curv<thr)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{base_name}_curvature_hist_label1.png"), dpi=200)
        plt.close()

    # =========================================================
    # voxel 计算与导出
    # =========================================================
    def _ComputeVoxelIjk(self, points: np.ndarray, origin: np.ndarray, voxel_size: float) -> np.ndarray:
        """
        voxel_ijk = floor((p - origin) / voxel_size)
        """
        if voxel_size <= 0:
            raise ValueError("voxel_size 必须 > 0")
        rel = (points - origin.reshape(1, 3)) / float(voxel_size)
        return np.floor(rel).astype(np.int64)

    def _ExportVoxels(
            self,
            export_format: str,
            table: Union["object", np.ndarray],
            keep_idx: np.ndarray,
            voxel_index_csv: str,
            voxel_dir: str,
            base_name: str,
            uniq_vox: np.ndarray,
            groups: List[np.ndarray],
            pts_keep: np.ndarray,
            counts: np.ndarray,
            min_points_per_voxel: int
    ) -> int:
        """
        功能简介:
            逐 voxel 导出：
            - export_format="ply": 调用 ResultsExporter._ExportToMeshlabPly
            - export_format="csv": 调用 _WriteVoxelCsv（列与输入一致）

        注意:
            groups 的索引是针对 pts_keep 的行号，需要映射回原表行号：orig_idx = keep_idx[idx_arr]
        """
        exported = 0
        voxel_csv_paths_list = []
        # voxel 索引表
        with open(voxel_index_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["voxel_i", "voxel_j", "voxel_k", "num_points", "file_name"])

            for voxel_idx, idx_arr in enumerate(groups):
                num_pts = int(idx_arr.size)
                if num_pts < int(min_points_per_voxel):
                    continue

                vx, vy, vz = uniq_vox[voxel_idx].tolist()

                if export_format == "ply":
                    file_name = f"{base_name}_vx{vx}_vy{vy}_vz{vz}_n{num_pts}.ply"
                    out_path = os.path.join(voxel_dir, file_name)

                    # 从 table 中取颜色（仅 PLY 需要）
                    colors = self._BuildPlyColors(table, keep_idx, idx_arr)

                    vertices = pts_keep[idx_arr].astype(np.float32, copy=False)
                    ResultsExporter._ExportToMeshlabPly(
                        filename=out_path,
                        vertices=vertices,
                        edges=None,
                        faces=None,
                        colors=colors
                    )

                else:
                    file_name = f"{base_name}_vx{vx}_vy{vy}_vz{vz}_n{num_pts}.csv"
                    out_path = os.path.join(voxel_dir, file_name)

                    # 映射回原表索引并写出 CSV（列与输入一致）
                    orig_idx_arr = keep_idx[idx_arr]
                    self._WriteVoxelCsv(table, orig_idx_arr, out_path)

                writer.writerow([int(vx), int(vy), int(vz), num_pts, file_name])
                exported += 1
                voxel_csv_paths_list.append(out_path)
                if exported % 100 == 0:
                    self.logger.info(f"已导出 voxel: {exported}")

        return exported, voxel_csv_paths_list

    def _BuildPlyColors(self, table: Union["object", np.ndarray], keep_idx: np.ndarray,
                        idx_arr: np.ndarray) -> np.ndarray:
        """
        功能简介:
            为 PLY 导出构建颜色数组 (N,3) uint8

        实现思路:
            - color_mode="RGB": 用 R,G,B
            - color_mode="DRGB": 用 DR,DG,DB
            - 颜色统一 round + clip 到 [0,255] 转 uint8

        输入:
            table: DataFrame 或 structured array
            keep_idx: 原表的保留行号数组
            idx_arr: 保留子集内某 voxel 的局部索引

        输出:
            colors: (num_pts,3) uint8
        """
        mode = str(self.cfg.color_mode).upper().strip()
        if mode == "DRGB":
            cols = ("DR", "DG", "DB")
        else:
            cols = ("R", "G", "B")

        orig_idx_arr = keep_idx[idx_arr]

        # pandas DataFrame
        if hasattr(table, "iloc"):
            sub = table.iloc[orig_idx_arr][list(cols)].to_numpy(dtype=np.float64)
        else:
            # numpy structured
            sub = np.stack([np.asarray(table[c][orig_idx_arr], dtype=np.float64) for c in cols], axis=1)

        colors = np.clip(np.rint(sub), 0, 255).astype(np.uint8)
        return colors

    # =========================================================
    # 新增功能：导出 voxel 为 CSV（列与输入一致）
    # =========================================================
    def _WriteVoxelCsv(self, table: Union["object", np.ndarray], orig_idx_arr: np.ndarray, out_csv_path: str) -> None:
        """
        功能简介:
            将某个 voxel 对应的点，按输入一致的字段与顺序写出到 CSV。

        实现思路:
            - 若 table 是 pandas.DataFrame：直接 iloc 切片并 to_csv(columns=CSV_COLUMNS)
            - 若 table 是 numpy structured array：用 csv.writer 按 CSV_COLUMNS 逐行写出

        输入:
            table:
                pandas.DataFrame 或 numpy structured array（包含 CSV_COLUMNS 全字段）
            orig_idx_arr:
                原表的行号数组（针对 table 的行索引）
            out_csv_path:
                输出 csv 文件路径

        输出:
            None（写文件落盘）
        """
        os.makedirs(os.path.dirname(os.path.abspath(out_csv_path)), exist_ok=True)

        # pandas DataFrame
        if hasattr(table, "iloc"):
            # columns 强制顺序与输入一致
            table.iloc[orig_idx_arr][CSV_COLUMNS].to_csv(out_csv_path, index=False)
            return

        # numpy structured array
        with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_COLUMNS)
            for i in orig_idx_arr:
                row = table[int(i)]
                writer.writerow([row[c] for c in CSV_COLUMNS])


if __name__ == "__main__":
    # ================== 直接改这里的参数 ==================
    csv_path = r"D:\Research\20250313_RockFractureSeg\Code\RockDiscontinuity\result\20251229_140543_Supervoxel_Rock_GLS4_part1_localize_0.05m\Rock_GLS4_part1_localize_0.05m_points.csv"
    out_root_dir = r"D:\Research\20250313_RockFractureSeg\Code\RockDiscontinuity\result\CurvVoxel"

    voxel_size = 1.0
    curvature_threshold = 0.005

    # 导出格式： "ply" 或 "csv"
    # export_format = "ply"
    export_format = "csv"

    # 仅 PLY 生效： "RGB" 或 "DRGB"
    color_mode = "RGB"

    min_points_per_voxel = 1

    cfg = CsvVoxelExportConfig(
        csv_path=csv_path,
        out_root_dir=out_root_dir,
        voxel_size=voxel_size,
        curvature_threshold=curvature_threshold,
        color_mode=color_mode,
        export_format=export_format,
        min_points_per_voxel=min_points_per_voxel,
        create_timestamp_subdir=True,
        save_hist=True,
        save_labeled_csv=False,
        visualizer=True,
    )

    exporter = CsvCurvatureVoxelExporter(cfg)
    exporter.Run()
