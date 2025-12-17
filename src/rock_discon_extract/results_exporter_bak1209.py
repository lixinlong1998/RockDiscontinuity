# results_exporter.py

import csv
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from .geometry import Discontinuity, Plane
from .logging_utils import LoggerManager, Timer
from .pointcloud import PointCloud


class ResultsExporter:
    """
    功能简介:
        从 PointCloud 与 Discontinuity 列表中, 生成用于分析与可视化的结果文件,
        包括:
            1) 点级 CSV (每行一个点, 含平面/结构面信息)
            2) 结构面级 CSV (每行一个 Discontinuity)
            3) 所有结构面的多边形 PLY 文件(使用三角扇表示每个结构面)
            4) 每个结构面的独立点/多边形文件, 便于单独检查

    实现思路(概要):
        - 在构造函数中保存 point_cloud 与 discontinuities, 以及可选的 cluster_labels。
        - 点级 CSV:
            遍历所有 Discontinuity/Segment/point_indices, 对每个点写出一行记录,
            包含点坐标/法向/颜色, 隶属 Discontinuity/Segment, 以及平面参数 A,B,C,D 和 RMS、
            以及根据 Discontinuity id 生成的可视化颜色 DR,DG,DB。
        - 结构面级 CSV:
            对每个 Discontinuity:
                * 汇总该结构面所有点索引(去重)
                * 基于代表平面(通常为第一个 Segment 的 plane) 和结构面自身的凸包计算面积
                * 汇总 PointsNumber / TraceLength / Roughness / MeanRMS 等指标
        - 多边形 PLY (全局):
            对每个 Discontinuity:
                * 调用 Discontinuity.ComputeGeometry 计算:
                  polygon_points_proj / centroid_proj / area / trace_length 等;
                * 构造顶点列表: [centroid_proj, 边界多边形点...]
                * 构造三角扇 faces: (centroid, v_i, v_{i+1});
                * 构造边界折线 edges: (v_i, v_{i+1}) 并闭合;
                * 所有结构面的顶点/边/面合并到一个 PLY 文件中。
        - 单结构面文件夹:
            在 <out_dir>/<base_name>_discontinuitys/ 下, 对每个 Discontinuity 输出:
                * DisconPoints_{id}_{PointsNumber}.csv
                * DisconPolygon_{id}_{PointsNumber}.ply

    输入:
        point_cloud: PointCloud
            包含所有点的点云对象.
        discontinuities: List[Discontinuity]
            结构面结果列表.
        cluster_labels: Optional[np.ndarray]
            若给定, 每个点的 cluster_id 从该数组中读取; 否则为 -1 或 Discontinuity.cluster_id。
        algorithm_name: str
            用于在日志中标识当前算法来源, 也可用于输出目录命名。
    """

    def __init__(
            self,
            point_cloud: PointCloud,
            discontinuities: List[Discontinuity],
            cluster_labels: Optional[np.ndarray] = None,
            algorithm_name: str = ""
    ):
        self.logger = LoggerManager.GetLogger(self.__class__.__name__)
        self.point_cloud = point_cloud
        self.discontinuities = discontinuities
        self.algorithm_name = algorithm_name if algorithm_name else "UnknownAlgo"

        num_points = len(self.point_cloud.points)
        if cluster_labels is not None and cluster_labels.shape[0] != num_points:
            raise ValueError(
                f"cluster_labels 长度({cluster_labels.shape[0]})"
                f"与点云点数({num_points})不一致。"
            )
        self.cluster_labels = cluster_labels

        self.logger.info(
            f"ResultsExporter 初始化: N_points={num_points}, "
            f"N_discontinuities={len(discontinuities)}, "
            f"has_cluster_labels={cluster_labels is not None}"
        )

    # ---------------------------------------------------------
    # 总入口: 一次性导出全部结果
    # ---------------------------------------------------------
    def ExportAll(
            self,
            result_root_dir: str,
            point_cloud_path: str
    ) -> Dict[str, str]:
        """
        功能简介:
            一次性导出三类结果文件:
                - 点级 CSV: <basename>_points.csv
                - 结构面级 CSV: <basename>_discontinuitys.csv
                - 多边形 PLY: <basename>_polygons.ply
                - 每个结构面的独立 CSV/PLY 文件夹: <basename>_discontinuitys/

        实现思路:
            1) 调用 _CreateOutputSubdir 在 result_root_dir 下创建
               "YYYYMMDD_HHMMSS_算法名" 子目录;
            2) 从 point_cloud_path 提取 basename, 去掉扩展名;
            3) 在子目录中组合三个主文件路径:
               - basename_points.csv
               - basename_discontinuitys.csv
               - basename_polygons.ply
               以及子文件夹:
               - basename_discontinuitys/
            4) 分别调用 ExportPointLevelCsv / ExportDiscontinuityLevelCsv /
               ExportDiscontinuityPolygonsToPly / ExportPerDiscontinuityFiles;
            5) 返回包含输出路径信息的字典, 便于上层记录或打印.

        输入:
            result_root_dir: str
                结果根目录.
            point_cloud_path: str
                输入点云文件路径, 用于提取文件名前缀.

        输出:
            paths: Dict[str, str]
                包含 "dir", "points_csv", "disc_csv", "polygons_ply",
                "discontinuity_dir", "point_cloud_path".
        """
        base_name = os.path.splitext(os.path.basename(point_cloud_path))[0]
        out_dir = self._CreateOutputSubdir(result_root_dir, base_name)

        points_csv = os.path.join(out_dir, f"{base_name}_points.csv")
        disc_csv = os.path.join(out_dir, f"{base_name}_discontinuitys.csv")
        polygons_ply = os.path.join(out_dir, f"{base_name}_polygons.ply")

        # 单结构面文件夹
        discon_dir = os.path.join(out_dir, f"{base_name}_discontinuitys")
        os.makedirs(discon_dir, exist_ok=True)

        self.ExportPointLevelCsv(points_csv)
        self.ExportDiscontinuityLevelCsv(disc_csv)
        self.ExportDiscontinuityPolygonsToPly(polygons_ply)
        self.ExportPerDiscontinuityFiles(discon_dir, base_name)

        return {
            "dir": out_dir,
            "points_csv": points_csv,
            "disc_csv": disc_csv,
            "polygons_ply": polygons_ply,
            "discontinuity_dir": discon_dir,
            "point_cloud_path": point_cloud_path,
        }

    # ---------------------------------------------------------
    # 点级 CSV 导出
    # ---------------------------------------------------------
    def ExportPointLevelCsv(self, csv_path: str) -> None:
        """
        功能简介:
            导出点级 CSV 文件, 每一行代表一个属于某个结构面的点,
            包含几何属性和结构面/平面信息, 以及可视化颜色 DR/DG/DB。

        实现思路:
            1) 打开 csv 文件, 写入表头:
               [X, Y, Z, nx, ny, nz, R, G, B,
                DR, DG, DB,
                Discontinuity_id, Cluster_id, Segment_id,
                A, B, C, D, RMS]
            2) 遍历所有 Discontinuity (按列表索引作为 Discontinuity_id):
               遍历其所有 Segment (索引作为 Segment_id):
                 对于 segment.point_indices 中的每个点索引:
                     - 取出点的坐标/颜色/法向;
                     - cluster_id: 若有 cluster_labels 则用对应值,
                       否则尝试使用 disc.cluster_id, 再否则为 -1;
                     - 通过 _GenerateColorFromId(disc_id) 得到 DR,DG,DB;
                     - 平面参数 A,B,C,D 与 plane.rmse;
                     - 写入一行 CSV.
            3) 记录总写入点数并输出日志.

        输入:
            csv_path: str
                输出 CSV 文件路径.

        输出:
            无, 但在 csv_path 处生成点级结果文件.
        """
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)

        with Timer(f"ExportPointLevelCsv({os.path.basename(csv_path)})", self.logger):
            count_rows = 0

            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                # 表头
                writer.writerow([
                    "X", "Y", "Z",
                    "nx", "ny", "nz",
                    "R", "G", "B",
                    "DR", "DG", "DB",
                    "Discontinuity_id",
                    "Cluster_id",
                    "Segment_id",
                    "A", "B", "C", "D",
                    "RMS",
                ])

                for disc_id, disc in enumerate(self.discontinuities):
                    disc_cluster_id = getattr(disc, "cluster_id", -1)
                    base_color = self._GenerateColorFromId(disc_id)
                    dr, dg, db = base_color

                    for seg_id, seg in enumerate(disc.segments):
                        plane = seg.plane
                        if plane is None:
                            continue
                        A, B, C = plane.normal
                        D = plane.d
                        rms = plane.rmse

                        for pt_idx in seg.point_indices:
                            p = self.point_cloud.points[pt_idx]

                            # 坐标
                            x, y, z = p.x, p.y, p.z

                            # 法向: 若 point.normal 为空则导出 NaN
                            if getattr(p, "normal", None) is not None:
                                nx, ny, nz = p.normal
                            else:
                                nx = ny = nz = float("nan")

                            # 颜色
                            r_val = getattr(p, "r", 0)
                            g_val = getattr(p, "g", 0)
                            b_val = getattr(p, "b", 0)

                            # cluster_id: 优先使用外部 cluster_labels
                            if self.cluster_labels is not None:
                                cluster_id = int(self.cluster_labels[pt_idx])
                            else:
                                cluster_id = disc_cluster_id if disc_cluster_id != -1 else -1

                            writer.writerow([
                                x, y, z,
                                nx, ny, nz,
                                r_val, g_val, b_val,
                                dr, dg, db,
                                disc_id,
                                cluster_id,
                                seg_id,
                                A, B, C, D,
                                rms,
                            ])
                            count_rows += 1

            self.logger.info(
                f"点级 CSV 导出完成: {csv_path}, 共写入 {count_rows} 行."
            )

    # ---------------------------------------------------------
    # 结构面级 CSV 导出
    # ---------------------------------------------------------
    def ExportDiscontinuityLevelCsv(self, csv_path: str) -> None:
        """
        功能简介:
            导出结构面级 CSV 文件, 每一行代表一个 Discontinuity,
            包含倾角/倾向、平面参数、面积等统计信息.

        实现思路:
            对每个 Discontinuity:
                1) 代表平面 plane_ref:
                   - 若 segments 非空, 取第一个 segment 的 plane;
                   - 若 segments 为空或 plane 为空, 跳过该结构面;
                2) 点数 PointsNumber:
                   - 优先使用 Discontinuity.point_indices 去重后的长度;
                   - 若为空则为 0.
                3) TraceLength:
                   - 仍保持为所有 segment.trace_length 之和(与之前逻辑一致);
                   - 若你之后希望改用 disc.trace_length, 可在此处替换。
                4) MeanRMS: 各 segment 所属平面 rmse 的平均值。
                5) 面积:
                   - 调用 disc.ComputeGeometry(self.point_cloud, plane_ref),
                     直接使用 disc.area。
                6) 写入 CSV 行:
                   [Discontinuity_id, Cluster_id, Dip, Dipdir,
                    A, B, C, D, PointsNumber, Area, TraceLength,
                    Roughness, MeanRMS]

        输入:
            csv_path: str
                输出 CSV 文件路径.

        输出:
            无, 但在 csv_path 生成结构面级结果文件.
        """
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)

        with Timer(f"ExportDiscontinuityLevelCsv({os.path.basename(csv_path)})", self.logger):
            count_rows = 0

            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                # 表头
                writer.writerow([
                    "Discontinuity_id",
                    "Cluster_id",
                    "Dip",
                    "Dipdir",
                    "A", "B", "C", "D",
                    "PointsNumber",
                    "Area",
                    "TraceLength",
                    "Roughness",
                    "MeanRMS",
                ])

                for disc_id, disc in enumerate(self.discontinuities):
                    if not disc.segments:
                        continue
                    disc.ComputeGeometry(self.point_cloud)
                    disc_cluster_id = getattr(disc, "cluster_id", -1)

                    # 代表平面
                    plane_ref = disc.plane
                    if plane_ref is None:
                        continue

                    A, B, C = plane_ref.normal
                    D = plane_ref.d

                    # 点数: 使用 Discontinuity.point_indices
                    if getattr(disc, "point_indices", None):
                        points_number = len(set(disc.point_indices))
                    else:
                        points_number = 0

                    # TraceLength
                    trace_length = disc.trace_length

                    # MeanRMS: 各 segment 所属平面 rmse 的平均值
                    rms_list = [seg.plane.rmse for seg in disc.segments if seg.plane is not None]
                    mean_rms = float(np.mean(rms_list)) if rms_list else float("nan")

                    roughness = getattr(disc, "roughness", 0.0)

                    # 面积/多边形: 调用 Discontinuity 自身的方法计算

                    area = float(getattr(disc, "area", 0.0))

                    writer.writerow([
                        disc_id,
                        disc_cluster_id,
                        disc.dip,
                        disc.dip_direction,
                        A, B, C, D,
                        points_number,
                        area,
                        trace_length,
                        roughness,
                        mean_rms,
                    ])
                    count_rows += 1

            self.logger.info(
                f"结构面级 CSV 导出完成: {csv_path}, 共写入 {count_rows} 行."
            )

    # ---------------------------------------------------------
    # 全局多边形 PLY 导出 (三角扇)
    # ---------------------------------------------------------
    def ExportDiscontinuityPolygonsToPly(
            self,
            ply_path: str
    ) -> None:
        """
        功能简介:
            将所有 Discontinuity 的多边形面片导出到一个 PLY 文件,
            每个结构面的边界在其代表平面上形成一个三角扇(centroid 为公共顶点),
            同时保留边界折线 edges, 便于在 MeshLab/CloudCompare 中查看。

        实现思路:
            对每个 Discontinuity:
                1) 若无 segments 或代表平面, 则跳过;
                2) 调用 disc.ComputeGeometry(self.point_cloud, plane_ref),
                   确保 polygon_points_proj 与 centroid_proj 已计算;
                3) 若 polygon_points_proj 少于 3 个, 跳过;
                4) 构造顶点:
                    - v0 = centroid_proj
                    - v1...vM = polygon_points_proj
                5) 颜色:
                    - 使用 _GenerateColorFromId(disc_id) 生成单一颜色, 对该结构面所有顶点复用。
                6) 边:
                    - 仅对边界点 v1...vM 按顺序构造折线并首尾相连。
                7) 面:
                    - 构造三角扇: (v0, v_i, v_{i+1}), i=1..M-1, 再闭合 (v0, v_M, v_1)。
            最后调用 _ExportToMeshlabPly 将所有结构面合并写入一个 PLY。

        输入:
            ply_path: str
                输出 PLY 文件路径.

        输出:
            无, 但在 ply_path 生成包含所有结构面的多边形 mesh。
        """
        ply_path = os.path.abspath(ply_path)
        os.makedirs(os.path.dirname(ply_path), exist_ok=True)

        with Timer(f"ExportDiscontinuityPolygonsToPly({os.path.basename(ply_path)})", self.logger):
            vertices_list: List[np.ndarray] = []
            colors_list: List[np.ndarray] = []
            edges_list: List[np.ndarray] = []
            faces_list: List[np.ndarray] = []

            offset = 0  # 全局顶点索引偏移量

            for disc_id, disc in enumerate(self.discontinuities):
                if not disc.segments:
                    continue

                plane_ref = disc.plane
                if plane_ref is None:
                    continue

                # 确保多边形/面积等已计算
                disc.ComputeGeometry(self.point_cloud)

                pts_proj = getattr(disc, "polygon_points_proj", None)
                if pts_proj is None:
                    pts_proj = getattr(disc, "polygon_points", None)

                if pts_proj is None:
                    continue

                pts_proj = np.asarray(pts_proj, dtype=np.float32)
                m = pts_proj.shape[0]
                if m < 3:
                    continue

                centroid_proj = getattr(disc, "centroid_proj", None)
                if centroid_proj is None:
                    # 若未计算, 用边界点平均代替
                    centroid_proj = np.mean(pts_proj, axis=0)
                centroid_proj = np.asarray(centroid_proj, dtype=np.float32).reshape(1, 3)

                # 顶点列表: [centroid_proj, 边界点...]
                vertices_disc = np.vstack([centroid_proj, pts_proj])  # (M+1, 3)
                vertices_list.append(vertices_disc)

                # 颜色: 整个结构面统一颜色
                base_color = self._GenerateColorFromId(disc_id)
                dr, dg, db = base_color
                colors_disc = np.tile(np.array([dr, dg, db], dtype=np.uint8), (m + 1, 1))
                colors_list.append(colors_disc)

                # 边: 仅边界折线 (v1..vM), 首尾相连
                edges_disc: List[List[int]] = []
                if m >= 2:
                    for i in range(m - 1):
                        edges_disc.append([offset + 1 + i, offset + 1 + i + 1])
                    edges_disc.append([offset + 1 + m - 1, offset + 1])  # 闭合
                if edges_disc:
                    edges_list.append(np.asarray(edges_disc, dtype=np.int32))

                # 面: 三角扇 (center, v_i, v_{i+1})
                faces_disc: List[List[int]] = []
                center_idx = offset
                for i in range(m):
                    j = (i + 1) % m
                    faces_disc.append([center_idx, offset + 1 + i, offset + 1 + j])
                if faces_disc:
                    faces_list.append(np.asarray(faces_disc, dtype=np.int32))

                offset += (m + 1)

            if not vertices_list:
                self.logger.warning("没有可导出的结构面多边形, 跳过写 PLY。")
                return

            vertices = np.vstack(vertices_list)
            colors = np.vstack(colors_list)
            edges = np.vstack(edges_list) if edges_list else None
            faces = np.vstack(faces_list) if faces_list else None

            self._ExportToMeshlabPly(
                filename=ply_path,
                vertices=vertices,
                edges=edges,
                faces=faces,
                colors=colors
            )

            self.logger.info(
                f"结构面多边形 PLY 导出完成: {ply_path}, "
                f"共 {vertices.shape[0]} 个顶点, "
                f"{edges.shape[0] if edges is not None else 0} 条边, "
                f"{faces.shape[0] if faces is not None else 0} 个三角面."
            )

    # ---------------------------------------------------------
    # 单结构面 CSV/PLY 导出
    # ---------------------------------------------------------
    def ExportPerDiscontinuityFiles(
            self,
            out_dir_root: str,
            base_name: str
    ) -> None:
        """
        功能简介:
            为每个 Discontinuity 单独导出:
                - DisconPoints_{id}_{PointsNumber}.csv
                - DisconPolygon_{id}_{PointsNumber}.ply

        实现思路:
            1) 为每个结构面汇总所有点索引(去重), 提取点坐标/法向/颜色, 写入 CSV。
               CSV 字段结构与全局 points.csv 基本一致, 但仅包含单个结构面的点。
            2) 调用 disc.ComputeGeometry, 基于 polygon_points_proj 和 centroid_proj
               构造三角扇形式的多边形 PLY, 与全局 _polygons.ply 保持一致。
        """
        out_dir_root = os.path.abspath(out_dir_root)
        os.makedirs(out_dir_root, exist_ok=True)

        with Timer(f"ExportPerDiscontinuityFiles({os.path.basename(out_dir_root)})", self.logger):
            for disc_id, disc in enumerate(self.discontinuities):
                # 汇总点索引
                if getattr(disc, "point_indices", None):
                    idx_all = np.unique(np.asarray(disc.point_indices, dtype=int))
                else:
                    # 若 point_indices 为空, 尝试从 segments 合并
                    idx_list: List[int] = []
                    for seg in disc.segments:
                        idx_list.extend(list(seg.point_indices))
                    if not idx_list:
                        continue
                    idx_all = np.unique(np.asarray(idx_list, dtype=int))

                points_number = int(idx_all.shape[0])
                if points_number == 0:
                    continue

                # 文件名
                csv_name = f"DisconPoints_{disc_id}_{points_number}.csv"
                ply_name = f"DisconPolygon_{disc_id}_{points_number}.ply"

                csv_path = os.path.join(out_dir_root, csv_name)
                ply_path = os.path.join(out_dir_root, ply_name)

                # 1) 单结构面点级 CSV
                with open(csv_path, "w", newline="", encoding="utf-8") as f_csv:
                    writer = csv.writer(f_csv)
                    writer.writerow([
                        "X", "Y", "Z",
                        "nx", "ny", "nz",
                        "R", "G", "B",
                        "DR", "DG", "DB",
                        "Discontinuity_id",
                        "Cluster_id",
                        "Segment_id",
                    ])

                    disc_cluster_id = getattr(disc, "cluster_id", -1)
                    base_color = self._GenerateColorFromId(disc_id)
                    dr, dg, db = base_color

                    # 为加速查找 segment_id, 使用一个简单的点索引 -> segment_id 映射
                    seg_id_map = {}
                    for seg_id, seg in enumerate(disc.segments):
                        for pid in seg.point_indices:
                            seg_id_map[int(pid)] = seg_id

                    for pid in idx_all:
                        p = self.point_cloud.points[int(pid)]
                        x, y, z = p.x, p.y, p.z

                        if getattr(p, "normal", None) is not None:
                            nx, ny, nz = p.normal
                        else:
                            nx = ny = nz = float("nan")

                        r_val = getattr(p, "r", 0)
                        g_val = getattr(p, "g", 0)
                        b_val = getattr(p, "b", 0)

                        if self.cluster_labels is not None:
                            cluster_id = int(self.cluster_labels[int(pid)])
                        else:
                            cluster_id = disc_cluster_id if disc_cluster_id != -1 else -1

                        seg_id = seg_id_map.get(int(pid), -1)

                        writer.writerow([
                            x, y, z,
                            nx, ny, nz,
                            r_val, g_val, b_val,
                            dr, dg, db,
                            disc_id,
                            cluster_id,
                            seg_id,
                        ])

                # 2) 单结构面多边形 PLY (三角扇)
                plane_ref = disc.plane
                if plane_ref is None:
                    continue

                disc.ComputeGeometry(self.point_cloud)

                pts_proj = getattr(disc, "polygon_points_proj", None)
                if pts_proj is None:
                    pts_proj = getattr(disc, "polygon_points", None)
                if pts_proj is None:
                    continue

                pts_proj = np.asarray(pts_proj, dtype=np.float32)
                m = pts_proj.shape[0]
                if m < 3:
                    continue

                centroid_proj = getattr(disc, "centroid_proj", None)
                if centroid_proj is None:
                    centroid_proj = np.mean(pts_proj, axis=0)
                centroid_proj = np.asarray(centroid_proj, dtype=np.float32).reshape(1, 3)

                vertices_disc = np.vstack([centroid_proj, pts_proj])  # (M+1, 3)

                base_color = self._GenerateColorFromId(disc_id)
                dr, dg, db = base_color
                colors_disc = np.tile(np.array([dr, dg, db], dtype=np.uint8), (m + 1, 1))

                # 边: 边界折线
                edges_disc: List[List[int]] = []
                if m >= 2:
                    for i in range(m - 1):
                        edges_disc.append([1 + i, 1 + i + 1])
                    edges_disc.append([1 + m - 1, 1])  # 闭合
                edges_arr = np.asarray(edges_disc, dtype=np.int32) if edges_disc else None

                # 面: 三角扇
                faces_disc: List[List[int]] = []
                center_idx = 0
                for i in range(m):
                    j = (i + 1) % m
                    faces_disc.append([center_idx, 1 + i, 1 + j])
                faces_arr = np.asarray(faces_disc, dtype=np.int32) if faces_disc else None

                self._ExportToMeshlabPly(
                    filename=ply_path,
                    vertices=vertices_disc,
                    edges=edges_arr,
                    faces=faces_arr,
                    colors=colors_disc
                )

    # ---------------------------------------------------------
    # 通用 PLY 导出
    # ---------------------------------------------------------
    @staticmethod
    def _ExportToMeshlabPly(
            filename: str,
            vertices: Optional[np.ndarray] = None,
            edges: Optional[np.ndarray] = None,
            faces: Optional[np.ndarray] = None,
            colors: Optional[np.ndarray] = None
    ) -> None:
        """
        功能简介:
            将顶点/边/面导出到一个 ASCII PLY 文件, 便于在 MeshLab / CloudCompare 中可视化。

        输入:
            filename: str
                输出 PLY 文件路径。
            vertices: np.ndarray
                顶点数组, shape=(N,3), 每行 [x,y,z]。
            edges: np.ndarray 或 None
                边数组, shape=(M,2), 每行 [v1,v2]。
            faces: np.ndarray 或 None
                面数组, shape=(K,3), 每行 [v1,v2,v3]。
            colors: np.ndarray 或 None
                顶点颜色数组, shape=(N,3), 每行 [r,g,b], 范围 [0,255]。
        """
        if vertices is None:
            raise ValueError("顶点数据不能为空！")

        vertices = np.asarray(vertices, dtype=np.float32)
        has_edges = edges is not None
        has_faces = faces is not None
        has_colors = colors is not None

        if has_edges:
            edges = np.asarray(edges, dtype=np.int32)
        if has_faces:
            faces = np.asarray(faces, dtype=np.int32)
        if has_colors:
            colors = np.asarray(colors, dtype=np.uint8)
            if colors.shape[0] != vertices.shape[0]:
                raise ValueError("colors 行数必须与 vertices 相同！")

        with open(filename, "w", encoding="utf-8") as f:
            # 写入 PLY 头部
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(vertices)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            if has_colors:
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
            if has_edges:
                f.write(f"element edge {len(edges)}\n")
                f.write("property int vertex1\n")
                f.write("property int vertex2\n")
            if has_faces:
                f.write(f"element face {len(faces)}\n")
                f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")

            # 写入顶点数据（+颜色）
            for i, v in enumerate(vertices):
                line = f"{v[0]} {v[1]} {v[2]}"
                if has_colors:
                    c = colors[i]
                    line += f" {int(c[0])} {int(c[1])} {int(c[2])}"
                f.write(line + "\n")

            # 写入边数据
            if has_edges:
                for e in edges:
                    f.write(f"{int(e[0])} {int(e[1])}\n")

            # 写入面数据
            if has_faces:
                for face in faces:
                    f.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")

    # ---------------------------------------------------------
    # 输出子目录 & 颜色生成
    # ---------------------------------------------------------
    def _CreateOutputSubdir(self, result_root_dir: str, base_name: str) -> str:
        """
        功能简介:
            在给定结果根目录下, 根据当前时间和算法名称创建子目录,
            例如: <result_root_dir>/20251119_224205_RANSAC_open3d

        实现思路:
            1) 使用 datetime.now() 获取当前时间, 格式化为 "YYYYMMDD_HHMMSS";
            2) 将算法名中的空格替换为下划线, 避免目录名不便使用;
            3) 在 result_root_dir 下拼接出子目录路径, 若不存在则创建.

        输入:
            result_root_dir: str
                结果根目录路径.

        输出:
            out_dir: str
                创建好的子目录绝对路径.
        """
        result_root_dir = os.path.abspath(result_root_dir)
        os.makedirs(result_root_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_algo = self.algorithm_name.replace(" ", "_")
        subdir_name = f"{timestamp}_{safe_algo}_{base_name}"
        out_dir = os.path.join(result_root_dir, subdir_name)

        os.makedirs(out_dir, exist_ok=True)
        self.logger.info(f"结果输出子目录: {out_dir}")
        return out_dir

    @staticmethod
    def _GenerateColorFromId(idx: int) -> Tuple[int, int, int]:
        """
        功能简介:
            根据结构面编号 idx 生成一个稳定的伪随机 RGB 颜色,
            用于点云/多边形可视化时按结构面着色。

        实现思路:
            使用简单的线性同余类哈希, 确保:
                - 同一个 idx 每次调用得到相同颜色;
                - 不同 idx 颜色有一定区分度。

        输入:
            idx: int
                结构面编号(通常为列表索引).

        输出:
            (r, g, b): Tuple[int, int, int]
                颜色分量, 取值范围 [0, 255].
        """
        # 简单哈希到 [0,255] 区间
        r = (37 * idx + 59) % 256
        g = (17 * idx + 97) % 256
        b = (73 * idx + 23) % 256
        return int(r), int(g), int(b)
