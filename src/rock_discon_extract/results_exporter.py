# results_exporter.py

import csv
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from .algorithms.base import PlaneClusterInfo
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
            包含点坐标/法向/颜色, 所属 Discontinuity/Segment, 以及平面参数 A,B,C,D,
            平均拟合误差 RMS、局部曲率 Curvature、点到平面的距离 DistToPlane,
            以及基于 Discontinuity_id 的统一颜色 DR,DG,DB。
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
            discontinuities: List,
            clusters=List,
            algorithm_name: str = "",
            parameters=Dict,
    ):
        self.logger = LoggerManager.GetLogger(self.__class__.__name__)
        self.point_cloud = point_cloud
        self.discontinuities = discontinuities
        self.clusters = clusters
        self.algorithm_name = algorithm_name if algorithm_name else "UnknownAlgo"
        num_points = len(self.point_cloud.points)
        self.parameters = parameters
        self.logger.info(
            f"ResultsExporter 初始化: N_points={num_points}, "
            f"N_discontinuities={len(discontinuities)}, "
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
                - 聚类级 CSV: <basename>_clusters.csv
                - 多边形 PLY: <basename>_polygons.ply
                - 参数 JSON: <base_name>_parameters.json
                - 每个结构面的独立 CSV/PLY 文件夹: <basename>_discontinuitys/

        实现思路:
            1) 调用 _CreateOutputSubdir 在 result_root_dir 下创建
               "YYYYMMDD_HHMMSS_算法名" 子目录;
            2) 从 point_cloud_path 提取 basename, 去掉扩展名;
            3) 在子目录中组合三个主文件路径:
               - basename_points.csv
               - basename_discontinuitys.csv
               - basename_clusters.csv
               - basename_polygons.ply
               以及子文件夹:
               - basename_discontinuitys/
            4) 分别调用 ExportPointLevelCsv / ExportDiscontinuityLevelCsv /
               ExportClusterLevelCsv / ExportDiscontinuityPolygonsToPly /
               ExportPerDiscontinuityFiles / ExportParametersJson;
            5) 返回包含输出路径信息的字典, 便于上层记录或打印.

        输入:
            result_root_dir: str
                结果根目录.
            point_cloud_path: str
                输入点云文件路径, 用于提取文件名前缀.

        输出:
            paths: Dict[str, str]
                包含 "dir", "points_csv", "disc_csv", "clusters_csv", "polygons_ply",
                "discontinuity_dir", "point_cloud_path", "parameters_json"
        """
        base_name = os.path.splitext(os.path.basename(point_cloud_path))[0]
        out_dir = self._CreateOutputSubdir(result_root_dir, base_name)

        points_csv = os.path.join(out_dir, f"{base_name}_points.csv")
        disc_csv = os.path.join(out_dir, f"{base_name}_discontinuitys.csv")
        clusters_csv = os.path.join(out_dir, f"{base_name}_clusters.csv")
        polygons_ply = os.path.join(out_dir, f"{base_name}_polygons.ply")
        parameters_json = os.path.join(out_dir, f"{base_name}_parameters.json")

        # 单结构面文件夹
        discon_dir = os.path.join(out_dir, f"{base_name}_discontinuitys")
        os.makedirs(discon_dir, exist_ok=True)

        self.ExportPointLevelCsv(points_csv)
        self.ExportDiscontinuityLevelCsv(disc_csv)
        self.ExportClusterLevelCsv(clusters_csv)
        self.ExportDiscontinuityPolygonsToPly(polygons_ply)
        self.ExportPerDiscontinuityFiles(discon_dir, base_name)
        self.ExportParametersJson(parameters_json)

        return {
            "dir": out_dir,
            "points_csv": points_csv,
            "disc_csv": disc_csv,
            "clusters_csv": clusters_csv,
            "polygons_ply": polygons_ply,
            "discontinuity_dir": discon_dir,
            "point_cloud_path": point_cloud_path,
            "parameters_json": parameters_json,

        }

    # ---------------------------------------------------------
    # 参数 JSON 导出
    # ---------------------------------------------------------
    def ExportParametersJson(self, json_path: str) -> None:
        """
        功能简介:
            将本次运行的算法参数导出为 JSON 文件, 便于复现实验与结果追溯。

        实现思路:
            - 将 self.parameters (dict) 直接写入 JSON。
            - 同时补充本次导出的基本上下文信息(算法名、时间、点数、结构面数、簇数)。

        输入变量:
            json_path: str
                输出 JSON 文件路径。

        输出:
            None
        """
        try:
            os.makedirs(os.path.dirname(json_path), exist_ok=True)
        except Exception:
            pass

        payload = {
            "algorithm_name": self.algorithm_name,
            "export_time": datetime.now().isoformat(timespec="seconds"),
            "n_points": int(len(self.point_cloud.points)) if getattr(self.point_cloud, "points",
                                                                     None) is not None else None,
            "n_discontinuities": int(len(self.discontinuities)) if self.discontinuities is not None else 0,
            "n_clusters": int(len(self.clusters)) if self.clusters is not None else 0,
            "parameters": self.parameters if self.parameters is not None else {},
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        self.logger.info(f"参数 JSON 导出完成: {json_path}")

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
                A, B, C, D, RMS,
                Curvature, DistToPlane]
            2) 遍历所有 Discontinuity (按列表索引作为 Discontinuity_id):
               遍历其所有 Segment (索引作为 Segment_id):
                 对于 segment.point_indices 中的每个点索引:
                     - 取出点的坐标/颜色/法向/曲率;
                     - cluster_id: 若有 cluster_labels 则用对应值, 否则为 -1 或 disc.cluster_id;
                     - 平面参数 A,B,C,D 与 plane.rmse;
                     - DistToPlane: 当前点到该平面的距离(绝对值);
                     - DR,DG,DB: 由 _GenerateColorFromId(disc_id) 生成的统一颜色;
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
                    "Curvature",
                    "DistToPlane",
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

                            # 曲率
                            curvature = (
                                float(p.curvature)
                                if hasattr(p, "curvature")
                                else float("nan")
                            )
                            # 点到平面的距离(绝对值)
                            dist_to_plane = abs(A * x + B * y + C * z + D)

                            writer.writerow([
                                x, y, z,
                                nx, ny, nz,
                                r_val, g_val, b_val,
                                dr, dg, db,
                                disc_id,
                                disc.cluster_id,
                                seg_id,
                                A, B, C, D,
                                rms,
                                curvature,
                                dist_to_plane,
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
    # 添加聚类级CSV导出方法
    # ---------------------------------------------------------
    def ExportClusterLevelCsv(self, csv_path: str) -> None:
        """
        功能简介:
            导出聚类级 CSV 文件, 每一行代表一个聚类簇,
            包含簇ID、平均倾角/倾向、结构面数量等信息.

        实现思路:
            1) 从 discontinuities 中提取所有有效聚类(cluster_id >= 0);
            2) 按 cluster_id 分组, 统计每个簇的结构面数量;
            3) 对每个簇, 计算面积加权平均倾角/倾向;
            4) 写入 CSV 行: [Cluster_id, Dip, Dipdir, Discontinuity_Number].

        输入:
            csv_path: str
                输出 CSV 文件路径.

        输出:
            无, 但在 csv_path 生成聚类级结果文件.
        """
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)

        with Timer(f"ExportClusterLevelCsv({os.path.basename(csv_path)})", self.logger):
            # # 收集所有有效聚类信息
            # clusters_dict = {}
            #
            # for disc in self.discontinuities:
            #     cluster_id = getattr(disc, "cluster_id", -1)
            #     if cluster_id < 0:
            #         continue
            #
            #     if cluster_id not in clusters_dict:
            #         clusters_dict[cluster_id] = {
            #             "dip_list": [],
            #             "dipdir_list": [],
            #             "area_list": [],
            #             "count": 0
            #         }
            #
            #     clusters_dict[cluster_id]["dip_list"].append(disc.dip)
            #     clusters_dict[cluster_id]["dipdir_list"].append(disc.dip_direction)
            #     clusters_dict[cluster_id]["area_list"].append(getattr(disc, "area", 0.0))
            #     clusters_dict[cluster_id]["count"] += 1
            #
            # if not clusters_dict:
            #     self.logger.warning(f"没有有效的聚类数据, 跳过聚类CSV导出: {csv_path}")
            #     return

            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                # 表头
                writer.writerow([
                    "Cluster_id",
                    "Dip",
                    "Dipdir",
                    "Discontinuity_Number",
                    "confidence",
                    "is_manual_seed",
                ])
                """ 关于confidence的计算
                # 使用 α_i * μ_max_i 计算加权平均置信度
                alpha_k = alpha[idx_k]
                mu_max_k = membership_max[idx_k]
                num = float(np.sum(alpha_k * mu_max_k))
                den = float(np.sum(alpha_k)) if np.sum(alpha_k) > 1e-12 else 1.0
                conf_k = num / den
                cluster_conf[k] = conf_k
                """

                for cluster_info in self.clusters:
                    cluster_id = cluster_info.cluster_id
                    center_normal = cluster_info.center_normal
                    num_members = cluster_info.num_members
                    confidence = cluster_info.confidence
                    is_manual_seed = cluster_info.is_manual_seed

                    # 将一个 3D 上半球法向量 (nx, ny, nz) 转换为地质学中的倾角 (dip) 与倾向 (dip direction, dipdir)，单位为度。
                    n = center_normal / np.linalg.norm(center_normal)
                    nx, ny, nz = float(n[0]), float(n[1]), float(n[2])

                    # ---------- 步骤3: 计算倾角 dip ----------
                    # 数值安全: 限制 nz 在 [-1, 1] 范围内, 避免浮点误差导致 arccos 域错误
                    nz_clamped = max(-1.0, min(1.0, nz))
                    dip_rad = np.arccos(nz_clamped)
                    dip_deg = np.degrees(dip_rad)

                    # ---------- 步骤4: 计算最大下倾方向 -> 倾向 dipdir ----------
                    dipdir_rad = np.arctan2(nx, ny)
                    dipdir_deg = np.degrees(dipdir_rad)
                    # 将方位角归一到 [0, 360)
                    dipdir_deg = np.where(dipdir_deg < 0.0, dipdir_deg + 360.0, dipdir_deg)

                    writer.writerow([
                        cluster_id,
                        f"{dip_deg:.2f}",
                        f"{dipdir_deg:.2f}",
                        num_members,
                        confidence,
                        is_manual_seed,
                    ])
                # for cluster_id, cluster_data in sorted(clusters_dict.items()):
                #     # 计算面积加权平均倾角/倾向
                #     dip_list = np.array(cluster_data["dip_list"])
                #     dipdir_list = np.array(cluster_data["dipdir_list"])
                #     area_list = np.array(cluster_data["area_list"])
                #
                #     if np.sum(area_list) > 0:
                #         # 面积加权平均
                #         weights = area_list / np.sum(area_list)
                #         mean_dip = np.average(dip_list, weights=weights)
                #         mean_dipdir = np.average(dipdir_list, weights=weights)
                #     else:
                #         # 简单平均
                #         mean_dip = np.mean(dip_list)
                #         mean_dipdir = np.mean(dipdir_list)
                #
                #     # 确保倾向在 [0, 360) 范围内
                #     mean_dipdir = mean_dipdir % 360.0
                #
                #     writer.writerow([
                #         cluster_id,
                #         f"{mean_dip:.2f}",
                #         f"{mean_dipdir:.2f}",
                #         cluster_data["count"],
                #     ])

            self.logger.info(
                f"聚类级 CSV 导出完成: {csv_path}, 共写入 {len(self.clusters)} 行."
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
                1) 所有 inlier 点的点级 CSV;
                2) 对应凸包多边形的 PLY。
            文件命名为:
                DisconPoints_{id}_{PointsNumber}.csv
                DisconPolygon_{id}_{PointsNumber}.ply

        实现思路:
            1) 在 discon_dir 下创建目录;
            2) 对每个 disc:
                - 若无 segments 则跳过;
                - PointsNumber 取 len(set(disc.point_indices)) 若存在,
                  否则由所有 segment.point_indices 合并去重得到;
                - 构造文件名并写入:
                    * 点 CSV 的列结构与 ExportPointLevelCsv 相同,
                      但仅写当前 disc 的点;
                    * polygon PLY 使用 disc.polygon_points, 统一颜色由
                      _GenerateColorFromId(disc_id) 给出。
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
                        "A", "B", "C", "D",
                        "RMS",
                        "Curvature",
                        "DistToPlane",
                    ])

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

                        seg_id = seg_id_map.get(int(pid), -1)
                        seg = disc.segments[seg_id]
                        plane = seg.plane
                        A, B, C = plane.normal
                        D = plane.d
                        rms = plane.rmse

                        curvature = (
                            float(p.curvature)
                            if hasattr(p, "curvature")
                            else float("nan")
                        )

                        dist_to_plane = abs(A * x + B * y + C * z + D)

                        writer.writerow([
                            x, y, z,
                            nx, ny, nz,
                            r_val, g_val, b_val,
                            dr, dg, db,
                            disc_id,
                            disc.cluster_id,
                            seg_id,
                            A, B, C, D,
                            rms,
                            curvature,
                            dist_to_plane,
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
