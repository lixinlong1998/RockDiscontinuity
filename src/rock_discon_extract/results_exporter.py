import csv
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import ConvexHull
from datetime import datetime

from .logging_utils import LoggerManager, Timer
from .geometry import Discontinuity, Plane
from .pointcloud import PointCloud


class ResultsExporter:
    """
    功能简介:
        从 PointCloud 与 Discontinuity 列表中, 生成用于分析与可视化的结果文件,
        包括:
            1) 点级 CSV (每行一个点, 含平面/结构面信息以及局部曲率、点到平面距离)
            2) 结构面级 CSV (每行一个 Discontinuity)
            3) 所有结构面的凸包多边形 PLY 文件(便于 MeshLab / CloudCompare 可视化,
               当前实现为 vertex + face, 同时可选 edge)

    实现思路(概要):
        - 在构造函数中保存 point_cloud 与 discontinuities, 以及可选的 cluster_labels。
        - 点级 CSV:
            遍历所有 Discontinuity/Segment/point_indices, 对每个点写出一行记录,
            包含点坐标/法向/颜色, 所属 Discontinuity/Segment, 以及平面参数 A,B,C,D,
            平均拟合误差 RMS、局部曲率 Curvature、点到平面的距离 DistToPlane。
        - 结构面级 CSV:
            对每个 Discontinuity:
                * 汇总该结构面所有点索引(去重)
                * 基于代表平面 plane_ref 和结构面自身的凸包计算面积
                * 汇总 PointsNumber / TraceLength / Roughness / MeanRMS 等指标
        - 凸包 PLY:
            对每个 Discontinuity:
                * 使用 Discontinuity 中预计算好的 polygon_points (凸包边界点)
                * 将边界点按顺序构造三角面(扇形剖分), 以 PLY 的 vertex + face 形式导出;
                * 每个结构面使用统一颜色, 有利于在可视化软件中区分不同结构面。

    输入:
        point_cloud: PointCloud
            包含所有点的点云对象, 点中应已包含坐标/颜色/法向等信息.
        discontinuities: List[Discontinuity]
            结构面结果列表, 每个 Discontinuity 中包含一个或多个 Segment。
        cluster_labels: Optional[np.ndarray]
            可选, 形状为 (N,), N 为点云点数。若给定, 每个点的 cluster_id
            将从该数组中读取; 若为 None, 则导出时 cluster_id 默认为 -1。
        algorithm_name: str
            用于在日志中标识当前算法来源, 也可用于输出文件命名。
    """

    def __init__(
            self,
            point_cloud: PointCloud,
            discontinuities: List[Discontinuity],
            cluster_labels: Optional[np.ndarray] = None,
            algorithm_name: str = "",
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

    def ExportAll(
            self,
            result_root_dir: str,
            point_cloud_path: str,
    ) -> Dict[str, str]:
        """
        功能简介:
            一次性导出三类结果文件:
                - 点级 CSV: <basename>_points.csv
                - 结构面级 CSV: <basename>_discontinuitys.csv
                - 凸包边界点 PLY: <basename>_polygons.ply

        实现思路:
            1) 调用 _CreateOutputSubdir 在 result_root_dir 下创建
               "YYYYMMDD_HHMMSS_算法名" 子目录;
            2) 从 point_cloud_path 提取 basename, 去掉扩展名;
            3) 在子目录中组合三个文件路径:
               - basename_points.csv
               - basename_discontinuitys.csv
               - basename_polygons.ply
            4) 分别调用 ExportPointLevelCsv / ExportDiscontinuityLevelCsv /
               ExportDiscontinuityPolygonsToPly;
            5) 返回包含输出路径信息的字典, 便于上层记录或打印。

        输入:
            result_root_dir: str
                结果根目录.
            point_cloud_path: str
                输入点云文件路径, 用于提取文件名前缀.

        输出:
            paths: Dict[str, str]
                包含 "dir", "points_csv", "disc_csv", "polygons_ply".
        """
        out_dir = self._CreateOutputSubdir(result_root_dir)
        base_name = os.path.splitext(os.path.basename(point_cloud_path))[0]

        points_csv = os.path.join(out_dir, f"{base_name}_points.csv")
        disc_csv = os.path.join(out_dir, f"{base_name}_discontinuitys.csv")
        polygons_ply = os.path.join(out_dir, f"{base_name}_polygons.ply")

        self.ExportPointLevelCsv(points_csv)
        self.ExportDiscontinuityLevelCsv(disc_csv)
        self.ExportDiscontinuityPolygonsToPly(polygons_ply)

        return {
            "dir": out_dir,
            "points_csv": points_csv,
            "disc_csv": disc_csv,
            "polygons_ply": polygons_ply,
            "point_cloud_path": point_cloud_path
        }

    # =========================
    # 点级 CSV 导出
    # =========================

    def ExportPointLevelCsv(self, csv_path: str) -> None:
        """
        功能简介:
            导出点级 CSV 文件, 每一行代表一个属于某个结构面的点,
            包含几何属性、结构面/平面信息, 以及局部曲率与点到平面的距离。

        实现思路:
            1) 打开 csv 文件, 写入表头:
               [X, Y, Z, nx, ny, nz, R, G, B,
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
                     - 写入一行 CSV.
            3) 记录总写入点数并输出日志。

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
                writer.writerow(
                    [
                        "X",
                        "Y",
                        "Z",
                        "nx",
                        "ny",
                        "nz",
                        "R",
                        "G",
                        "B",
                        "Discontinuity_id",
                        "Cluster_id",
                        "Segment_id",
                        "A",
                        "B",
                        "C",
                        "D",
                        "RMS",
                        "Curvature",
                        "DistToPlane",
                    ]
                )

                for disc_id, disc in enumerate(self.discontinuities):
                    # 若 Discontinuity 对象上有 cluster_id 属性则使用, 否则默认 -1
                    disc_cluster_id = getattr(disc, "cluster_id", -1)

                    for seg_id, seg in enumerate(disc.segments):
                        plane = seg.plane
                        A, B, C = plane.normal
                        D = plane.d
                        rms = plane.rmse

                        for pt_idx in seg.point_indices:
                            p = self.point_cloud.points[pt_idx]

                            # 坐标
                            x, y, z = p.x, p.y, p.z

                            # 法向: 若 point.normal 为空则导出 NaN
                            if p.normal is not None:
                                nx, ny, nz = p.normal
                            else:
                                nx = ny = nz = float("nan")

                            # 颜色
                            R = p.r
                            G = p.g
                            Bc = p.b  # 避免与变量 B 重名

                            # 曲率: 若未赋值则导出 NaN
                            curvature = (
                                float(p.curvature)
                                if hasattr(p, "curvature")
                                else float("nan")
                            )

                            # cluster_id: 优先使用外部 cluster_labels
                            if self.cluster_labels is not None:
                                cluster_id = int(self.cluster_labels[pt_idx])
                            else:
                                cluster_id = disc_cluster_id if disc_cluster_id != -1 else -1

                            # 点到平面的距离(绝对值)
                            dist_to_plane = abs(A * x + B * y + C * z + D)

                            writer.writerow(
                                [
                                    x,
                                    y,
                                    z,
                                    nx,
                                    ny,
                                    nz,
                                    R,
                                    G,
                                    Bc,
                                    disc_id,
                                    cluster_id,
                                    seg_id,
                                    A,
                                    B,
                                    C,
                                    D,
                                    rms,
                                    curvature,
                                    dist_to_plane,
                                ]
                            )
                            count_rows += 1

            self.logger.info(
                f"点级 CSV 导出完成: {csv_path}, 共写入 {count_rows} 行."
            )

    # =========================
    # 结构面级 CSV 导出
    # =========================

    def ExportDiscontinuityLevelCsv(self, csv_path: str) -> None:
        """
        功能简介:
            导出结构面级 CSV 文件, 每一行代表一个 Discontinuity,
            包含倾角/倾向、平面参数、面积等统计信息.

        实现思路:
            对每个 Discontinuity:
                1) 使用其自身的 point_indices 计算 PointsNumber;
                2) 选择代表平面 plane_ref:
                   - 若 segments 非空, 取第一个 segment 的 plane;
                   - 若 segments 为空, 跳过该结构面;
                3) 基于 plane_ref 和 Discontinuity.ComputePolygonAndArea 计算:
                   - Area: 凸包多边形面积;
                4) TraceLength: 所有 segment.trace_length 之和;
                5) MeanRMS: 各 segment plane.rmse 的平均值(若无 segment 则跳过);
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

        with Timer(
                f"ExportDiscontinuityLevelCsv({os.path.basename(csv_path)})", self.logger
        ):
            count_rows = 0

            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                # 表头
                writer.writerow(
                    [
                        "Discontinuity_id",
                        "Cluster_id",
                        "Dip",
                        "Dipdir",
                        "A",
                        "B",
                        "C",
                        "D",
                        "PointsNumber",
                        "Area",
                        "TraceLength",
                        "Roughness",
                        "MeanRMS",
                    ]
                )

                for disc_id, disc in enumerate(self.discontinuities):
                    if not disc.segments:
                        continue

                    disc_cluster_id = getattr(disc, "cluster_id", -1)

                    # 代表平面: 取第一个 segment 的 plane
                    plane_ref = disc.segments[0].plane
                    A, B, C = plane_ref.normal
                    D = plane_ref.d

                    # 点数: 使用 Discontinuity.point_indices
                    if getattr(disc, "point_indices", None):
                        points_number = len(set(disc.point_indices))
                    else:
                        points_number = 0

                    # TraceLength: 所有 segment.trace_length 之和
                    trace_length = float(sum(seg.trace_length for seg in disc.segments))

                    # MeanRMS: 各 segment 所属平面 rmse 的平均值
                    rms_list = [seg.plane.rmse for seg in disc.segments]
                    mean_rms = float(np.mean(rms_list)) if rms_list else float("nan")

                    roughness = disc.roughness

                    # 面积: 通过 Discontinuity 自身的方法计算
                    disc.ComputePolygonAndArea(self.point_cloud, plane_ref)
                    area = float(getattr(disc, "area", 0.0))

                    writer.writerow(
                        [
                            disc_id,
                            disc_cluster_id,
                            disc.dip,
                            disc.dip_direction,
                            A,
                            B,
                            C,
                            D,
                            points_number,
                            area,
                            trace_length,
                            roughness,
                            mean_rms,
                        ]
                    )
                    count_rows += 1

            self.logger.info(
                f"结构面级 CSV 导出完成: {csv_path}, 共写入 {count_rows} 行."
            )

    # =========================
    # 凸包多边形 PLY 导出
    # =========================

    def ExportDiscontinuityPolygonsToPly(self, ply_path: str) -> None:
        """
        功能简介:
            将所有 Discontinuity 的凸包边界点导出到一个 PLY 文件,
            每个结构面的边界以点 + 三角面(face) 的形式表示, 可选增加边(edge),
            便于在 MeshLab / CloudCompare 中查看多边形边界与面片。

        实现思路:
            对每个 Discontinuity:
                1) 调用 disc.ComputePolygonAndArea(self.point_cloud) 保证 polygon_points 可用;
                2) 从 disc.polygon_points 追加到全局顶点列表;
                3) 颜色: 按结构面 ID 生成统一 RGB 颜色, 每个结构面所有边界点
                   颜色相同, 方便区分不同结构面;
                4) 为该结构面顶点构造一圈 edges (可选), 按顺序连接并首尾相连;
                5) 为该结构面顶点构造三角面 faces:
                   - 采用简单扇形剖分: (0,1,2), (0,2,3), ..., (0,m-2,m-1);
            最后调用 _ExportToMeshlabPly:
                - vertices: 所有结构面的边界点
                - edges: 所有结构面的边界边(当前可留空或仅作辅线)
                - faces: 所有结构面的三角面片
                - colors: 对应的顶点颜色
        """
        ply_path = os.path.abspath(ply_path)
        os.makedirs(os.path.dirname(ply_path), exist_ok=True)

        with Timer(
                f"ExportDiscontinuityPolygonsToPly({os.path.basename(ply_path)})",
                self.logger,
        ):
            vertices_list: List[np.ndarray] = []
            colors_list: List[np.ndarray] = []
            edges_list: List[np.ndarray] = []
            faces_list: List[np.ndarray] = []
            offset = 0  # 全局顶点索引偏移量

            for disc_id, disc in enumerate(self.discontinuities):
                if not disc.segments:
                    continue

                # 确保 polygon_points 已计算
                disc.ComputePolygonAndArea(self.point_cloud)
                pts = getattr(disc, "polygon_points", None)
                if pts is None or pts.shape[0] < 2:
                    continue

                pts = np.asarray(pts, dtype=np.float32)
                m = pts.shape[0]
                vertices_list.append(pts)

                # 颜色: 按 Discontinuity ID 分配统一颜色
                base_color = self._GenerateColorFromId(disc_id)
                colors_disc = np.tile(
                    np.array(base_color, dtype=np.uint8), (m, 1)
                )  # (m, 3)
                colors_list.append(colors_disc)

                # 构造边: (offset+i, offset+i+1), 首尾相连(可选)
                edges_disc: List[List[int]] = []
                if m >= 2:
                    for i in range(m - 1):
                        edges_disc.append([offset + i, offset + i + 1])
                    if m > 2:
                        edges_disc.append([offset + m - 1, offset])  # 闭合
                if edges_disc:
                    edges_list.append(np.asarray(edges_disc, dtype=np.int32))

                # 构造三角面: 简单扇形剖分
                faces_disc: List[List[int]] = []
                if m >= 3:
                    for i in range(1, m - 1):
                        faces_disc.append([offset, offset + i, offset + i + 1])
                if faces_disc:
                    faces_list.append(np.asarray(faces_disc, dtype=np.int32))

                offset += m

            if not vertices_list:
                self.logger.warning("没有可导出的凸包边界点, 跳过写 PLY。")
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
                colors=colors,
            )

            self.logger.info(
                f"结构面凸包边界 PLY 导出完成: {ply_path}, "
                f"共 {vertices.shape[0]} 个边界点, "
                f"{edges.shape[0] if edges is not None else 0} 条边, "
                f"{faces.shape[0] if faces is not None else 0} 个三角面."
            )

    @staticmethod
    def _ExportToMeshlabPly(
            filename, vertices=None, edges=None, faces=None, colors=None
    ):
        """
        功能简介:
            导出点、线、面到 PLY 文件，支持 MeshLab / CloudCompare 可视化.

        实现思路:
            - 按 PLY ASCII 格式写入 header 和各 element 数据;
            - 顶点部分支持附带 RGB 颜色;
            - 边部分为 element edge, 每行 "vertex1 vertex2";
            - 面部分为 element face, 每行 "3 v0 v1 v2"。
            只要传入对应数组即可, 不要求三者必须同时存在。

        参数:
            filename: str
                输出文件名（如 "output.ply"）
            vertices: np.ndarray
                顶点数组，shape=(N, 3)，每行是 [x, y, z]
            edges: np.ndarray 或 None
                边数组，shape=(M, 2)，每行是 [vertex_idx1, vertex_idx2]
            faces: np.ndarray 或 None
                面数组，shape=(K, 3)，每行是 [vertex_idx1, vertex_idx2, vertex_idx3]
            colors: np.ndarray 或 None
                顶点颜色，shape=(N, 3)，每行是 [r, g, b]（0-255）
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

    # =========================
    # 辅助函数: 创建输出子目录
    # =========================

    def _CreateOutputSubdir(self, result_root_dir: str) -> str:
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
        subdir_name = f"{timestamp}_{safe_algo}"
        out_dir = os.path.join(result_root_dir, subdir_name)

        os.makedirs(out_dir, exist_ok=True)
        self.logger.info(f"结果输出子目录: {out_dir}")
        return out_dir

    # =========================
    # 辅助函数: 在平面上计算面积
    # =========================

    def _ComputeAreaOnPlane(
            self,
            point_indices: np.ndarray,
            plane: Plane,
    ) -> float:
        """
        【未经验证】功能简介:
            对给定平面 plane 上的一组点, 通过平面投影 + 凸包,
            计算该点集在该平面上的近似面积(单位与坐标单位的平方一致).

        【未经验证】实现思路:
            1) 构造平面法向 n = plane.normal 并单位化;
            2) 选取 ref 向量:
               - 若 |n·(0,0,1)| < 0.9, 则 ref=(0,0,1); 否则 ref=(0,1,0);
            3) 构造局部坐标系:
               - u = normalize(ref × n)
               - v = n × u
            4) 对每个点 p:
               - p3 = (p.x, p.y, p.z)
               - q = p3 - centroid
               - 2D 坐标为 (q·u, q·v)
            5) 使用 ConvexHull 对 2D 点集做凸包, 得到顶点顺序 hull.vertices;
            6) 使用“鞋带公式”计算多边形面积:
               area = 0.5 * |sum(x_i*y_{i+1} - y_i*x_{i+1})|.

        输入:
            point_indices: np.ndarray
                一维整型数组, 结构面所有点的全局索引(可重复, 内部会去重).
            plane: Plane
                代表该结构面的平面.

        输出:
            area: float
                近似面积, 若点数不足 3 或凸包失败, 则返回 0.0.
        """
        indices_unique = np.unique(point_indices.astype(int))
        if indices_unique.shape[0] < 3:
            return 0.0

        # 提取 3D 坐标
        pts3d = np.array(
            [
                [
                    self.point_cloud.points[i].x,
                    self.point_cloud.points[i].y,
                    self.point_cloud.points[i].z,
                ]
                for i in indices_unique
            ],
            dtype=np.float64,
        )

        # 法向与质心
        n = np.array(plane.normal, dtype=np.float64)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            return 0.0
        n = n / n_norm
        centroid = np.array(plane.centroid, dtype=np.float64)

        # 构造局部坐标系 (u, v)
        ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(np.dot(n, ref)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        u = np.cross(ref, n)
        u_norm = np.linalg.norm(u)
        if u_norm < 1e-12:
            return 0.0
        u = u / u_norm
        v = np.cross(n, u)

        # 投影到 2D
        q = pts3d - centroid  # (N, 3)
        u_coord = q @ u  # (N,)
        v_coord = q @ v  # (N,)
        pts2d = np.stack([u_coord, v_coord], axis=1)

        try:
            hull = ConvexHull(pts2d)
        except Exception as e:
            self.logger.warning(f"ConvexHull 计算失败, 返回面积 0.0, 错误: {e}")
            return 0.0

        hull_indices = hull.vertices
        poly = pts2d[hull_indices]
        x = poly[:, 0]
        y = poly[:, 1]
        # 鞋带公式
        area = 0.5 * abs(
            np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
        )
        return float(area)

    # =========================
    # 辅助函数: 计算 3D 凸包顶点顺序(用于 PLY 导出)
    # =========================

    def _ComputeConvexHull3D(
            self,
            point_indices: np.ndarray,
            plane: Plane,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        【未经验证】功能简介:
            对给定平面 plane 上的一组点, 计算其凸包边界的 3D 顶点集合,
            返回边界点的 3D 坐标和对应的全局点索引(按凸包顺序).

        实现思路:
            与 _ComputeAreaOnPlane 类似, 先在平面上构造局部 2D 坐标系并计算 2D 凸包,
            再用凸包顶点顺序索引回 3D 点坐标。

        输入:
            point_indices: np.ndarray
                一维整型数组, 点云中的全局索引.
            plane: Plane
                代表平面.

        输出:
            boundary_points_3d: np.ndarray, 形状 (M, 3)
                凸包边界点的 3D 坐标, 按逆时针或顺时针顺序排列.
            boundary_indices_global: np.ndarray, 形状 (M,)
                对应的全局点索引.
        """
        indices_unique = np.unique(point_indices.astype(int))
        if indices_unique.shape[0] < 3:
            return np.empty((0, 3), dtype=np.float64), indices_unique

        # 提取 3D 坐标
        pts3d = np.array(
            [
                [
                    self.point_cloud.points[i].x,
                    self.point_cloud.points[i].y,
                    self.point_cloud.points[i].z,
                ]
                for i in indices_unique
            ],
            dtype=np.float64,
        )

        # 法向与质心
        n = np.array(plane.normal, dtype=np.float64)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            return pts3d, indices_unique
        n = n / n_norm
        centroid = np.array(plane.centroid, dtype=np.float64)

        # 构造局部坐标系 (u, v)
        ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(np.dot(n, ref)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        u = np.cross(ref, n)
        u_norm = np.linalg.norm(u)
        if u_norm < 1e-12:
            return pts3d, indices_unique
        u = u / u_norm
        v = np.cross(n, u)

        # 投影到 2D
        q = pts3d - centroid
        u_coord = q @ u
        v_coord = q @ v
        pts2d = np.stack([u_coord, v_coord], axis=1)

        try:
            hull = ConvexHull(pts2d)
        except Exception as e:
            self.logger.warning(f"ConvexHull 计算失败, 使用全部点作为边界, 错误: {e}")
            return pts3d, indices_unique

        hull_indices_local = hull.vertices  # 在 indices_unique 内部的索引
        boundary_points_3d = pts3d[hull_indices_local]
        boundary_indices_global = indices_unique[hull_indices_local]

        return boundary_points_3d, boundary_indices_global

    # =========================
    # 辅助函数: 按结构面 ID 生成颜色
    # =========================

    @staticmethod
    def _GenerateColorFromId(disc_id: int) -> Tuple[int, int, int]:
        """
        功能简介:
            根据结构面 ID 生成一个可重复的 RGB 颜色, 用于多边形 PLY 着色。

        实现思路:
            使用简单的取模运算, 将 disc_id 映射到 [50, 255] 区间内的 RGB 值,
            尽量避免过暗颜色, 并保证不同 ID 颜色有一定差异。
        """
        base = disc_id + 1
        r = 50 + (base * 73) % 205
        g = 50 + (base * 131) % 205
        b = 50 + (base * 197) % 205
        return int(r), int(g), int(b)
