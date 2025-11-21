import csv
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import ConvexHull
from plyfile import PlyData, PlyElement
from datetime import datetime

from .logging_utils import LoggerManager, Timer
from .geometry import Discontinuity, Plane
from .pointcloud import PointCloud


class ResultsExporter:
    """
    功能简介:
        从 PointCloud 与 Discontinuity 列表中, 生成用于分析与可视化的结果文件,
        包括:
            1) 点级 CSV (每行一个点, 含平面/结构面信息)
            2) 结构面级 CSV (每行一个 Discontinuity)
            3) 每个结构面的凸包多边形 PLY 文件(便于 Meshlab 可视化)

    实现思路(概要):
        - 在构造函数中保存 point_cloud 与 discontinuities, 以及可选的 cluster_labels。
        - 点级 CSV:
            遍历所有 Discontinuity/Segment/point_indices, 对每个点写出一行记录,
            包含点坐标/法向/颜色, 所属 Discontinuity/Segment, 以及平面参数 A,B,C,D 和 RMS。
        - 结构面级 CSV:
            对每个 Discontinuity:
                * 汇总该结构面所有点索引(去重)
                * 基于代表平面(第一个 Segment 的 plane) 和凸包计算面积(2D 投影后多边形面积)
                * 汇总 PointsNumber / TraceLength / Roughness / MeanRMS 等指标
        - 凸包 PLY:
            对每个 Discontinuity:
                * 计算结构面所有点在平面局部坐标系上的 2D 坐标, 做 ConvexHull
                * 将凸包顶点按顺序写为 PLY 的 vertex, 并写一个 face 多边形元素。

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
            5) 返回包含输出路径信息的字典, 便于上层记录或打印.

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
        }

    # =========================
    # 点级 CSV 导出
    # =========================

    def ExportPointLevelCsv(self, csv_path: str) -> None:
        """
        功能简介:
            导出点级 CSV 文件, 每一行代表一个属于某个结构面的点,
            包含几何属性和结构面/平面信息.

        实现思路:
            1) 打开 csv 文件, 写入表头:
               [X, Y, Z, nx, ny, nz, R, G, B,
                Discontinuity_id, Cluster_id, Segment_id,
                A, B, C, D, RMS]
            2) 遍历所有 Discontinuity (按列表索引作为 Discontinuity_id):
               遍历其所有 Segment (索引作为 Segment_id):
                 对于 segment.point_indices 中的每个点索引:
                     - 取出点的坐标/颜色/法向;
                     - cluster_id: 若有 cluster_labels 则用对应值, 否则为 -1;
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
                    "Discontinuity_id",
                    "Cluster_id",
                    "Segment_id",
                    "A", "B", "C", "D",
                    "RMS"
                ])

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

                            # cluster_id: 优先使用外部 cluster_labels
                            if self.cluster_labels is not None:
                                cluster_id = int(self.cluster_labels[pt_idx])
                            else:
                                cluster_id = disc_cluster_id if disc_cluster_id != -1 else -1

                            writer.writerow([
                                x, y, z,
                                nx, ny, nz,
                                R, G, Bc,
                                disc_id,
                                cluster_id,
                                seg_id,
                                A, B, C, D,
                                rms
                            ])
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
                1) 汇总所有 segment 的点索引, 去重得到 indices_all;
                2) 选择代表平面 plane_ref:
                   - 若 segments 非空, 取第一个 segment 的 plane;
                   - 若 segments 为空, 跳过该结构面;
                3) 基于 indices_all 和 plane_ref 计算:
                   - Area: 将点云投影到 plane_ref 的局部 2D 坐标系中,
                     对 2D 点做 ConvexHull, 按顶点顺序用"鞋带公式"计算面积;
                   - TraceLength: 所有 segment.trace_length 之和;
                   - PointsNumber: indices_all 的大小;
                   - MeanRMS: 各 segment plane.rmse 的平均值(若无 segment 则跳过);
                4) 写入 CSV 行:
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
                    "MeanRMS"
                ])

                for disc_id, disc in enumerate(self.discontinuities):
                    if not disc.segments:
                        continue

                    # Discontinuity 级 cluster_id
                    disc_cluster_id = getattr(disc, "cluster_id", -1)

                    # 代表平面: 取第一个 segment 的 plane
                    plane_ref: Plane = disc.segments[0].plane
                    A, B, C = plane_ref.normal
                    D = plane_ref.d

                    # 汇总点索引并去重
                    all_indices: List[int] = []
                    for seg in disc.segments:
                        all_indices.extend(seg.point_indices)
                    if not all_indices:
                        continue
                    indices_all_unique = np.unique(np.array(all_indices, dtype=int))
                    points_number = int(indices_all_unique.shape[0])

                    # 统计 TraceLength (简单求和)
                    trace_length = float(sum(seg.trace_length for seg in disc.segments))

                    # MeanRMS: 各 segment 所属平面 rmse 的平均值
                    rms_list = [seg.plane.rmse for seg in disc.segments]
                    mean_rms = float(np.mean(rms_list)) if rms_list else float("nan")

                    # 结构面 Roughness: 使用 Discontinuity.roughness
                    roughness = disc.roughness

                    # 面积 Area: 基于凸包的投影面积
                    area = self._ComputeAreaOnPlane(indices_all_unique, plane_ref)

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
                        mean_rms
                    ])
                    count_rows += 1

            self.logger.info(
                f"结构面级 CSV 导出完成: {csv_path}, 共写入 {count_rows} 行."
            )

    # =========================
    # 凸包多边形 PLY 导出
    # =========================

    def ExportDiscontinuityPolygonsToPly(
            self,
            ply_path: str
    ) -> None:
        """
        功能简介:
            将所有 Discontinuity 的凸包边界点导出到一个 PLY 文件,
            文件名形如: <basename>_polygons.ply。

        实现思路:
            对每个 Discontinuity:
                1) 汇总并去重所有点索引 indices_all_unique;
                2) 若点数 < 3, 无法形成凸包, 跳过;
                3) 使用代表平面 plane_ref (第一个 segment 的 plane);
                4) 调用 _ComputeConvexHull3D 计算凸包边界点的 3D 坐标
                   和对应的全局点索引;
                5) 对每个边界点, 取出:
                   - x, y, z
                   - r, g, b
                   - discontinuity_id
                6) 汇总所有结构面的边界点, 构造成一个 vertex 数组,
                   使用 PlyElement.describe 写为包含 vertex 的 PLY 文件。

        注意:
            - 此处仅输出 vertex, 不写 face(多边形), 方便在 Meshlab 中以点的形式
              查看每个结构面的边界, 后续可根据 discontinuity_id 做过滤或着色。
        """
        ply_path = os.path.abspath(ply_path)
        os.makedirs(os.path.dirname(ply_path), exist_ok=True)

        with Timer(f"ExportDiscontinuityPolygonsToPly({os.path.basename(ply_path)})", self.logger):
            vertex_records = []

            for disc_id, disc in enumerate(self.discontinuities):
                if not disc.segments:
                    continue

                # 汇总点索引并去重
                all_indices: List[int] = []
                for seg in disc.segments:
                    all_indices.extend(seg.point_indices)
                if not all_indices:
                    continue
                indices_all_unique = np.unique(np.array(all_indices, dtype=int))
                if indices_all_unique.shape[0] < 3:
                    continue

                # 代表平面
                plane_ref: Plane = disc.segments[0].plane

                # 计算 3D 凸包边界点
                boundary_points_3d, boundary_point_ids = self._ComputeConvexHull3D(
                    indices_all_unique,
                    plane_ref
                )
                if boundary_points_3d.shape[0] < 3:
                    continue

                # 为每个边界点构造一条记录
                for i, pt_idx in enumerate(boundary_point_ids):
                    p = self.point_cloud.points[int(pt_idx)]
                    vertex_records.append((
                        float(p.x),
                        float(p.y),
                        float(p.z),
                        int(max(0, min(255, p.r))),
                        int(max(0, min(255, p.g))),
                        int(max(0, min(255, p.b))),
                        int(disc_id),
                    ))

            if not vertex_records:
                self.logger.warning("没有可导出的凸包边界点, 跳过写 PLY。")
                return

            vertex_dtype = np.dtype([
                ("x", "f4"),
                ("y", "f4"),
                ("z", "f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
                ("discontinuity_id", "i4"),
            ])
            vertex_array = np.array(vertex_records, dtype=vertex_dtype)

            el_verts = PlyElement.describe(vertex_array, "vertex")
            PlyData([el_verts], text=False).write(ply_path)

            self.logger.info(
                f"结构面凸包边界 PLY 导出完成: {ply_path}, "
                f"共 {vertex_array.shape[0]} 个边界点."
            )

    # =========================
    # 辅助函数: 在平面上计算面积
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

    def _ComputeAreaOnPlane(
            self,
            point_indices: np.ndarray,
            plane: Plane
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
            [[self.point_cloud.points[i].x,
              self.point_cloud.points[i].y,
              self.point_cloud.points[i].z]
             for i in indices_unique],
            dtype=np.float64
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
            plane: Plane
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
            [[self.point_cloud.points[i].x,
              self.point_cloud.points[i].y,
              self.point_cloud.points[i].z]
             for i in indices_unique],
            dtype=np.float64
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
