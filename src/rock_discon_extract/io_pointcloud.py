import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from plyfile import PlyData
import laspy

from .logging_utils import LoggerManager, Timer
from .geometry import Point
from .pointcloud import PointCloud


class PointCloudIO:
    """
    功能简介:
        提供统一的点云读取接口, 支持 PLY / XYZ / LAS 等常见格式,
        自动识别坐标、法向、颜色等属性, 并可选择构造 PointCloud 对象.

    实现思路:
        1) 根据文件后缀判断点云格式, 分发到对应的私有读取函数:
           - _ReadPlyGeneric
           - _ReadLasGeneric
           - _ReadXyzGeneric
        2) 每个读取函数返回:
           - points: np.ndarray, 形状 (N, 3), float64
           - normals: Optional[np.ndarray], 形状 (N, 3), float64 或 None
           - colors: Optional[np.ndarray], 形状 (N, 3), uint8 或 None
           - extra_attrs: Dict[str, np.ndarray], 其他属性字段
        3) 在需要时, 将上述数组进一步封装为 Point 列表和 PointCloud 对象.

    输入约定:
        - file_path: str
            点云文件路径, 支持 .ply / .las / .laz / .xyz / .txt 等后缀.
        - attach_extra_attrs: bool
            在构造 PointCloud 时是否将 extra_attrs 填入 Point.features 中。
            若为 True, 会对每个点逐属性赋值, 在超大点云上可能有一定性能开销。

    输出约定:
        - ReadPointCloudArrays:
            返回 (points, normals, colors, extra_attrs)
        - ReadPointCloudAsObjects:
            返回 PointCloud 对象, 内部 Point 已附带 coords / color / normal / features 等信息.
    """

    logger = LoggerManager.GetLogger("PointCloudIO")

    # =========================
    # 对外接口
    # =========================

    @classmethod
    def ReadPointCloudArrays(
        cls,
        file_path: str
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Dict[str, np.ndarray]]:
        """
        功能简介:
            读取点云文件, 返回坐标 / 法向 / 颜色 / 其他属性的 numpy 数组表示.

        实现思路:
            1) 根据文件扩展名判断格式:
               - ".ply"   -> _ReadPlyGeneric
               - ".las"/".laz" -> _ReadLasGeneric
               - ".xyz"/".txt" -> _ReadXyzGeneric
               其他后缀暂不支持, 抛出异常.
            2) 在内部使用 Timer 统计整体读取耗时, 并记录日志.

        输入:
            file_path: str
                点云文件路径.

        输出:
            points: np.ndarray
                形状 (N, 3), float64, 表示点的 (x, y, z) 坐标.
            normals: Optional[np.ndarray]
                若文件包含法向信息, 则为形状 (N, 3), float64; 否则为 None.
            colors: Optional[np.ndarray]
                若文件包含颜色信息, 则为形状 (N, 3), uint8; 否则为 None.
            extra_attrs: Dict[str, np.ndarray]
                其他所有属性字段, 键为属性名, 值为一维数组 (N,).
        """
        file_path = os.path.abspath(file_path)
        ext = os.path.splitext(file_path)[1].lower()

        with Timer(f"ReadPointCloudArrays({os.path.basename(file_path)})", cls.logger):
            if ext == ".ply":
                return cls._ReadPlyGeneric(file_path)
            elif ext in (".las", ".laz"):
                return cls._ReadLasGeneric(file_path)
            elif ext in (".xyz", ".txt"):
                return cls._ReadXyzGeneric(file_path)
            else:
                raise ValueError(f"不支持的点云格式: {ext}")

    @classmethod
    def ReadPointCloudAsObjects(
        cls,
        file_path: str,
        attach_extra_attrs: bool = False
    ) -> PointCloud:
        """
        功能简介:
            读取点云文件并构造 PointCloud 对象, 兼顾几何坐标、法向和颜色等属性.

        实现思路:
            1) 调用 ReadPointCloudArrays 获取 points / normals / colors / extra_attrs;
            2) 遍历每一个点 i:
               - 从 points[i] 提取 (x, y, z);
               - 若 colors 不为空, 提取 (r, g, b), 否则设为 0;
               - 若 normals 不为空, 提取 (nx, ny, nz), 否则 normal=None;
               - 若 attach_extra_attrs=True, 则将 extra_attrs 中每个字段在 i 处的值
                 填入 Point.features 字典中, 键为字段名, 值为标量.
            3) 将生成的 Point 列表传入 PointCloud 构造函数.

        输入:
            file_path: str
                点云文件路径.
            attach_extra_attrs: bool
                是否将 extra_attrs 中的信息附加到 Point.features 中。
                对于大规模点云, 建议先设为 False 做算法验证, 需要时再打开。

        输出:
            cloud: PointCloud
                含有所有点的 PointCloud 对象.
        """
        points, normals, colors, extra_attrs = cls.ReadPointCloudArrays(file_path)
        num_points = points.shape[0]

        cls.logger.info(
            f"构造 PointCloud 对象: N={num_points}, "
            f"has_normals={normals is not None}, "
            f"has_colors={colors is not None}, "
            f"extra_attrs={list(extra_attrs.keys())}"
        )

        point_list: List[Point] = []

        for i in range(num_points):
            x, y, z = float(points[i, 0]), float(points[i, 1]), float(points[i, 2])

            if colors is not None:
                r = float(colors[i, 0])
                g = float(colors[i, 1])
                b = float(colors[i, 2])
            else:
                r = g = b = 0.0

            if normals is not None:
                nx = float(normals[i, 0])
                ny = float(normals[i, 1])
                nz = float(normals[i, 2])
                normal = (nx, ny, nz)
            else:
                normal = None

            if attach_extra_attrs:
                features: Dict[str, float] = {}
                for key, arr in extra_attrs.items():
                    # 对于每个属性, 提取该点对应的标量值
                    # 注意: 这里假定 extra_attrs[key] 是一维数组, 长度为 N
                    features[key] = float(arr[i])
            else:
                features = {}

            p = Point(
                x=x,
                y=y,
                z=z,
                r=r,
                g=g,
                b=b,
                intensity=0.0,  # 可在后续根据某个 extra_attr 映射
                normal=normal,
                curvature=0.0,
                features=features,
                point_id=i
            )
            point_list.append(p)

        cloud = PointCloud(point_list)
        return cloud

    # =========================
    # PLY 读取实现
    # =========================

    @classmethod
    def _ReadPlyGeneric(
        cls,
        file_path: str
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Dict[str, np.ndarray]]:
        """
        功能简介:
            读取 PLY 点云文件, 自动提取坐标 / 法向 / 颜色 等属性信息.

        实现思路:
            1) 使用 plyfile.PlyData 读取 PLY 文件, 获取 "vertex" 元素结构化数组;
            2) 坐标:
               - 要求字段包含 "x", "y", "z", 否则抛出异常;
               - 组合为 (N, 3) 的 float64 数组.
            3) 法向:
               - 优先查找字段 {"nx", "ny", "nz"};
               - 若不存在, 再查找 {"normal_x", "normal_y", "normal_z"};
               - 若仍不存在, 返回 normals=None.
            4) 颜色:
               - 优先查找字段 {"red", "green", "blue"};
               - 若不存在, 再查找 {"r", "g", "b"};
               - 若仍不存在, 返回 colors=None.
            5) extra_attrs:
               - 对于 vertex 中除 x/y/z, 法向, 颜色之外的所有字段,
                 将其值转换为一维 numpy 数组, 以字段名为键存入字典.

        输入:
            file_path: str
                PLY 文件路径.

        输出:
            points: np.ndarray
                (N, 3), float64, 坐标.
            normals: Optional[np.ndarray]
                (N, 3), float64 或 None.
            colors: Optional[np.ndarray]
                (N, 3), uint8 或 None.
            extra_attrs: Dict[str, np.ndarray]
                其他属性字段.
        """
        cls.logger.info(f"读取 PLY 点云: {file_path}")
        ply = PlyData.read(file_path)

        if "vertex" not in ply:
            raise ValueError("PLY 文件中缺少 'vertex' 元素。")

        v = ply["vertex"].data  # structured array
        names = v.dtype.names
        if names is None:
            raise ValueError("PLY 'vertex' 元素字段信息为空。")

        name_set = set(names)
        required_xyz = {"x", "y", "z"}
        if not required_xyz.issubset(name_set):
            raise ValueError("PLY 'vertex' 元素中缺少 x/y/z 字段。")

        # 坐标
        points = np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float64, copy=False)

        # 法向: 支持 nx,ny,nz 或 normal_x,normal_y,normal_z
        normals: Optional[np.ndarray] = None
        normal_keys_candidate = [("nx", "ny", "nz"), ("normal_x", "normal_y", "normal_z")]
        for kx, ky, kz in normal_keys_candidate:
            if {kx, ky, kz}.issubset(name_set):
                normals = np.vstack([v[kx], v[ky], v[kz]]).T.astype(np.float64, copy=False)
                cls.logger.info(f"检测到法向字段: {kx}, {ky}, {kz}")
                break

        # 颜色: 支持 red,green,blue 或 r,g,b
        colors: Optional[np.ndarray] = None
        color_keys_candidate = [("red", "green", "blue"), ("r", "g", "b")]
        for cr, cg, cb in color_keys_candidate:
            if {cr, cg, cb}.issubset(name_set):
                colors = np.vstack([v[cr], v[cg], v[cb]]).T
                # 保留原始类型(一般是 uint8), 若需要可在外部转 float
                cls.logger.info(f"检测到颜色字段: {cr}, {cg}, {cb}")
                break

        # 其他属性字段
        used_keys = {"x", "y", "z"}
        if normals is not None:
            for k in normal_keys_candidate:
                used_keys.update(k)
        if colors is not None:
            for k in color_keys_candidate:
                used_keys.update(k)

        extra_attrs: Dict[str, np.ndarray] = {}
        for name in names:
            if name in used_keys:
                continue
            extra_attrs[name] = np.asarray(v[name])
        if extra_attrs:
            cls.logger.info(f"PLY 额外属性字段: {list(extra_attrs.keys())}")

        return points, normals, colors, extra_attrs

    # =========================
    # LAS/LAZ 读取实现
    # =========================

    @classmethod
    def _ReadLasGeneric(
        cls,
        file_path: str
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Dict[str, np.ndarray]]:
        """
        功能简介:
            读取 LAS/LAZ 点云文件, 自动提取坐标 / 颜色 等属性信息,
            并尝试识别可能存在的法向字段.

        实现思路:
            1) 使用 laspy.read 读取 LAS/LAZ 文件;
            2) 坐标:
               - 使用 las.x, las.y, las.z 组合为 (N, 3) float64 数组;
            3) 颜色:
               - 若点格式包含 red/green/blue, 则组合为 (N, 3) uint16/uint8 数组;
            4) 法向【推测】:
               - 若维度中存在 "nx,ny,nz" 或 "normal_x,normal_y,normal_z", 则组合为法向数组;
               - 多数 LAS 不包含法向, 此时返回 normals=None;
            5) extra_attrs:
               - 对于除坐标 / 颜色 / 法向外的所有维度, 统一放入 extra_attrs 字典.

        输入:
            file_path: str
                LAS/LAZ 文件路径.

        输出:
            points: np.ndarray
            normals: Optional[np.ndarray]
            colors: Optional[np.ndarray]
            extra_attrs: Dict[str, np.ndarray]
        """
        cls.logger.info(f"读取 LAS/LAZ 点云: {file_path}")
        las = laspy.read(file_path)

        # 坐标
        points = np.vstack([las.x, las.y, las.z]).T.astype(np.float64, copy=False)

        # 颜色
        dim_names = list(las.point_format.dimension_names)
        dim_set = set(dim_names)
        colors: Optional[np.ndarray] = None
        if {"red", "green", "blue"}.issubset(dim_set):
            colors = np.vstack([las.red, las.green, las.blue]).T
            cls.logger.info("检测到 LAS 颜色字段: red, green, blue")

        # 法向【推测】: LAS 中通常不含法向, 此处仅作简单尝试
        normals: Optional[np.ndarray] = None
        normal_keys_candidate = [("nx", "ny", "nz"), ("normal_x", "normal_y", "normal_z")]
        for kx, ky, kz in normal_keys_candidate:
            if {kx, ky, kz}.issubset(dim_set):
                nx_arr = getattr(las, kx)
                ny_arr = getattr(las, ky)
                nz_arr = getattr(las, kz)
                normals = np.vstack([nx_arr, ny_arr, nz_arr]).T.astype(np.float64, copy=False)
                cls.logger.info(f"检测到 LAS 法向字段: {kx}, {ky}, {kz}")
                break

        # 其他属性字段
        excluded = {
            "x", "y", "z",
            "red", "green", "blue",
            "nx", "ny", "nz",
            "normal_x", "normal_y", "normal_z",
        }
        extra_attrs: Dict[str, np.ndarray] = {}
        for name in dim_names:
            if name in excluded:
                continue
            data = getattr(las, name)
            extra_attrs[name] = np.asarray(data)
        if extra_attrs:
            cls.logger.info(f"LAS 额外属性字段: {list(extra_attrs.keys())}")

        return points, normals, colors, extra_attrs

    # =========================
    # XYZ 读取实现
    # =========================

    @classmethod
    def _ReadXyzGeneric(
        cls,
        file_path: str
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Dict[str, np.ndarray]]:
        """
        【未经验证】功能简介:
            读取 XYZ 文本点云文件, 尝试根据列数与数值范围推断法向和颜色信息。

        【未经验证】实现思路:
            1) 使用 numpy.loadtxt 读取文本文件, 默认以空白字符分隔, 忽略以 '#' 开头的行;
            2) 要求至少 3 列, 前 3 列视为 x,y,z 坐标;
            3) 若总列数 >= 6, 则对第 4~n 列进行启发式判断:
               - 【推测】若第 4~6 列绝对值均在 [-1.1, 1.1] 内, 视为法向 (nx,ny,nz);
               - 【推测】若最后 3 列数值在 [0,255] 且接近整数, 视为颜色 (R,G,B);
               - 未被使用的列以 attr{原始列索引} 的形式加入 extra_attrs.
            4) 若列数在 4~5 或其他不符合上述模式的情况, 所有额外列统一作为 extra_attrs,
               不作法向/颜色推断。

        注意:
            - 由于 XYZ 文件通常不包含显式字段名, 该方法对法向/颜色的识别是启发式的,
              在不同数据源上不保证完全正确, 建议在关键实验前手动检查一次列含义。

        输入:
            file_path: str
                XYZ 文本点云路径.

        输出:
            points: np.ndarray
            normals: Optional[np.ndarray]
            colors: Optional[np.ndarray]
            extra_attrs: Dict[str, np.ndarray]
        """
        cls.logger.info(f"读取 XYZ 点云: {file_path}")
        # 使用 loadtxt 读取, 默认以空白分隔, 忽略 '#' 注释行
        data = np.loadtxt(file_path, comments="#", dtype=np.float64)

        if data.ndim == 1:
            # 单行数据情况, 扩展为 (1, M)
            data = data[None, :]

        if data.shape[1] < 3:
            raise ValueError(f"XYZ 文件列数小于 3, 无法作为点云使用: {file_path}")

        num_points, num_cols = data.shape
        cls.logger.info(f"XYZ 点云: N={num_points}, 列数={num_cols}")

        points = data[:, 0:3].astype(np.float64, copy=False)

        normals: Optional[np.ndarray] = None
        colors: Optional[np.ndarray] = None
        extra_attrs: Dict[str, np.ndarray] = {}

        if num_cols <= 3:
            # 只有坐标, 无额外属性
            return points, None, None, extra_attrs

        # 额外列数据
        extra = data[:, 3:]  # 形状 (N, num_cols-3)
        extra_dim = extra.shape[1]
        used_mask = np.zeros(extra_dim, dtype=bool)

        # 【推测】法向: 若额外列数 >= 3, 且 extra[:,0:3] 在 [-1.1, 1.1] 内
        if extra_dim >= 3:
            cand_normals = extra[:, 0:3]
            if np.all(np.abs(cand_normals) <= 1.1):
                normals = cand_normals.astype(np.float64, copy=False)
                used_mask[0:3] = True
                cls.logger.info("XYZ 启发式识别: 使用第 4~6 列作为法向 (nx, ny, nz)")

        # 【推测】颜色: 若额外列数 >= 3, 尝试使用最后 3 列作为颜色
        if extra_dim >= 3:
            cand_colors = extra[:, -3:]
            # 检查 [0,255] 范围且接近整数
            if np.all((cand_colors >= 0) & (cand_colors <= 255)):
                int_like = np.all(np.abs(cand_colors - np.round(cand_colors)) <= 1e-3)
                if int_like:
                    colors = cand_colors.astype(np.uint8)
                    used_mask[-3:] = True
                    cls.logger.info("XYZ 启发式识别: 使用最后 3 列作为颜色 (R, G, B)")

        # 未被使用的额外列作为 extra_attrs
        for idx in range(extra_dim):
            if used_mask[idx]:
                continue
            col_index = idx + 3  # 恢复原始列索引(从 0 开始计)
            key = f"attr{col_index}"
            extra_attrs[key] = extra[:, idx]
        if extra_attrs:
            cls.logger.info(f"XYZ 额外属性列: {list(extra_attrs.keys())}")

        return points, normals, colors, extra_attrs
