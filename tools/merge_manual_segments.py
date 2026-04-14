import os
import re
import csv
import time
import math
import hashlib
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
'''
在cloudcompare打开manual segmentation bin文件;

在DB树下选中“文件夹组”Rock_GLS4_part1_localize_0.02m_ManualSegments_v2.bin (D:/Research/20250313_RockFractureSeg/Code/RockDiscontinuity/data/rock_data)，点“保存”，点云格式选择"csv", 参数为:coordinate precision=8, scalar precision=6, separator=comma, order="[ASC] point, color, SF(s), normal", Header-columns title=True, Header-number of points=False, Save colors as float values=False, Save alpha channel=False.

随后cloudcompare会将文件夹组下的每个cloud逐一导出到选择的文件夹下,例如:
D:\Research\20250313_RockFractureSeg\Code\RockDiscontinuity\data\rock_data\Rock_GLS4_part1_localize_0.02m_ManualSegments
"D:\Research\20250313_RockFractureSeg\Code\RockDiscontinuity\data\rock_data\Rock_GLS4_part1_localize_0.02m_ManualSegments\Rock_GLS4_part1_localize_0.02m_discontinuity_000003.csv"
"D:\Research\20250313_RockFractureSeg\Code\RockDiscontinuity\data\rock_data\Rock_GLS4_part1_localize_0.02m_ManualSegments\Rock_GLS4_part1_localize_0.02m_discontinuity_000000.csv"
"D:\Research\20250313_RockFractureSeg\Code\RockDiscontinuity\data\rock_data\Rock_GLS4_part1_localize_0.02m_ManualSegments\Rock_GLS4_part1_localize_0.02m_discontinuity_000001.csv"
"D:\Research\20250313_RockFractureSeg\Code\RockDiscontinuity\data\rock_data\Rock_GLS4_part1_localize_0.02m_ManualSegments\Rock_GLS4_part1_localize_0.02m_discontinuity_000002.csv"
...

这些文件的表头有两类,第一类例如:
//X,Y,Z,R,G,B,Nx,Ny,Nz
94.52654266,39.05059052,75.29547119,30,31,33,0.212465,0.967036,0.140355
94.38317871,39.01905060,75.39311218,35,35,36,0.390562,0.814600,0.428821
94.47760010,39.06729126,75.26600647,34,36,35,0.155203,0.981323,0.113653
...
第二类例如:
//X,Y,Z,R,G,B,Dip (degrees),Dip direction (degrees),Nx,Ny,Nz
85.70324707,19.65340805,68.85858154,122,129,129,87.488403,322.011230,0.614915,-0.787374,-0.043822
86.03137207,19.84701157,68.90746307,57,60,64,78.392334,154.702850,0.418574,-0.885613,0.201209
85.88317871,19.75671577,68.82491302,101,106,107,84.397881,142.375137,0.607573,-0.788242,0.097620
...

现在需要写一个python脚本将这些csv文件合并成一个csv文件(输出1)和一个ply文件(输出2),要求:
# 逐文件、逐行顺序读取 → 点顺序保持一致（文件内顺序不变；文件间按 discontinuity_XXXXXX 数字升序拼接)
# 兼容两类 CloudCompare CSV 表头（含/不含 Dip/Dipdir）
# 输出的 CSV 列顺序严格按"X", "Y", "Z", "nx", "ny", "nz", "R", "G", "B", "DR", "DG", "DB", "Discontinuity_id", "Cluster_id", "Segment_id", "A", "B", "C", "D", "RMS", "Curvature", "DistToPlane"; 其中"X", "Y", "Z", "nx", "ny", "nz", "R", "G", "B"的值直接来源于csv文件序列, DR/DG/DB 按 Discontinuity_id 确定性随机生成（同一 id 永远同色，便于复现）, Discontinuity_id来自csv文件名的后缀(例如Rock_GLS4_part1_localize_0.02m_discontinuity_000003对应的Discontinuity_id=3) 但是最后一个Discontinuity_id,例如Rock_GLS4_part1_localize_0.02m_discontinuity_000313.csv, 它的Discontinuity_id要强制设为-1,以表示分割后剩余的非结构面点.其余部分"Cluster_id", "Segment_id", "A", "B", "C", "D", "RMS", "Curvature", "DistToPlane"用0占位.
# 输出 PLY 的字段仅包含 x,y,z,nx,ny,nz,red,green,blue，能被如下读取函数识别,并且确保读取后的points顺序与输出的CSV的行顺序保持严格一致:

    @classmethod
    def _ReadPlyGeneric(
            cls,
            file_path: str
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Dict[str, np.ndarray]]:
                cls.logger.info(f"读取 PLY 点云: {file_path}")
        ply = PlyData.read(file_path)

        if "vertex" not in ply:
            raise ValueError("PLY 文件中缺少 'vertex' 元素。")

        v = ply["vertex"].data  # structured array
        names = v.dtype.names
        if names is None:
            raise ValueError("PLY 'vertex' 元素字段信息为空。")
        cls.logger.info(f"PLY中的name信息: {names}")

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
'''

@dataclass(frozen=True)
class FileItem:
    """文件条目数据结构"""
    file_path: str
    file_name: str
    disc_num: int


class CloudCompareCsvMerger:
    logger = logging.getLogger("CloudCompareCsvMerger")

    @staticmethod
    def SetupLogger(log_path: str) -> None:
        """
        功能简介:
            配置日志系统：控制台 + 文件双输出。
        实现思路:
            1) 创建 logger，设置 INFO 级别
            2) 绑定 FileHandler 与 StreamHandler
        输入:
            log_path (str): 日志输出文件路径
        输出:
            None
        """
        CloudCompareCsvMerger.logger.setLevel(logging.INFO)
        CloudCompareCsvMerger.logger.handlers.clear()

        fmt = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # 文件日志
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)

        # 控制台日志
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(fmt)

        CloudCompareCsvMerger.logger.addHandler(fh)
        CloudCompareCsvMerger.logger.addHandler(sh)

    @staticmethod
    def ListDiscontinuityCsvFiles(input_dir: str) -> List[FileItem]:
        """
        功能简介:
            扫描目录，找到所有符合 *_discontinuity_XXXXXX.csv 的文件并解析编号。
        实现思路:
            1) os.listdir 遍历
            2) 正则提取 discontinuity_ 后的数字
            3) 组装 FileItem 并按 disc_num 升序排序
        输入:
            input_dir (str): 输入目录
        输出:
            files (List[FileItem]): 排序后的文件列表
        """
        pattern = re.compile(r"discontinuity_(\d+)\.csv$", re.IGNORECASE)
        items: List[FileItem] = []

        for fn in os.listdir(input_dir):
            fp = os.path.join(input_dir, fn)
            if not os.path.isfile(fp):
                continue
            m = pattern.search(fn)
            if not m:
                continue
            disc_num = int(m.group(1))
            items.append(FileItem(file_path=fp, file_name=fn, disc_num=disc_num))

        items.sort(key=lambda x: x.disc_num)
        return items

    @staticmethod
    def DeterministicColorById(discontinuity_id: int) -> Tuple[int, int, int]:
        """
        功能简介:
            根据 Discontinuity_id 生成确定性“随机”颜色 (0-255)。
        实现思路:
            - 若 id == -1：按约定返回黑色 (0,0,0)
            - 否则：对 id 字符串做 md5，取 digest 前 3 字节作为 RGB
        输入:
            discontinuity_id (int): 结构面 id（可为 -1）
        输出:
            (dr, dg, db) (Tuple[int,int,int]): 0~255 颜色
        """
        if discontinuity_id == -1:
            return 0, 0, 0

        md5 = hashlib.md5(str(discontinuity_id).encode("utf-8")).digest()
        dr, dg, db = md5[0], md5[1], md5[2]
        return int(dr), int(dg), int(db)


    @staticmethod
    def ParseHeaderAndBuildIndex(header_line: str) -> Dict[str, int]:
        """
        功能简介:
            解析 CloudCompare CSV 的表头，并建立列名到索引的映射。
        实现思路:
            1) 去掉开头的 //（若存在）
            2) 按逗号分割得到列名
            3) 允许大小写差异（Nx/nx 等）
        输入:
            header_line (str): CSV 第一行（表头）
        输出:
            col_index (Dict[str,int]): 关键字段到索引的映射
        """
        line = header_line.strip()
        if line.startswith("//"):
            line = line[2:].strip()

        cols = [c.strip() for c in line.split(",") if c.strip() != ""]
        col_lut: Dict[str, int] = {}
        for i, c in enumerate(cols):
            col_lut[c] = i
            col_lut[c.lower()] = i  # 兼容大小写

        # 规范字段键：统一用小写 key 来取
        required_any = [
            "x", "y", "z",
            "r", "g", "b",
            "nx", "ny", "nz"
        ]
        missing = [k for k in required_any if k not in col_lut]
        if missing:
            raise ValueError(f"CSV 表头缺少必要字段: {missing} | header={header_line.strip()}")

        return col_lut

    @staticmethod
    def CountPointsInCsv(file_path: str) -> int:
        """
        功能简介:
            统计一个 CSV 文件中有效点行数量（跳过表头/空行）。
        实现思路:
            1) 读取首行表头
            2) 遍历后续行，过滤空行与列数不足行
        输入:
            file_path (str): CSV 文件路径
        输出:
            n_points (int): 点数量
        """
        n_points = 0
        with open(file_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                return 0
            for row in reader:
                if not row:
                    continue
                # 过滤空白行（例如全空字符串）
                if all((c.strip() == "" for c in row)):
                    continue
                n_points += 1
        return n_points

    @staticmethod
    def WritePlyAsciiHeader(
        ply_f,
        vertex_count: int
    ) -> None:
        """
        功能简介:
            写 ASCII PLY 头，字段仅包含 x,y,z,nx,ny,nz,red,green,blue。
        实现思路:
            按 plyfile 可识别的标准 PLY header 输出。
        输入:
            ply_f: 已打开的文件句柄
            vertex_count (int): 顶点数量
        输出:
            None
        """
        ply_f.write("ply\n")
        ply_f.write("format ascii 1.0\n")
        ply_f.write(f"element vertex {vertex_count}\n")
        ply_f.write("property float x\n")
        ply_f.write("property float y\n")
        ply_f.write("property float z\n")
        ply_f.write("property float nx\n")
        ply_f.write("property float ny\n")
        ply_f.write("property float nz\n")
        ply_f.write("property uchar red\n")
        ply_f.write("property uchar green\n")
        ply_f.write("property uchar blue\n")
        ply_f.write("end_header\n")

    @staticmethod
    def Merge(
        input_dir: str,
        output_csv_path: str,
        output_ply_path: str,
        enable_viz: bool = True
    ) -> None:
        """
        功能简介:
            合并多个 CloudCompare 导出的 discontinuity_XXXXXX.csv 为一个 CSV + 一个 PLY，
            并保证输出点顺序严格一致。
        实现思路(详细):
            1) 扫描并按编号排序输入文件
            2) 将排序最后一个文件的 Discontinuity_id 强制设为 -1
            3) 第一遍统计总点数，用于写 PLY header
            4) 第二遍逐文件逐行写出 CSV 与 PLY（同一循环内写，保证顺序一致）
            5) 输出点数统计图（可选）
        输入:
            input_dir (str): 输入目录（多个 discontinuity_*.csv）
            output_csv_path (str): 输出合并 CSV 路径
            output_ply_path (str): 输出合并 PLY 路径
            enable_viz (bool): 是否输出统计图
        输出:
            None
        """
        t0 = time.perf_counter()

        # 1) 列出并排序文件
        files = CloudCompareCsvMerger.ListDiscontinuityCsvFiles(input_dir)
        if not files:
            raise FileNotFoundError(f"在目录中未找到 discontinuity_*.csv: {input_dir}")

        CloudCompareCsvMerger.logger.info(f"发现 {len(files)} 个 discontinuity CSV 文件。")

        # 2) 设定最后一个文件 id=-1
        last_file_path = files[-1].file_path
        CloudCompareCsvMerger.logger.info(
            f"按规则：最后一个文件将强制 Discontinuity_id=-1: {os.path.basename(last_file_path)}"
        )

        # 3) 第一遍：统计总点数 + 每文件点数
        total_points = 0
        counts_by_id: Dict[int, int] = {}
        t_count0 = time.perf_counter()

        for i, item in enumerate(files):
            file_t0 = time.perf_counter()
            n_pts = CloudCompareCsvMerger.CountPointsInCsv(item.file_path)

            disc_id = -1 if item.file_path == last_file_path else item.disc_num
            total_points += n_pts
            counts_by_id[disc_id] = counts_by_id.get(disc_id, 0) + n_pts

            file_dt = time.perf_counter() - file_t0
            CloudCompareCsvMerger.logger.info(
                f"[计数] {i+1}/{len(files)} {item.file_name} | disc_id={disc_id} | 点数={n_pts} | 耗时={file_dt:.3f}s"
            )

        t_count1 = time.perf_counter()
        CloudCompareCsvMerger.logger.info(
            f"总点数={total_points} | 计数阶段耗时={(t_count1 - t_count0):.3f}s"
        )

        # 4) 第二遍：写出 CSV + PLY
        out_dir = os.path.dirname(output_csv_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        csv_header_out = [
            "X", "Y", "Z",
            "nx", "ny", "nz",
            "R", "G", "B",
            "DR", "DG", "DB",
            "Discontinuity_id",
            "Cluster_id", "Segment_id",
            "A", "B", "C", "D",
            "RMS", "Curvature", "DistToPlane"
        ]

        t_write0 = time.perf_counter()

        with open(output_csv_path, "w", encoding="utf-8", newline="") as csv_f, \
             open(output_ply_path, "w", encoding="utf-8", newline="\n") as ply_f:

            # 写 CSV 表头
            csv_writer = csv.writer(csv_f)
            csv_writer.writerow(csv_header_out)

            # 写 PLY header
            CloudCompareCsvMerger.WritePlyAsciiHeader(ply_f, total_points)

            written = 0

            for i, item in enumerate(files):
                file_t0 = time.perf_counter()

                disc_id = -1 if item.file_path == last_file_path else item.disc_num
                dr, dg, db = CloudCompareCsvMerger.DeterministicColorById(disc_id)

                # 逐行读取
                with open(item.file_path, "r", encoding="utf-8-sig", newline="") as f:
                    reader = csv.reader(f)
                    header_row = next(reader, None)
                    if header_row is None:
                        continue

                    # CloudCompare header 往往是单列字符串（含逗号），也可能被 csv.reader 拆分
                    # 这里统一回拼为一行再解析
                    header_line = ",".join(header_row)
                    col_index = CloudCompareCsvMerger.ParseHeaderAndBuildIndex(header_line)

                    for row in reader:
                        if not row:
                            continue
                        if all((c.strip() == "" for c in row)):
                            continue

                        # 兼容：若某些行被拆分异常（一般不会），尝试修正
                        # 这里假设标准逗号分隔，row 已是列数组；不足列则跳过
                        max_needed = max(
                            col_index["x"], col_index["y"], col_index["z"],
                            col_index["r"], col_index["g"], col_index["b"],
                            col_index["nx"], col_index["ny"], col_index["nz"]
                        )
                        if len(row) <= max_needed:
                            continue

                        # 取值（坐标/法向/颜色来自原 CSV）
                        x = float(row[col_index["x"]])
                        y = float(row[col_index["y"]])
                        z = float(row[col_index["z"]])

                        nx = float(row[col_index["nx"]])
                        ny = float(row[col_index["ny"]])
                        nz = float(row[col_index["nz"]])

                        r = int(float(row[col_index["r"]]))
                        g = int(float(row[col_index["g"]]))
                        b = int(float(row[col_index["b"]]))

                        # 占位字段
                        cluster_id = 0
                        segment_id = 0
                        A = Bc = C = D = 0
                        rms = 0
                        curvature = 0
                        dist_to_plane = 0

                        # 写 CSV（严格列顺序）
                        csv_writer.writerow([
                            f"{x:.8f}", f"{y:.8f}", f"{z:.8f}",
                            f"{nx:.6f}", f"{ny:.6f}", f"{nz:.6f}",
                            r, g, b,
                            dr, dg, db,
                            disc_id,
                            cluster_id, segment_id,
                            A, Bc, C, D,
                            rms, curvature, dist_to_plane
                        ])

                        # 写 PLY（保证顺序一致：同一循环内写）
                        ply_f.write(
                            f"{x:.8f} {y:.8f} {z:.8f} "
                            f"{nx:.6f} {ny:.6f} {nz:.6f} "
                            f"{r:d} {g:d} {b:d}\n"
                        )

                        written += 1

                file_dt = time.perf_counter() - file_t0
                CloudCompareCsvMerger.logger.info(
                    f"[写出] {i+1}/{len(files)} {item.file_name} | disc_id={disc_id} | DRGB=({dr},{dg},{db}) | 累计写出={written} | 耗时={file_dt:.3f}s"
                )

        t_write1 = time.perf_counter()
        CloudCompareCsvMerger.logger.info(
            f"写出完成：CSV={output_csv_path} | PLY={output_ply_path} | 写出点数={written} | 写出阶段耗时={(t_write1 - t_write0):.3f}s"
        )

        if written != total_points:
            CloudCompareCsvMerger.logger.warning(
                f"写出点数({written}) != 统计点数({total_points})。若输入 CSV 存在异常行/列数不足行，可能被跳过。"
            )

        # 5) 简单可视化：每个 Discontinuity_id 点数统计
        if enable_viz:
            viz_t0 = time.perf_counter()
            fig_path = os.path.splitext(output_csv_path)[0] + "_counts.png"

            # x 轴按 id 升序显示（-1 放最后更直观，可按你偏好调整）
            ids = sorted([k for k in counts_by_id.keys() if k != -1]) + ([-1] if -1 in counts_by_id else [])
            vals = [counts_by_id[k] for k in ids]

            plt.figure()
            plt.bar([str(k) for k in ids], vals)
            plt.xticks(rotation=90)
            plt.xlabel("Discontinuity_id")
            plt.ylabel("Number of points")
            plt.title("Point counts per discontinuity id")
            plt.tight_layout()
            plt.savefig(fig_path, dpi=200)
            plt.close()

            viz_dt = time.perf_counter() - viz_t0
            CloudCompareCsvMerger.logger.info(f"统计图已输出: {fig_path} | 耗时={viz_dt:.3f}s")

        dt = time.perf_counter() - t0
        CloudCompareCsvMerger.logger.info(f"全部完成，总耗时={dt:.3f}s")


def Main() -> None:
    """
    功能简介:
        脚本入口：配置输入输出路径并执行合并。
    实现思路:
        1) 设置路径
        2) 初始化日志
        3) 调用 Merge
    输入:
        None（在函数内直接修改路径变量）
    输出:
        None（生成 CSV/PLY/日志/统计图）
    """
    # ====== 你只需要改这里 ======
    input_dir = r"D:\Research\20250313_RockFractureSeg\Code\RockDiscontinuity\data\rock_data\Rock_GLS4_part1_localize_0.02m_ManualSegments"

    output_csv = r"D:\Research\20250313_RockFractureSeg\Code\RockDiscontinuity\data\rock_data\Rock_GLS4_part1_localize_0.02m_ManualSegments_merged.csv"
    output_ply = r"D:\Research\20250313_RockFractureSeg\Code\RockDiscontinuity\data\rock_data\Rock_GLS4_part1_localize_0.02m_ManualSegments_merged.ply"
    # ==========================

    log_path = os.path.splitext(output_csv)[0] + ".log"
    CloudCompareCsvMerger.SetupLogger(log_path)

    CloudCompareCsvMerger.Merge(
        input_dir=input_dir,
        output_csv_path=output_csv,
        output_ply_path=output_ply,
        enable_viz=True
    )


if __name__ == "__main__":
    Main()
