# -*- coding: utf-8 -*-
"""
DSE xyz-js-c-abcd.txt  ->  RockDiscontinuity ExportPointLevelCsv-compatible CSV

输入(每行9列, 制表符分隔; 允许混合空格):
    X, Y, Z, js, c, A, B, C, D

输出CSV表头(与 ExportPointLevelCsv 一致):
    "X","Y","Z","nx","ny","nz","R","G","B",
    "DR","DG","DB",
    "Discontinuity_id","Cluster_id","Segment_id",
    "A","B","C","D","RMS","Curvature","DistToPlane"

字段赋值规则:
    - X,Y,Z <- 输入
    - nx,ny,nz,R,G,B <- 0
    - Discontinuity_id <- 按 (js,c) 组合重新编号(首次出现顺序)
    - Cluster_id <- js
    - Segment_id <- 0
    - A,B,C,D <- 输入
    - RMS,Curvature,DistToPlane <- 0
    - DR,DG,DB <- _GenerateColorFromId(Discontinuity_id)
"""

import csv
import os
import time
from typing import Dict, Tuple


# =========================
# 用户参数区：按需修改
# =========================
INPUT_TXT_PATH = r"D:\Research\20250313_RockFractureSeg\Code\qfacet_gpu\data\facet_export\Rock_GLS4_part1_localize_0.05m_facets_Kd_E0.2A25\Rock_GLS4_part1_localize_0.05m_facets xyz-js-c-abcd.txt"
OUTPUT_CSV_PATH = r"D:\Research\20250313_RockFractureSeg\Code\qfacet_gpu\data\facet_export\Rock_GLS4_part1_localize_0.05m_facets_Kd_E0.2A25\Rock_GLS4_part1_localize_0.05m_points_fromQFacet.csv"

# INPUT_TXT_PATH = r"D:\Research\20250313_RockFractureSeg\Experiments\_DSE_Rock_GLS4_part1_localize_0.05m\Rock_GLS4_part1_localize_0.05m xyz-js-c-abcd.txt"
# OUTPUT_CSV_PATH = r"D:\Research\20250313_RockFractureSeg\Experiments\_DSE_Rock_GLS4_part1_localize_0.05m\Rock_GLS4_part1_localize_0.05m_points_fromDSE.csv"

# 进度打印间隔（行）
PRINT_EVERY_N = 2_000_000

# 是否跳过空行/坏行
SKIP_BAD_LINES = True


def GenerateColorFromId(idx: int) -> Tuple[int, int, int]:
    """
    功能简介:
        复刻 ResultsExporter._GenerateColorFromId 的颜色生成逻辑。

    输入:
        idx (int): Discontinuity_id

    输出:
        (dr,dg,db) (Tuple[int,int,int]): [0,255] 颜色
    """
    r = (37 * idx + 59) % 256
    g = (17 * idx + 97) % 256
    b = (73 * idx + 23) % 256
    return int(r), int(g), int(b)


def _ParseIntLike(x: str) -> int:
    """
    将 '1' / '1.0' / '  1  ' 等解析为 int。
    """
    return int(float(x.strip()))


def ConvertDseTxtToPointsCsv(
    input_txt_path: str,
    output_csv_path: str,
) -> None:
    """
    功能简介:
        将 DSE 导出的 xyz-js-c-abcd.txt 转为点级 CSV（对齐 ExportPointLevelCsv）。

    实现思路:
        - 逐行读取，按 split() 解析（可兼容 tab/多空格）。
        - 为 (js,c) 建立映射 -> discontinuity_id（首次出现顺序）。
        - 写 CSV 表头 + 行记录。

    输入:
        input_txt_path (str): DSE txt 路径
        output_csv_path (str): 输出 CSV 路径

    输出:
        None（写文件）
    """
    t0 = time.time()

    if not os.path.isfile(input_txt_path):
        raise FileNotFoundError(f"输入文件不存在: {input_txt_path}")

    os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)

    # (js,c) -> new_discontinuity_id
    disc_map: Dict[Tuple[int, int], int] = {}

    n_in = 0
    n_out = 0
    n_bad = 0

    header = [
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
    ]

    with open(input_txt_path, "r", encoding="utf-8", errors="ignore") as fin, \
         open(output_csv_path, "w", newline="", encoding="utf-8") as fout:

        writer = csv.writer(fout)
        writer.writerow(header)

        for line in fin:
            n_in += 1
            s = line.strip()
            if not s:
                continue

            parts = s.split()  # 兼容 tab / 多空格
            if len(parts) < 9:
                n_bad += 1
                if SKIP_BAD_LINES:
                    continue
                raise ValueError(f"第 {n_in} 行列数不足9: {s}")

            try:
                x = float(parts[0])
                y = float(parts[1])
                z = float(parts[2])

                js = _ParseIntLike(parts[3])
                c = _ParseIntLike(parts[4])

                A = float(parts[5])
                B = float(parts[6])
                C = float(parts[7])
                D = float(parts[8])
            except Exception:
                n_bad += 1
                if SKIP_BAD_LINES:
                    continue
                raise

            key = (js, c)
            if key not in disc_map:
                disc_map[key] = len(disc_map)
            disc_id = disc_map[key]

            dr, dg, db = GenerateColorFromId(disc_id)

            writer.writerow([
                x, y, z,
                0, 0, 0,          # nx,ny,nz
                0, 0, 0,          # R,G,B
                dr, dg, db,       # DR,DG,DB
                disc_id,          # Discontinuity_id
                js,               # Cluster_id
                0,                # Segment_id
                A, B, C, D,       # plane params
                0,                # RMS
                0,                # Curvature
                0,                # DistToPlane
            ])
            n_out += 1

            if PRINT_EVERY_N > 0 and (n_in % PRINT_EVERY_N == 0):
                dt = time.time() - t0
                print(f"[LOG] read={n_in:,}  wrote={n_out:,}  bad={n_bad:,}  discs={len(disc_map):,}  time={dt:.1f}s")

    dt = time.time() - t0
    print(f"[DONE] input_lines={n_in:,} output_rows={n_out:,} bad_lines={n_bad:,} "
          f"unique_discontinuities={len(disc_map):,} time={dt:.2f}s")
    print(f"[DONE] output_csv: {output_csv_path}")


if __name__ == "__main__":
    ConvertDseTxtToPointsCsv(INPUT_TXT_PATH, OUTPUT_CSV_PATH)
