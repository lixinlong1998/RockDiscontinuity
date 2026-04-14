"""
PTv3 Rock Mass Point Cloud Preprocessor  v2.0
================================================
功能:
  1. 3D Voxel 切块 (默认 5m×5m×5m, 10% 重叠率)
  2. 数据集划分 train/val/test (默认 0.8/0.2/0.0)
  3. 有效性检验 (点数 >= min_points)
  4. 数据集报表 (点数/内存/结构面比例/实例数)
  5. PLY 可视化文件输出 (单独文件夹)
  6. Pointcept 兼容的 .pth 格式存储

输出目录结构:
  <output_dir>/
  ├── train/    *.pth   (Pointcept DefaultDataset 格式)
  ├── val/      *.pth
  ├── test/     *.pth   (split_ratio[2]==0 时为空)
  ├── ply/      *.ply   (可视化用, 含结构面彩色)
  └── report.csv

用法:
  python preprocess.py --input rock.csv --output data/rock_fracture
  python preprocess.py --input rock.csv --output data/rock_fracture \
      --tile_size 5.0 --overlap 0.1 --max_points 120000

依赖:
  pip install numpy pandas scikit-learn plyfile tqdm
"""

import os
import sys
import argparse
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

try:
    from plyfile import PlyData, PlyElement

    HAS_PLY = True
except ImportError:
    HAS_PLY = False
    print("[警告] plyfile 未安装，跳过 PLY 导出。安装: pip install plyfile")

# ─────────────────────────────────────────────────────────────
#  默认配置（也可通过命令行覆盖）
# ─────────────────────────────────────────────────────────────
SIZE = 10.0  # 切块边长 (m)
DEFAULTS = dict(
    input=r"D:\Research\20250313_RockFractureSeg\Code\RockDiscontinuity\data\rock_data\Rock_GLS4_part1_localize_0.02m_ManualSegments_merged_label.csv",
    output=r"D:\Research\20250313_RockFractureSeg\Code\DockerVolume\PointTransformerV3\Workspace1\PointTransformerV3\Pointcept\data\rock_fracture10",
    tile_x=SIZE,  # 切块 X 边长 (m)
    tile_y=SIZE,  # 切块 Y 边长 (m)
    tile_z=SIZE,  # 切块 Z 边长 (m)
    overlap=0.10,  # 重叠率 0~1
    min_points=6000,  # 有效性阈值
    max_points=120000,  # 超出则随机降采样
    split_ratios=(0.8, 0.2, 0.0),  # train/val/test
    seed=42,
    grid_size=0.02,  # 仅用于报表估算体素数，与 config 保持一致
    save_ply=True,
)

# CSV 列名（如有差异请修改）
COL_XYZ = ["X", "Y", "Z"]
COL_NRM = ["nx", "ny", "nz"]
COL_RGB = ["R", "G", "B"]
COL_LABEL = "Discontinuity_id"  # -1=背景, 0~N=结构面实例


# ─────────────────────────────────────────────────────────────
#  工具函数
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="PTv3 Rock Pointcloud Preprocessor")
    p.add_argument("--input", default=DEFAULTS["input"])
    p.add_argument("--output", default=DEFAULTS["output"])
    p.add_argument("--tile_x", type=float, default=DEFAULTS["tile_x"])
    p.add_argument("--tile_y", type=float, default=DEFAULTS["tile_y"])
    p.add_argument("--tile_z", type=float, default=DEFAULTS["tile_z"])
    p.add_argument("--overlap", type=float, default=DEFAULTS["overlap"])
    p.add_argument("--min_points", type=int, default=DEFAULTS["min_points"])
    p.add_argument("--max_points", type=int, default=DEFAULTS["max_points"])
    p.add_argument("--split_ratios", type=float, nargs=3,
                   default=list(DEFAULTS["split_ratios"]),
                   metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    # BooleanOptionalAction: --save_ply 开启(默认), --no-save-ply 关闭
    # 用法示例: python preprocess.py --no-save-ply
    p.add_argument("--save_ply",
                   action=argparse.BooleanOptionalAction,
                   default=DEFAULTS["save_ply"],
                   help="输出 PLY 可视化文件 (default: True)，关闭用 --no-save-ply")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def load_csv(path: str) -> pd.DataFrame:
    print(f"\n{'=' * 60}")
    print(f"[Step 1/5] 读取 CSV: {path}")
    t0 = time.time()
    df = pd.read_csv(path)
    elapsed = time.time() - t0
    print(f"  总点数   : {len(df):>12,}")
    print(f"  列名     : {list(df.columns)}")
    print(f"  读取耗时 : {elapsed:.1f}s")

    # 验证必需列
    required = COL_XYZ + COL_NRM + COL_RGB + [COL_LABEL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(f"[错误] CSV 缺少必需列: {missing}")

    # 数据范围
    xyz = df[COL_XYZ].values
    print(f"  X 范围   : [{xyz[:, 0].min():.2f}, {xyz[:, 0].max():.2f}]")
    print(f"  Y 范围   : [{xyz[:, 1].min():.2f}, {xyz[:, 1].max():.2f}]")
    print(f"  Z 范围   : [{xyz[:, 2].min():.2f}, {xyz[:, 2].max():.2f}]")

    ids = df[COL_LABEL].unique()
    n_inst = (ids >= 0).sum()
    n_bg = (df[COL_LABEL] < 0).sum()
    print(f"  结构面实例数: {n_inst}  背景点: {n_bg:,}  ({100 * n_bg / len(df):.1f}%)")
    return df


def compute_tile_grid(xyz: np.ndarray, tile_size: tuple, overlap: float):
    """生成 3D 切块的起始坐标列表"""
    tx, ty, tz = tile_size
    step_x = tx * (1 - overlap)
    step_y = ty * (1 - overlap)
    step_z = tz * (1 - overlap)

    x0_list = np.arange(xyz[:, 0].min(), xyz[:, 0].max(), step_x)
    y0_list = np.arange(xyz[:, 1].min(), xyz[:, 1].max(), step_y)
    z0_list = np.arange(xyz[:, 2].min(), xyz[:, 2].max(), step_z)

    tiles = []
    for x0 in x0_list:
        for y0 in y0_list:
            for z0 in z0_list:
                tiles.append((x0, y0, z0))
    return tiles, (x0_list, y0_list, z0_list)


def tile_pointcloud(df: pd.DataFrame, tile_size: tuple, overlap: float,
                    min_points: int, max_points: int):
    """
    3D 切块，返回有效块列表
    每个元素: dict(indices, origin_xyz, tile_id)
    """
    print(f"\n[Step 2/5] 3D 切块")
    print(f"  切块尺寸  : {tile_size[0]}m × {tile_size[1]}m × {tile_size[2]}m")
    print(f"  重叠率    : {overlap * 100:.0f}%")

    xyz = df[COL_XYZ].values.astype(np.float32)
    tiles, (xs, ys, zs) = compute_tile_grid(xyz, tile_size, overlap)

    tx, ty, tz = tile_size
    total_tiles = len(tiles)
    valid_blocks = []
    skipped_few = 0

    for tid, (x0, y0, z0) in enumerate(tqdm(tiles, desc="  切块中", ncols=70)):
        mask = (
                (xyz[:, 0] >= x0) & (xyz[:, 0] < x0 + tx) &
                (xyz[:, 1] >= y0) & (xyz[:, 1] < y0 + ty) &
                (xyz[:, 2] >= z0) & (xyz[:, 2] < z0 + tz)
        )
        indices = np.where(mask)[0]

        if len(indices) < min_points:
            skipped_few += 1
            continue

        # 超出上限 → 分层采样（保证结构面点被采到）
        if len(indices) > max_points:
            disc = indices[df[COL_LABEL].values[indices] >= 0]
            bg = indices[df[COL_LABEL].values[indices] < 0]
            # 按比例分配采样配额
            n_disc = min(len(disc), int(max_points * len(disc) / len(indices)) + 1)
            n_bg = max_points - n_disc
            sampled = []
            if len(disc) > 0:
                sampled.append(np.random.choice(disc, min(n_disc, len(disc)), replace=False))
            if len(bg) > 0:
                sampled.append(np.random.choice(bg, min(n_bg, len(bg)), replace=False))
            indices = np.concatenate(sampled) if sampled else indices[:max_points]

        valid_blocks.append(dict(
            indices=indices,
            origin_xyz=(x0, y0, z0),
            tile_id=tid,
        ))

    print(f"  总切块数  : {total_tiles}")
    print(f"  有效块数  : {len(valid_blocks)}")
    print(f"  丢弃(点<{min_points}): {skipped_few}")
    return valid_blocks


def build_sample(df: pd.DataFrame, indices: np.ndarray, normalize_xyz=True):
    """
    构建 Pointcept 兼容的数据字典
    coord  : float32 (N,3)  块内中心化 XYZ
    feat   : float32 (N,6)  [nx,ny,nz, R/255, G/255, B/255]
    segment: int64   (N,)   0=背景 1=结构面  (语义标签)
    instance:int64   (N,)   -1=背景, 0..N=实例id (实例标签)
    """
    sub = df.iloc[indices]

    coord = sub[COL_XYZ].values.astype(np.float32)
    if normalize_xyz:
        coord -= coord.mean(axis=0)

    # 法向量归一化
    normals = sub[COL_NRM].values.astype(np.float32)
    nlen = np.linalg.norm(normals, axis=1, keepdims=True).clip(min=1e-6)
    normals /= nlen

    # RGB 归一化到 [0,1]
    rgb = sub[COL_RGB].values.astype(np.float32) / 255.0

    feat = np.concatenate([normals, rgb], axis=1)  # (N,6)

    disc_id = sub[COL_LABEL].values.astype(np.int64)
    segment = (disc_id >= 0).astype(np.int64)  # 0/1 语义
    instance = disc_id.copy()  # -1/0..N 实例

    return dict(coord=coord, feat=feat, segment=segment, instance=instance)


def split_blocks(blocks, ratios, seed):
    """
    将 blocks 随机打乱后按比例划分。

    修复策略（彻底消除 int/round 截断误差）：
      - n_train = floor(n * ratios[0])
      - n_test  = floor(n * ratios[2])，若 ratios[2]==0.0 则强制为 0
      - n_val   = n - n_train - n_test  （差值法，保证无遗漏、无重复）
      - 若 ratios[1]==0.0 则 n_val 强制为 0，剩余全归 test

    示例: n=5, ratios=(0.8, 0.2, 0.0)
      n_train=4, n_test=0(强制), n_val=5-4-0=1  → test 严格为空 ✓
    示例: n=7, ratios=(0.7, 0.15, 0.15)
      n_train=4, n_test=1, n_val=7-4-1=2  → 无遗漏 ✓
    """
    random.seed(seed)
    idx = list(range(len(blocks)))
    random.shuffle(idx)
    n = len(idx)

    # 使用 math.floor 保证不超分配
    import math
    n_train = math.floor(n * ratios[0])
    n_test = 0 if ratios[2] == 0.0 else math.floor(n * ratios[2])
    n_val = 0 if ratios[1] == 0.0 else (n - n_train - n_test)

    # 防御：val 或 test 被算成负数时归零
    n_val = max(0, n_val)
    n_test = max(0, n - n_train - n_val)  # test 取剩余，ratio==0 时上面已强制 n_test=0

    return (
        idx[:n_train],
        idx[n_train: n_train + n_val],
        idx[n_train + n_val: n_train + n_val + n_test],
    )


# ─────────────────────────────────────────────────────────────
#  PLY 导出
# ─────────────────────────────────────────────────────────────

# 为每个 Discontinuity_id 生成固定随机颜色（复用原始 DR/DG/DB 逻辑）
def _instance_colormap(max_id=2000, seed=0):
    rng = np.random.default_rng(seed)
    colors = rng.integers(30, 230, size=(max_id + 1, 3), dtype=np.uint8)
    return colors


_COLORMAP = _instance_colormap()


def save_ply(df: pd.DataFrame, indices: np.ndarray, ply_path: str):
    """保存带实例彩色的 PLY 文件"""
    if not HAS_PLY:
        return
    sub = df.iloc[indices]
    xyz = sub[COL_XYZ].values.astype(np.float32)
    rgb_raw = sub[COL_RGB].values.astype(np.uint8)
    disc_id = sub[COL_LABEL].values.astype(np.int64)

    # 实例彩色：背景用原始 RGB，结构面用随机色
    r_out = rgb_raw[:, 0].copy()
    g_out = rgb_raw[:, 1].copy()
    b_out = rgb_raw[:, 2].copy()
    for i, did in enumerate(disc_id):
        if did >= 0:
            cid = did % len(_COLORMAP)
            r_out[i] = _COLORMAP[cid, 0]
            g_out[i] = _COLORMAP[cid, 1]
            b_out[i] = _COLORMAP[cid, 2]

    vert = np.array(
        list(zip(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                 r_out, g_out, b_out,
                 disc_id.astype(np.int32))),
        dtype=[
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
            ("discontinuity_id", "i4"),
        ]
    )
    PlyData([PlyElement.describe(vert, "vertex")], text=False).write(ply_path)


# ─────────────────────────────────────────────────────────────
#  报表 + 直方图
# ─────────────────────────────────────────────────────────────

def generate_report(records: list, out_path: str):
    df_rep = pd.DataFrame(records)
    df_rep.to_csv(out_path, index=False, float_format="%.4f")
    print(f"\n  报表已保存: {out_path}")

    # 控制台摘要
    for split in ["train", "val", "test"]:
        sub = df_rep[df_rep["split"] == split]
        if len(sub) == 0:
            continue
        print(f"\n  [{split}]  块数={len(sub)}")
        print(f"    点数    : 均值={sub['n_points'].mean():.0f}  "
              f"最小={sub['n_points'].min()}  最大={sub['n_points'].max()}")
        print(f"    结构面% : 均值={sub['frac_ratio_%'].mean():.1f}%  "
              f"最大={sub['frac_ratio_%'].max():.1f}%")
        print(f"    实例数  : 均值={sub['n_instances'].mean():.1f}  "
              f"最大={sub['n_instances'].max()}")
        print(f"    内存MB  : 均值={sub['mem_MB'].mean():.2f}  "
              f"总计={sub['mem_MB'].sum():.1f} MB")

    return df_rep  # 供直方图函数使用


def generate_report_dashboard(df_rep: pd.DataFrame, out_path: str):
    """
    基于 report.csv 生成完整的可视化仪表板，保存为 PNG。

    布局 (3行 × 3列，共 7 个有效子图):
    ┌──────────────────┬─────────────────┬─────────────────┐
    │  [0,0] 数据集    │ [0,1] 点数分布  │ [0,2] 结构面%  │
    │  Split 分布      │   Histogram     │   Histogram     │
    │  (横向堆叠柱状图) │                 │                 │
    ├──────────────────┼─────────────────┼─────────────────┤
    │ [1,0] 实例数分布  │ [1,1] 内存分布  │ [1,2] 点数 vs  │
    │   Histogram      │   Histogram     │  结构面% 散点   │
    ├──────────────────┴─────────────────┴─────────────────┤
    │ [2, :] 块点数折线图（按 block_id 顺序，各 split 分段） │
    └─────────────────────────────────────────────────────┘
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("  [警告] matplotlib 未安装，跳过报告图。安装: pip install matplotlib")
        return

    SPLIT_COLORS = {"train": "#4C72B0", "val": "#DD8452", "test": "#55A868"}
    active_splits = [s for s in ["train", "val", "test"]
                     if s in df_rep["split"].values]

    fig = plt.figure(figsize=(16, 13))
    fig.suptitle("PTv3 Rock Dataset Preprocessing Report",
                 fontsize=15, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(3, 3, figure=fig,
                           hspace=0.45, wspace=0.35,
                           left=0.07, right=0.97,
                           top=0.93, bottom=0.06)

    # ── 辅助函数 ────────────────────────────────────────────
    def add_stat_box(ax, col):
        lines = []
        for sp in active_splits:
            sub = df_rep[df_rep["split"] == sp][col]
            if len(sub):
                lines.append(f"{sp}: μ={sub.mean():.1f} n={len(sub)}")
        ax.text(0.97, 0.97, "\n".join(lines),
                transform=ax.transAxes, fontsize=7,
                va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          alpha=0.8, ec="#cccccc"))

    def styled_hist(ax, col, title, xlabel, bins=20):
        all_vals = df_rep[col].dropna().values
        if len(all_vals) == 0:
            return
        vmin, vmax = all_vals.min(), all_vals.max()
        if vmax == vmin:
            vmax = vmin + 1
        edges = np.linspace(vmin, vmax, bins + 1)
        for sp in active_splits:
            sub = df_rep[df_rep["split"] == sp][col].dropna().values
            if len(sub):
                ax.hist(sub, bins=edges, alpha=0.55,
                        color=SPLIT_COLORS[sp], edgecolor="white",
                        linewidth=0.4, label=sp)
                ax.axvline(sub.mean(), color=SPLIT_COLORS[sp],
                           linestyle="--", linewidth=1.4, alpha=0.85)
        ax.set_title(title, fontsize=10, pad=5)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("Block Count", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", alpha=0.25, linestyle=":")
        add_stat_box(ax, col)

    # ── [0,0] Split 分布横向堆叠柱状图 ───────────────────────
    ax00 = fig.add_subplot(gs[0, 0])
    split_counts = {sp: int((df_rep["split"] == sp).sum()) for sp in active_splits}
    split_frac_pt = {sp: df_rep[df_rep["split"] == sp]["frac_ratio_%"].mean()
                     for sp in active_splits}
    bar_h = 0.35
    y_pos = np.arange(len(active_splits))
    bars = ax00.barh(y_pos, [split_counts[s] for s in active_splits],
                     height=bar_h, color=[SPLIT_COLORS[s] for s in active_splits],
                     edgecolor="white", linewidth=0.6)
    ax00.set_yticks(y_pos)
    ax00.set_yticklabels(active_splits, fontsize=9)
    ax00.set_xlabel("Block Count", fontsize=8)
    ax00.set_title("Dataset Split Distribution", fontsize=10, pad=5)
    ax00.tick_params(labelsize=7)
    ax00.grid(axis="x", alpha=0.25, linestyle=":")
    for bar, sp in zip(bars, active_splits):
        n = split_counts[sp]
        fp = split_frac_pt[sp]
        ax00.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                  f"n={n}  frac={fp:.1f}%",
                  va="center", fontsize=7.5)
    total = sum(split_counts.values())
    ax00.set_xlim(right=max(split_counts.values()) * 1.45)
    ax00.set_title(f"Dataset Split Distribution  (total={total})",
                   fontsize=10, pad=5)

    # ── [0,1] 点数直方图 ─────────────────────────────────────
    ax01 = fig.add_subplot(gs[0, 1])
    styled_hist(ax01, "n_points", "Point Count per Block", "# Points")

    # ── [0,2] 结构面比例直方图 ───────────────────────────────
    ax02 = fig.add_subplot(gs[0, 2])
    styled_hist(ax02, "frac_ratio_%", "Fracture Ratio per Block", "Ratio (%)", bins=20)

    # ── [1,0] 实例数直方图 ───────────────────────────────────
    ax10 = fig.add_subplot(gs[1, 0])
    styled_hist(ax10, "n_instances", "Instance Count per Block", "# Instances", bins=15)

    # ── [1,1] 内存直方图 ─────────────────────────────────────
    ax11 = fig.add_subplot(gs[1, 1])
    styled_hist(ax11, "mem_MB", "Memory per Block", "Memory (MB)")

    # ── [1,2] 点数 vs 结构面% 散点图 ────────────────────────
    ax12 = fig.add_subplot(gs[1, 2])
    for sp in active_splits:
        sub = df_rep[df_rep["split"] == sp]
        ax12.scatter(sub["n_points"], sub["frac_ratio_%"],
                     c=SPLIT_COLORS[sp], alpha=0.55, s=18,
                     edgecolors="none", label=sp)
    ax12.set_xlabel("# Points", fontsize=8)
    ax12.set_ylabel("Fracture Ratio (%)", fontsize=8)
    ax12.set_title("Points vs Fracture Ratio", fontsize=10, pad=5)
    ax12.tick_params(labelsize=7)
    ax12.grid(alpha=0.2, linestyle=":")

    # ── [2, :] 块点数折线图（各 split 分段，按顺序排列） ────
    ax2 = fig.add_subplot(gs[2, :])
    offset = 0
    xtick_pos, xtick_lbl = [], []
    for sp in active_splits:
        sub = df_rep[df_rep["split"] == sp]["n_points"].values
        if len(sub) == 0:
            continue
        x = np.arange(offset, offset + len(sub))
        ax2.fill_between(x, sub, alpha=0.18, color=SPLIT_COLORS[sp])
        ax2.plot(x, sub, color=SPLIT_COLORS[sp], linewidth=0.9,
                 label=f"{sp} (n={len(sub)})")
        ax2.axvline(offset, color="gray", linewidth=0.6, linestyle=":")
        xtick_pos.append(offset + len(sub) / 2)
        xtick_lbl.append(sp)
        offset += len(sub)
    ax2.axhline(df_rep["n_points"].mean(), color="red", linewidth=0.8,
                linestyle="--", alpha=0.7, label=f"overall μ={df_rep['n_points'].mean():.0f}")
    ax2.set_xlabel("Block Index (by split)", fontsize=8)
    ax2.set_ylabel("# Points", fontsize=8)
    ax2.set_title("Point Count Across All Blocks", fontsize=10, pad=5)
    ax2.tick_params(labelsize=7)
    ax2.grid(alpha=0.2, linestyle=":")
    ax2.set_xticks(xtick_pos)
    ax2.set_xticklabels(xtick_lbl, fontsize=8)

    # ── 全局图例 ─────────────────────────────────────────────
    patches = [mpatches.Patch(color=SPLIT_COLORS[s], label=s, alpha=0.75)
               for s in active_splits]
    fig.legend(handles=patches, loc="lower center", ncol=len(active_splits),
               fontsize=9, frameon=True,
               bbox_to_anchor=(0.5, 0.002))

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  可视化报告已保存: {out_path}")


# ─────────────────────────────────────────────────────────────
#  主流程
# ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)

    ratios = args.split_ratios
    assert abs(sum(ratios) - 1.0) < 1e-3 or sum(ratios) <= 1.0, \
        "split_ratios 之和必须 ≤ 1.0"

    out_root = Path(args.output)
    for d in ["train", "val", "test", "ply"]:
        (out_root / d).mkdir(parents=True, exist_ok=True)

    # Step 1: 读 CSV
    df = load_csv(args.input)

    # Step 2: 3D 切块
    tile_size = (args.tile_x, args.tile_y, args.tile_z)
    blocks = tile_pointcloud(df, tile_size, args.overlap,
                             args.min_points, args.max_points)

    # Step 3: 划分
    print(f"\n[Step 3/5] 数据集划分  train={ratios[0]:.0%} val={ratios[1]:.0%} test={ratios[2]:.0%}")
    train_idx, val_idx, test_idx = split_blocks(blocks, ratios, args.seed)
    split_map = (
            [(i, "train") for i in train_idx] +
            [(i, "val") for i in val_idx] +
            [(i, "test") for i in test_idx]
    )
    print(f"  train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")

    # Step 4 & 5: 构建样本 + 保存
    print(f"\n[Step 4/5] 构建并保存数据块")
    save_ply_flag = args.save_ply and HAS_PLY

    records = []
    split_counters = {"train": 0, "val": 0, "test": 0}

    for block_idx, split in tqdm(split_map, desc="  处理中", ncols=70):
        blk = blocks[block_idx]
        indices = blk["indices"]
        origin = blk["origin_xyz"]
        count = split_counters[split]
        split_counters[split] += 1

        sample = build_sample(df, indices)

        # ── 统计指标 ──
        disc_pts = int((sample["segment"] == 1).sum())
        n_pts = len(indices)
        frac_pct = 100.0 * disc_pts / max(n_pts, 1)
        n_inst = int(np.unique(sample["instance"][sample["instance"] >= 0]).shape[0])
        mem_bytes = (sample["coord"].nbytes + sample["feat"].nbytes +
                     sample["segment"].nbytes + sample["instance"].nbytes)
        mem_mb = mem_bytes / 1024 ** 2

        rec = dict(
            split=split,
            block_id=f"{split}_{count:05d}",
            tile_origin=f"({origin[0]:.1f},{origin[1]:.1f},{origin[2]:.1f})",
            n_points=n_pts,
            frac_pts=disc_pts,
            frac_ratio_percent=frac_pct,
            n_instances=n_inst,
            mem_MB=round(mem_mb, 3),
        )
        # 兼容旧代码和报表
        rec["frac_ratio_%"] = rec.pop("frac_ratio_percent")
        records.append(rec)

        # ── 保存 .pth (Pointcept DefaultDataset 格式) ──
        pth_name = f"{split}_{count:05d}.pth"
        torch.save(
            dict(
                coord=torch.from_numpy(sample["coord"]),
                feat=torch.from_numpy(sample["feat"]),
                segment=torch.from_numpy(sample["segment"]),
                instance=torch.from_numpy(sample["instance"]),
            ),
            out_root / split / pth_name,
        )

        # ── 保存 PLY ──
        if save_ply_flag:
            ply_name = f"{split}_{count:05d}.ply"
            save_ply(df, indices, str(out_root / "ply" / ply_name))

    # Step 5: 报表
    print(f"\n[Step 5/6] 生成数据集报表")
    df_rep = generate_report(records, str(out_root / "report.csv"))

    # Step 6: 可视化仪表板
    print(f"\n[Step 6/6] 生成可视化报告仪表板")
    generate_report_dashboard(df_rep, str(out_root / "report_dashboard.png"))

    print(f"\n{'=' * 60}")
    print(f"✅ 预处理完成！")
    print(f"   输出目录  : {out_root.resolve()}")
    print(f"   .pth 格式 : Pointcept DefaultDataset 直接兼容")
    print(f"   PLY 可视化: {out_root / 'ply'}")
    print(f"   数据报表  : {out_root / 'report.csv'}")
    print(f"   直方图    : {out_root / 'report_dashboard.png'}")
    print(f"\n📐 PTv3 config.py 对应设置:")
    print(f"   data_root      = \"{out_root.resolve()}\"")
    print(f"   in_channels    = 6    # nx ny nz R G B")
    print(f"   num_classes    = 2    # 背景 / 结构面")
    print(f"   grid_size      = 0.02 # 体素化步长(m)")
    print(f"   point_max      = {args.max_points}  # SphereCrop 上限")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
