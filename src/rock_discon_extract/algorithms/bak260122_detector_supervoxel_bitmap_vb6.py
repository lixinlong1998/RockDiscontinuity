# RockDiscontinuity/src/rock_discon_extract/algorithms/detector_supervoxel.py
'''
我改了哪些地方（只改 Step5，最小侵入）
修改点 1（关键）：seed 必须立刻从 unmerged 移除
原逻辑只把被吸收的 j 从 unmerged 移除，但 seed 一直留在 unmerged。这会导致两类严重后果：
跨簇串联：后续别的 seed 生长时，可能把之前某个 seed 再吸收进去，等价于把两个本不该连通的生长簇“串”起来；
self-merge：当合并 j 后，把 patch_neighbors[j] 加入队列时，seed_id 会被重新加入候选列表（因为 seed 还在 unmerged），于是出现 seed==neighbor 且被 “merged” 的记录（你 Step5 CSV 里已经出现了这种情况）。
修复：进入一个 seed 生长簇时，立即 unmerged.discard(seed_id)。
修改点 2：显式禁止 self-merge / 重复 merge
即使修改点 1 解决了大部分 self-merge 来源，我仍然增加了硬保护：
if (j == seed_id) or (j in merged_patch_ids): continue
修改点 3：增加“面内距离（质心欧氏距离）”约束，抑制远距离误合并
你指出的核心问题之一是：仅用“邻居质心到 seed 平面的法向距离”无法约束切向（面内）分离，所以即使两片段相距很远，只要在同一近似平面上，依旧可能通过 patch_distance。
我新增了一个温和阈值：
patch_centroid_th = max(3*voxel_size, 2*patch_distance)
合并条件从 (ang_diff < patch_angle and dist_diff < patch_distance) 变为：
pass_ang and pass_dist and pass_centroid
并在 supervoxel_debug_Step5_patch_merge_records.csv 追加三列（不破坏旧列）：
centroid_diff
pass_patch_centroid
patch_centroid_th
'''
from typing import List, Dict, Tuple, Set, Optional
import math
import os
import time
import csv
import json
import numpy as np

from .base import PlaneDetectionAlgorithm
from ..geometry import Discontinuity, Plane, Segment
from ..pointcloud import PointCloud
from ..logging_utils import LoggerManager, Timer

# 尝试导入 Open3D 库
try:
    import open3d as o3d
except ImportError:
    o3d = None

SUPERVOXEL_DEBUG_EXPORT_DIR = r'D:\Research\20250313_RockFractureSeg\Code\RockDiscontinuity\result\supervoxel_debug_visualizer'  # 调试导出目录；为空则不导出。
SUPERVOXEL_DEBUG_BASENAME = r'supervoxel_debug'  # 文件名前缀(可选)。
SUPERVOXEL_DEBUG_EXPORT = True  # 开启or关闭Debug。
SUPERVOXEL_DEBUG_EXPORT_VOXEL = True


# -----------------------------------------------------------------------------
# Debug / Monitoring helper functions (packed outside class for readability)
# NOTE: These helpers MUST NOT change algorithm logic; they only wrap logging/export.
# -----------------------------------------------------------------------------

def _MonCheckInputNormals(logger, num_points: int, normals_valid: np.ndarray) -> bool:
    """Log valid normal stats; return False if input empty (caller should exit early)."""
    valid_normals = int(np.count_nonzero(normals_valid))
    valid_ratio = float(valid_normals) / float(max(num_points, 1))
    logger.info(f"[MON] Points={num_points}, ValidNormals={valid_normals} ({valid_ratio:.2%})")
    if num_points == 0:
        logger.warning("Supervoxel: 输入点云为空，返回空结果.")
        return False
    logger.info(f"SupervoxelDetector starts: {num_points} points.")
    return True


def _MonLogVoxelOccupancy(logger, voxel_map: Dict[Tuple[int, int, int], List[int]]) -> None:
    """Log voxel occupancy distribution."""
    voxel_counts = np.array([len(v) for v in voxel_map.values()], dtype=int) if len(voxel_map) > 0 else np.array([],
                                                                                                                 dtype=int)
    if voxel_counts.size > 0:
        logger.info(
            f"[MON] VoxelOcc: mean={voxel_counts.mean():.1f}, "
            f"min={voxel_counts.min()}, p50={np.percentile(voxel_counts, 50):.0f}, max={voxel_counts.max()}"
            f"p25={np.percentile(voxel_counts, 25):.0f}, p75={np.percentile(voxel_counts, 75):.0f}"
        )


def _MonLogSeedClustersAndRemain(logger, clusters: List[Dict], remain_points: Dict[Tuple[int, int, int], Set[int]],
                                 tag: str) -> None:
    """Log seed cluster size distribution and remain stats."""
    if len(clusters) > 0:
        seed_sizes = np.array([len(c.get("points", [])) for c in clusters], dtype=int)
        logger.info(
            f"[MON] SeedClusters: n={len(clusters)}, meanPts={seed_sizes.mean():.1f}, "
            f"p50={np.percentile(seed_sizes, 50):.0f}, p90={np.percentile(seed_sizes, 90):.0f}, "
            f"minPts={seed_sizes.min()}, maxPts={seed_sizes.max()}"
        )
    remain_total = int(sum(len(s) for s in remain_points.values()))
    remain_vox = int(sum(1 for s in remain_points.values() if len(s) > 0))
    logger.info(f"[MON] {tag}: points={remain_total}, voxels={remain_vox}")


def _MonLogSupervoxels(logger, supervoxels: List[Dict], remain_points: Dict[Tuple[int, int, int], Set[int]],
                       absorbed_points_total: Set[int]) -> None:
    """Log supervoxel patch sizes and remain stats."""
    if len(supervoxels) > 0:
        sv_sizes = np.array([len(sv.get("points", [])) for sv in supervoxels], dtype=int)
        logger.info(
            f"[MON] Supervoxels: n={len(supervoxels)}, meanPts={sv_sizes.mean():.1f}, "
            f"p50={np.percentile(sv_sizes, 50):.0f}, p90={np.percentile(sv_sizes, 90):.0f}, "
            f"minPts={sv_sizes.min()}, maxPts={sv_sizes.max()}"
        )
    remain_total = int(sum(len(s) for s in remain_points.values()))
    remain_vox = int(sum(1 for s in remain_points.values() if len(s) > 0))
    logger.info(
        f"[MON] RemainAfterStep4: points={remain_total}, voxels={remain_vox}, absorbed={len(absorbed_points_total)}")


def _MonLogSuperGrowAggAndBitmap(
        logger,
        mon_sv_seeds: int,
        mon_sv_success: int,
        mon_sv_iters_total: int,
        mon_sv_absorb_total: int,
        mon_sv_best_pts_total: int,
        mon_bitmap_calls: int,
        mon_bitmap_inliers: int,
        mon_bitmap_connected: int,
        mon_bitmap_leftover: int,
) -> None:
    """Log aggregated supervoxel growth and bitmap filter stats."""
    logger.info(
        f"[MON] SuperGrowAgg: seeds={mon_sv_seeds}, success={mon_sv_success}, "
        f"itersTotal={mon_sv_iters_total}, absorbedTotal={mon_sv_absorb_total}, "
        f"bestPtsTotal={mon_sv_best_pts_total}"
    )
    try:
        if mon_bitmap_calls > 0:
            conn_ratio = (float(mon_bitmap_connected) / float(max(mon_bitmap_inliers, 1)))
            logger.info(
                f"[MON] BitmapMCC: calls={mon_bitmap_calls}, "
                f"inliers={mon_bitmap_inliers}, connected={mon_bitmap_connected}, "
                f"leftover={mon_bitmap_leftover}, connRatio={conn_ratio:.2%}"
            )
    except Exception:
        pass


def _MonLogPlaneSizesAndUnassigned(logger, discontinuities: List['Discontinuity'],
                                   remain_points: Dict[Tuple[int, int, int], Set[int]]) -> None:
    """Log final plane size distribution and remaining unassigned points."""
    if len(discontinuities) > 0:
        plane_sizes = np.array(
            [len(d.segments[0].points) if (d.segments and hasattr(d.segments[0], "points")) else 0 for d in
             discontinuities],
            dtype=int
        )
        if plane_sizes.size > 0:
            logger.info(
                f"[MON] PlaneSizes: meanPts={plane_sizes.mean():.1f}, p50={np.percentile(plane_sizes, 50):.0f}, "
                f"p90={np.percentile(plane_sizes, 90):.0f}, minPts={plane_sizes.min()}, maxPts={plane_sizes.max()}"
            )
    remain_total = int(sum(len(s) for s in remain_points.values()))
    remain_vox = int(sum(1 for s in remain_points.values() if len(s) > 0))
    logger.info(f"[MON] UnassignedRemainEnd: points={remain_total}, voxels={remain_vox}")


def _DebugExportStep2(
        detector,
        debug_dir: str,
        debug_base: str,
        num_points: int,
        coords: np.ndarray,
        normals: np.ndarray,
        rgb: np.ndarray,
        curvature: np.ndarray,
        clusters: List[Dict],
        voxel_map: Dict[Tuple[int, int, int], List[int]],
        step2_voxel_diag_records: List[Dict],
) -> None:
    """Export Step2 point-level colored CSVs + voxel distributions + per-voxel figures."""
    # --- 1) Step2: voxel-patch 的两种配色方案 ---
    # 1.1 基于 seed_id 的可复现彩色（排除纯红/纯绿/纯蓝）
    colored_sets_id: List[Tuple[Set[int], Tuple[int, int, int]]] = []
    for cid, c in enumerate(clusters):
        pts = set(c.get("points", set()))
        if len(pts) == 0:
            continue
        colored_sets_id.append((pts, GenerateDebugColorFromId(cid)))

    drdgdb_step2_id = detector._FillDrDgDbBySets(
        num_points=num_points,
        default_rgb=(160, 160, 160),
        colored_sets=colored_sets_id
    )

    # 1.2 纯绿色（便于叠加后续步骤结构）
    voxel_patch_all_pts: Set[int] = set()
    for c in clusters:
        voxel_patch_all_pts |= set(c.get("points", set()))

    drdgdb_step2_green = detector._FillDrDgDbBySets(
        num_points=num_points,
        default_rgb=(160, 160, 160),
        colored_sets=[(voxel_patch_all_pts, (0, 255, 0))]
    )

    try:
        csv_path = os.path.join(debug_dir, f"{debug_base}_Step2_voxel_patch_idcolor.csv")
        detector._ExportDebugPointLevelCsv(csv_path, coords, normals, rgb, curvature, drdgdb_step2_id)
        detector.logger.info(f"[DEBUG] Step2 point-level CSV exported: {csv_path}")
    except Exception as e:
        detector.logger.warning(f"[DEBUG] Step2(idcolor) export failed: {e}")

    try:
        csv_path = os.path.join(debug_dir, f"{debug_base}_Step2_voxel_patch_green.csv")
        detector._ExportDebugPointLevelCsv(csv_path, coords, normals, rgb, curvature, drdgdb_step2_green)
        detector.logger.info(f"[DEBUG] Step2 point-level CSV exported: {csv_path}")
    except Exception as e:
        detector.logger.warning(f"[DEBUG] Step2(green) export failed: {e}")

    # --- 2) Step2: 每个 voxel 内 patch vs other 的分布诊断表 ---
    try:
        rec_path = os.path.join(debug_dir, f"{debug_base}_Step2_voxel_patch_distributions.csv")
        if step2_voxel_diag_records:
            fieldnames = list(step2_voxel_diag_records[0].keys())
            detector._ExportDebugRecordsCsv(rec_path, fieldnames=fieldnames, records=step2_voxel_diag_records)
            detector.logger.info(f"[DEBUG] Step2 voxel-distributions CSV exported: {rec_path}")
    except Exception as e:
        detector.logger.warning(f"[DEBUG] Step2 voxel-distributions export failed: {e}")

    # --- 3) Step2: 为每个 voxel 绘制 dipdir_rose 与 points_stereonet_kde ---
    if SUPERVOXEL_DEBUG_EXPORT_VOXEL:
        try:
            from ..visualizer import ResultsVisualizer  # 延迟导入，避免非调试运行时引入绘图依赖
            voxel_csv_dir = os.path.join(debug_dir, f"{debug_base}_Step2_voxels_csv")
            voxel_fig_dir = os.path.join(debug_dir, f"{debug_base}_Step2_voxel_figs")
            os.makedirs(voxel_csv_dir, exist_ok=True)
            os.makedirs(voxel_fig_dir, exist_ok=True)

            voxel_csv_paths_list: List[str] = []
            for vid, vid_indices in voxel_map.items():
                if vid_indices is None:
                    continue
                if len(vid_indices) < 10:
                    continue
                vx, vy, vz = vid
                voxel_csv_path = os.path.join(voxel_csv_dir, f"{debug_base}_voxel_{vx}_{vy}_{vz}.csv")
                detector._ExportDebugPointLevelCsvSubset(
                    csv_path=voxel_csv_path,
                    indices=vid_indices,
                    coords=coords,
                    normals=normals,
                    rgb=rgb,
                    curvature=curvature
                )
                voxel_csv_paths_list.append(voxel_csv_path)

            if voxel_csv_paths_list:
                paths_list = [(voxel_fig_dir, csv_file_path) for csv_file_path in voxel_csv_paths_list]
                viz = ResultsVisualizer(paths_list=paths_list)
                viz.ExportAllSingleAnalysis(
                    plots_name=["points_stereonet_kde"],
                    output_formats=("png",),
                    show=False
                )
                detector.logger.info(f"[DEBUG] Step2 voxel figures exported: {voxel_fig_dir}")
        except Exception as e:
            detector.logger.warning(f"[DEBUG] Step2 voxel-figures export failed: {e}")


def _DebugExportStep3(
        detector,
        debug_dir: str,
        debug_base: str,
        num_points: int,
        coords: np.ndarray,
        normals: np.ndarray,
        rgb: np.ndarray,
        curvature: np.ndarray,
        clusters: List[Dict],
        n_voxel_seeds_step2: int,
        edge_patch_new_points: Set[int],
        edge_patch_merged_points: Set[int],
        edge_merge_records: List[Dict],
) -> None:
    """Export Step3 point-level colored CSVs + edge merge diagnostics."""
    # ========== Step3 配色方案 A ==========
    colored_sets_step3_a: List[Tuple[Set[int], Tuple[int, int, int]]] = []
    for cid, c in enumerate(clusters[:n_voxel_seeds_step2]):
        pts = set(c.get("points", set()))
        if len(pts) == 0:
            continue
        colored_sets_step3_a.append((pts, GenerateDebugColorFromId(cid)))
    colored_sets_step3_a.append((edge_patch_new_points, (0, 0, 255)))

    drdgdb_step3_a = detector._FillDrDgDbBySets(
        num_points=num_points,
        default_rgb=(160, 160, 160),
        colored_sets=colored_sets_step3_a
    )

    # ========== Step3 配色方案 B ==========
    voxel_pts_all: Set[int] = set()
    for c in clusters[:n_voxel_seeds_step2]:
        voxel_pts_all |= set(c.get("points", set()))
    voxel_pts_green = voxel_pts_all - set(edge_patch_merged_points)

    drdgdb_step3_b = detector._FillDrDgDbBySets(
        num_points=num_points,
        default_rgb=(160, 160, 160),
        colored_sets=[
            (voxel_pts_green, (0, 255, 0)),
            (edge_patch_new_points, (0, 0, 255)),
            (edge_patch_merged_points, (0, 255, 255)),
        ]
    )

    try:
        csv_path = os.path.join(debug_dir, f"{debug_base}_Step3_edge_patch_blue_mergeToVoxelColor.csv")
        detector._ExportDebugPointLevelCsv(csv_path, coords, normals, rgb, curvature, drdgdb_step3_a)
        detector.logger.info(f"[DEBUG] Step3(A) point-level CSV exported: {csv_path}")
    except Exception as e:
        detector.logger.warning(f"[DEBUG] Step3(A) export failed: {e}")

    try:
        csv_path = os.path.join(debug_dir, f"{debug_base}_Step3_edge_patch_greenBlueCyan.csv")
        detector._ExportDebugPointLevelCsv(csv_path, coords, normals, rgb, curvature, drdgdb_step3_b)
        detector.logger.info(f"[DEBUG] Step3(B) point-level CSV exported: {csv_path}")
    except Exception as e:
        detector.logger.warning(f"[DEBUG] Step3(B) export failed: {e}")

    try:
        rec_path = os.path.join(debug_dir, f"{debug_base}_Step3_edge_merge_records.csv")
        detector._ExportDebugRecordsCsv(
            rec_path,
            fieldnames=[
                "voxel_id",
                "edge_patch_points",
                "merged",
                "merge_reason",
                "best_candidate_cluster_idx",
                "best_candidate_voxel_id",
                "best_mw",
                "best_nw",
                "best_score_Pw",
                "pass_score_Pw",
                "best_pass_cluster_idx",
                "pass_edge_distance",
                "pass_edge_angle",
                "edge_distance",
                "edge_angle",
            ],
            records=edge_merge_records
        )
        detector.logger.info(f"[DEBUG] Step3 merge-records CSV exported: {rec_path}")
    except Exception as e:
        detector.logger.warning(f"[DEBUG] Step3 merge-records export failed: {e}")


def _DebugExportStep4(
        detector,
        debug_dir: str,
        debug_base: str,
        num_points: int,
        coords: np.ndarray,
        normals: np.ndarray,
        rgb: np.ndarray,
        curvature: np.ndarray,
        clusters: List[Dict],
        n_voxel_seeds_step2: int,
        edge_patch_new_points: Set[int],
        edge_patch_merged_points: Set[int],
        absorbed_sets_with_color: List[Tuple[Set[int], Tuple[int, int, int]]],
        absorbed_points_total: Set[int],
        supervoxels: List[Dict],
        supervoxel_growth_records: List[Dict],
) -> None:
    """Export Step4 multi-scheme point-level CSVs + supervoxel growth diagnostics.
        1) Supervoxel growth diagnostics table:
            - {debug_base}_Step4_supervoxel_growth_records.csv

        2) Per-supervoxel CSVs (for CloudCompare query / localization):
            - folder: {debug_base}_Step5_supervoxels_csv
            - file:   {debug_base}_supervoxel_<id>.csv
    """
    # ========== Step4 配色方案 1 ==========
    drdgdb_step4_absorbed_seedcolor = detector._FillDrDgDbBySets(
        num_points=num_points,
        default_rgb=(160, 160, 160),
        colored_sets=absorbed_sets_with_color
    )
    try:
        csv_path = os.path.join(debug_dir, f"{debug_base}_Step4_absorbed_seedcolor.csv")
        detector._ExportDebugPointLevelCsv(csv_path, coords, normals, rgb, curvature, drdgdb_step4_absorbed_seedcolor)
        detector.logger.info(f"[DEBUG] Step4(1) point-level CSV exported: {csv_path}")
    except Exception as e:
        detector.logger.warning(f"[DEBUG] Step4(1) export failed: {e}")

    # ========== Step4 配色方案 2 ==========
    voxel_pts_all: Set[int] = set()
    for c in clusters[:n_voxel_seeds_step2]:
        voxel_pts_all |= set(c.get("points", set()))
    voxel_pts_green = voxel_pts_all - set(edge_patch_merged_points)

    drdgdb_step4_absorbed_red_with_prior = detector._FillDrDgDbBySets(
        num_points=num_points,
        default_rgb=(160, 160, 160),
        colored_sets=[
            (voxel_pts_green, (0, 255, 0)),
            (edge_patch_new_points, (0, 0, 255)),
            (edge_patch_merged_points, (0, 255, 255)),
            (absorbed_points_total, (255, 0, 0)),
        ]
    )
    try:
        csv_path = os.path.join(debug_dir, f"{debug_base}_Step4_absorbed_red_with_prior.csv")
        detector._ExportDebugPointLevelCsv(csv_path, coords, normals, rgb, curvature,
                                           drdgdb_step4_absorbed_red_with_prior)
        detector.logger.info(f"[DEBUG] Step4(2) point-level CSV exported: {csv_path}")
    except Exception as e:
        detector.logger.warning(f"[DEBUG] Step4(2) export failed: {e}")

    # ========== Step4 配色方案 3 ==========
    colored_sets_step4_patches: List[Tuple[Set[int], Tuple[int, int, int]]] = []
    for sid, sv in enumerate(supervoxels):
        pts = set(sv.get("points", set()))
        if len(pts) == 0:
            continue
        colored_sets_step4_patches.append((pts, GenerateDebugColorFromId(sid)))

    drdgdb_step4_patches_idcolor = detector._FillDrDgDbBySets(
        num_points=num_points,
        default_rgb=(160, 160, 160),
        colored_sets=colored_sets_step4_patches
    )
    try:
        csv_path = os.path.join(debug_dir, f"{debug_base}_Step4_supervoxel_patches_idcolor.csv")
        detector._ExportDebugPointLevelCsv(csv_path, coords, normals, rgb, curvature, drdgdb_step4_patches_idcolor)
        detector.logger.info(f"[DEBUG] Step4(3) point-level CSV exported: {csv_path}")
    except Exception as e:
        detector.logger.warning(f"[DEBUG] Step4(3) export failed: {e}")

    # --- (A) Step4 supervoxel_growth_records CSV exports ---
    try:
        rec_path = os.path.join(debug_dir, f"{debug_base}_Step4_supervoxel_growth_records.csv")
        detector._ExportDebugRecordsCsv(
            rec_path,
            fieldnames=[
                "seed_idx",
                "seed_type",
                "seed_points",
                "best_points",
                "absorbed_points",
                "cand_points_seen",
                "iters",
                "init_dist_th",
                "init_ang_th",
                "distance_step",
                "angle_step",
                "final_dist_th",
                "final_ang_th",
                "orient_diff",
                "orient_diff_last",
                "max_refit_error",
                "fallback_used",
                "fallback_reason",
                "blocked_by_max_refit_error",
                "success",
            ],
            records=supervoxel_growth_records
        )
        detector.logger.info(f"[DEBUG] Step4 growth-records CSV exported: {rec_path}")
    except Exception as e:
        detector.logger.warning(f"[DEBUG] Step4 growth-records export failed: {e}")

    # --- (B) Step4 per-supervoxel CSV exports ---
    try:
        sv_csv_dir = os.path.join(debug_dir, f"{debug_base}_Step4_supervoxels_csv")
        os.makedirs(sv_csv_dir, exist_ok=True)

        if supervoxels:
            for sid, sv in enumerate(supervoxels):
                pts = sv.get("points", None)
                if pts is None:
                    continue
                if isinstance(pts, set):
                    idx = np.array(list(pts), dtype=np.int64)
                else:
                    idx = np.asarray(list(pts), dtype=np.int64).reshape(-1)

                if idx.size == 0:
                    continue

                csv_path = os.path.join(sv_csv_dir, f"{debug_base}_supervoxel_{sid}.csv")
                detector._ExportDebugPointLevelCsvSubset(
                    csv_path=csv_path,
                    indices=idx,
                    coords=coords,
                    normals=normals,
                    rgb=rgb,
                    curvature=curvature
                )
        detector.logger.info(f"[DEBUG] Step4 supervoxel CSVs exported: {sv_csv_dir}")
    except Exception as e:
        detector.logger.warning(f"[DEBUG] Step4 supervoxel-csv export failed: {e}")


def _DebugExportStep5(detector, debug_dir: str, debug_base: str, patch_merge_records: List[Dict]) -> None:
    """Export Step5 patch-merge diagnostics table."""
    try:
        rec_path = os.path.join(debug_dir, f"{debug_base}_Step5_patch_merge_records.csv")
        if patch_merge_records:
            fieldnames = list(patch_merge_records[0].keys())
            detector._ExportDebugRecordsCsv(rec_path, fieldnames=fieldnames, records=patch_merge_records)
            detector.logger.info(f"[DEBUG] Step5 patch-merge-records CSV exported: {rec_path}")
    except Exception as e:
        detector.logger.warning(f"[DEBUG] Step5 patch-merge-records export failed: {e}")


class SupervoxelDetector(PlaneDetectionAlgorithm):
    """
    基于超体素与区域生长的高精度平面检测算法 (Supervoxel-RegionGrowing).

    功能简介:
        针对包含大量平面结构面的岩体点云, 通过体素化、局部 RANSAC 平面提取、边缘平面分割、
        超体素生长以及基于片段的区域生长, 自动提取不连续面 (结构面) 的平面片段并进行合并,
        最终输出一组代表实际结构面的 Discontinuity 对象。

    实现思路(详细):
        1. 体素划分:
            - 使用 voxel_size 对点云做规则 3D 网格划分, 为每个点分配体素索引 (vx, vy, vz)。
            - 构建 voxel_id -> 点索引列表, 以及 point_index -> voxel_id 映射。
        2. 体素内局部平面提取 (voxel-patch):
            - 对每个体素 voxel_i 内点集执行一次 RANSAC:
                * 使用 ransac_distance 作为距离阈值提取局部平面。
                * 对得到的 inlier 再用 ransac_angle 做法向角度过滤, 确保 patch 内点在法向上也高度一致。
            - 若过滤后的 inlier 数 ≥ min_plane_points:
                * 该体素被视为 coplanar voxel, inlier 作为一个 voxel-patch, 记为一个生长单元 cluster。
                * 其余点作为 remain_points[voxel_i]。
            - 否则:
                * 该体素被视为 non-coplanar 或噪声体素, 所有点加入 remain_points[voxel_i]。
        3. 非共面体素内边缘平面提取 (edge-patch) 与拼接:
            - 对每个体素 voxel_i 的剩余点集 remain_points[voxel_i]:
                * 若点数 > min_edge_points, 则循环执行:
                    (a) 在该体素剩余点中执行 RANSAC(距离阈值 ransac_distance):
                        - 得到候选 edge-patch 的 inlier。
                        - 用 ransac_angle 对 inlier 做法向过滤, 获取精确 edge-patch。
                    (b) 若 edge-patch 点数 ≥ min_edge_patch_points:
                        - 在 voxel_i 的邻域体素中搜索已有平面 cluster, 计算:
                            + mw: edge-patch 点到候选平面的平均距离 (需 < edge_distance)。
                            + nw: edge-patch 平面法向与候选平面法向的夹角 (需 < edge_angle)。
                            + Pw = sqrt( (mw/edge_distance)^2 + (nw/edge_angle)^2 ) 作为相似性指标。
                        - 若存在满足阈值的邻近平面, 选取 Pw 最小者, 将 edge-patch 并入该平面 cluster。
                        - 否则, 将该 edge-patch 作为新的 cluster (新的平面生长单元)。
                        - 从 remain_points[voxel_i] 中移除 edge-patch 点, 再次循环。
                    (c) 若 edge-patch 点数 < min_edge_patch_points:
                        - 认为该体素剩余点无法构成稳定平面, 停止对该体素的边缘检测。
        4. 超体素平面生长 (Supervoxel segmentation):
            - 每个 cluster (voxel-patch 或 edge-patch) 都作为一个 grow_unit 种子:
                * seed_voxels = cluster["voxel_ids"] (可能跨多个体素)。
                * 构建候选体素集合: 所有 seed_voxels 及其一圈 26 邻域体素。
                * 从这些体素的 remain_points 中收集候选散点 candidate_points。
            - 以 cluster["orig_points"] 作为初始平面点集, 设置初始阈值:
                * dist_th = super_distance, ang_th = super_angle。
            - 迭代区域生长:
                * 在当前阈值下, 从 candidate_points 中选取“距离 < dist_th 且 法向角度 < ang_th”的点加入平面。
                * 对加入后的点集做 PCA, 拟合新的平面法向, 计算与原始法向的夹角变化 orientation_diff。
                * 若 orientation_diff ≤ max_refit_error, 接受当前点集为 supervoxel patch。
                * 否则收缩 dist_th, ang_th (按 distance_step, angle_step) 并重新尝试。
            - 对每个 supervoxel patch:
                * 用最终点集重新统计其覆盖的 voxel_ids (从 point_voxel_map 反查)。
                * 将 patch 内点从 remain_points 中删除, 不再参与其它 patch 生长。
        5. 基于平面片段的区域生长 (Patch-based region growing):
            - 将所有 supervoxel patch 存于 supervoxels 列表, 每个元素包含:
                * points: 点索引集合。
                * voxel_ids: 该 patch 覆盖的体素集合。
                * normal, d: 平面参数。
                * error: 生长过程中的最终法向变化。
            - 构建 patch 邻接关系:
                * 对每个体素 vid, 其内部所有 patch 互为邻居。
                * 对 vid 的 26 邻域体素中的 patch, 与 vid 内的 patch 互为邻居。
            - 按 error 从小到大排序 patch, 逐个作为 seed 做区域生长:
                * 若邻居 patch 与 seed 在法向夹角 < patch_angle, 质心到 seed 平面距离 < patch_distance,
                  则将其合并进 seed, 更新 seed 点集与 voxel_ids, 并重新拟合平面。
                * 同时将该邻居的邻居加入待检查队列, 实现多步生长。
            - 每次生长结束, 使用合并后的所有点重新拟合平面, 生成一个 Discontinuity。
    """

    def __init__(
            self,
            voxel_size: float = 0.05,
            sample_spacing: float = 0.05,
            ransac_distance: float = 0.05,
            ransac_angle: float = 5.0,
            min_plane_points: int = 30,
            edge_distance: float = 0.05,
            edge_angle: float = 5.0,
            min_edge_points: int = 30,
            min_edge_patch_points: int = 20,
            super_distance: float = 0.15,
            super_angle: float = 15.0,
            max_refit_error: float = 5.0,
            distance_step: float = 0.01,
            angle_step: float = 1.0,
            patch_distance: float = 0.30,
            patch_angle: float = 20.0
    ):
        """
        输入变量:
            voxel_size: float
                体素划分网格尺寸(单位: m)。
            ransac_distance: float
                初始 RANSAC 平面拟合的距离阈值。
            ransac_angle: float
                初始 RANSAC 平面拟合的法向夹角阈值(度), 用于对 inlier 做二次筛选。
            min_plane_points: int
                体素内平面被接受为 voxel-patch 的最少点数。
            edge_distance: float
                边缘平面与邻域平面拼接的距离阈值 Mw。
            edge_angle: float
                边缘平面与邻域平面拼接的法向角度阈值 Nw(度)。
            min_edge_points: int
                启动边缘平面提取的 voxel 剩余点数阈值, 少于此值不做边缘检测。
            min_edge_patch_points: int
                单个 edge-patch 被接受的最少点数。
            super_distance: float
                超体素区域生长初始距离阈值。
            super_angle: float
                超体素区域生长初始法向角度阈值(度)。
            max_refit_error: float
                超体素生长过程中允许的最大法向变化(度)。
            distance_step: float
                超体素生长每轮缩小的距离步长。
            angle_step: float
                超体素生长每轮缩小的角度步长。
            patch_distance: float
                patch-level 区域生长时的距离阈值。
            patch_angle: float
                patch-level 区域生长时的法向角度阈值(度)。
        """
        super().__init__(name="Supervoxel")
        self.voxel_size = voxel_size
        self.sample_spacing = sample_spacing
        self.ransac_distance = ransac_distance
        self.ransac_angle = ransac_angle
        self.min_plane_points = min_plane_points
        self.edge_distance = edge_distance
        self.edge_angle = edge_angle
        self.min_edge_points = min_edge_points
        self.min_edge_patch_points = min_edge_patch_points
        self.super_distance = super_distance
        self.super_angle = super_angle
        self.max_refit_error = max_refit_error
        self.distance_step = distance_step
        self.angle_step = angle_step
        self.patch_distance = patch_distance
        self.patch_angle = patch_angle

        self.logger = LoggerManager.GetLogger(self.__class__.__name__)

        # 预计算用于角度判断的cos值（注: 需将角度阈值从度数转换为余弦值比较）
        self._cos_edge_angle = math.cos(math.radians(self.edge_angle))
        self._cos_patch_angle = math.cos(math.radians(self.patch_angle))

        # -------------------------
        # Monitoring counters (for debugging / logging)
        # -------------------------
        self._mon_bitmap_calls = 0
        self._mon_bitmap_inliers = 0
        self._mon_bitmap_connected = 0
        self._mon_bitmap_leftover = 0

    def _DetectImpl(self, point_cloud: PointCloud) -> List[Discontinuity]:
        """
        功能简介:
            在输入点云上执行 Supervoxel-RegionGrowing 平面检测, 返回一组 Discontinuity 结构面。
        实现思路:
            - 调用 _Voxelize 将点云划分为体素。
            - 在每个体素内执行局部 RANSAC, 生成初始 voxel-patch 和 remain_points。
            - 在 remain_points 中对 non-coplanar voxel 进行多次 RANSAC, 提取 edge-patch 并与邻域平面拼接。
            - 将所有 patch 作为种子, 在其邻域剩余点中执行超体素生长, 形成 supervoxel patch。
            - 对 supervoxel patch 执行基于邻接关系的区域生长, 输出合并后的 Discontinuity。
        输入变量:
            point_cloud: PointCloud
                输入的点云对象, 要求至少包含 points(N,3) 和 normals(N,3)。
        输出变量:
            discontinuities: List[Discontinuity]
                检测到的结构面列表。
        """

        def get_point_cloud_data(point_cloud: PointCloud):
            points = point_cloud.points

            # 将点的坐标和法向提取为 NumPy 数组，方便向量化计算
            coords = np.array([[p.x, p.y, p.z] for p in points], dtype=float)

            # 颜色 (R,G,B) 与曲率（若不存在则 NaN）
            rgb = np.array(
                [[int(getattr(p, "r", 0)), int(getattr(p, "g", 0)), int(getattr(p, "b", 0))] for p in points],
                dtype=np.int32)
            curvature = np.array([
                float(getattr(p, "curvature", float("nan"))) for p in points], dtype=float)

            # 提取法向量，并标记法向是否有效 (非零)
            normals = []
            for p in points:
                nx, ny, nz = p.normal
                # 法向非零则认为有效
                if nx == 0 and ny == 0 and nz == 0:
                    normals.append([0, 0, 0])
                else:
                    # 将法向量归一化以确保角度计算准确
                    norm_len = math.sqrt(nx * nx + ny * ny + nz * nz) + 1e-12
                    normals.append((nx / norm_len, ny / norm_len, nz / norm_len))
            normals = np.array(normals, dtype=float)
            return coords, normals, rgb, curvature

        # -------------------------
        # main
        # -------------------------
        if o3d is None:
            # Open3D 库不可用，无法执行RANSAC平面提取
            self.logger.error("Open3D not found. SupervoxelDetector requires Open3D for plane segmentation.")
            return []

        coords, normals, rgb, curvature = get_point_cloud_data(point_cloud)  # (N, 3)
        num_points = coords.shape[0]

        # -------------------------
        # Debug export config (Step2-5)
        # -------------------------
        debug_dir, debug_base, debug_enabled = self._GetDebugExportConfig()
        normals_valid = ~np.isnan(normals).any(axis=1)

        # -------------------------
        # Monitoring: 有效法向数量和比例
        # -------------------------
        if not _MonCheckInputNormals(self.logger, num_points=num_points, normals_valid=normals_valid):
            return []

        # -------------------------
        # Step 1: 点云体素化
        # -------------------------
        with Timer("Supervoxel: Voxelization", self.logger):
            pts_min = coords.min(axis=0)
            voxel_indices = np.floor((coords - pts_min) / self.voxel_size).astype(int)
            voxel_map: Dict[Tuple[int, int, int], List[int]] = {}
            point_voxel_map: List[Tuple[int, int, int]] = [None] * num_points

            for i, vid in enumerate(map(tuple, voxel_indices)):
                point_voxel_map[i] = vid
                if vid in voxel_map:
                    voxel_map[vid].append(i)
                else:
                    voxel_map[vid] = [i]

        self.logger.info(f"Voxelization: generated {len(voxel_map)} voxels with size {self.voxel_size:.3f} m.")

        # Monitoring: voxel occupancy stats
        _MonLogVoxelOccupancy(self.logger, voxel_map)

        # -------------------------
        # Step 2: 体素内局部 RANSAC 提取 voxel-patch
        # -------------------------
        clusters: List[Dict] = []  # 每个 cluster 是一个初始平面片 (voxel-patch 或 edge-patch)
        remain_points: Dict[Tuple[int, int, int], Set[int]] = {}

        # Debug bookkeeping (Step2): voxel 内 patch vs other 的分布诊断 + voxel 级绘图
        step2_voxel_diag_records: List[Dict] = []
        step2_voxel_csv_paths: List[str] = []
        n_voxel_seeds_step2: int = 0
        with Timer("Supervoxel: Initial voxel-plane extraction", self.logger):
            for voxel_id, pts_indices_list in voxel_map.items():
                pts_indices = np.array(pts_indices_list, dtype=int)
                if pts_indices.size < self.min_plane_points:
                    # 点数不足, 整个体素视为 non-coplanar, 所有点进入 remain_points
                    remain_points[voxel_id] = set(pts_indices.tolist())
                    continue

                # 在该体素内执行 RANSAC 拟合平面
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(coords[pts_indices])
                plane_model, inliers = pcd.segment_plane(
                    distance_threshold=self.ransac_distance,  # todo  这里的阈值越低, 将位于边缘voxel判定为整个平面的概率越低
                    ransac_n=3,
                    num_iterations=1000
                )
                inliers = np.array(inliers, dtype=int)
                if inliers.size == 0:
                    # 未找到平面, 所有点进入 remain_points
                    remain_points[voxel_id] = set(pts_indices.tolist())
                    continue

                # RANSAC 平面参数归一化
                a, b, c, d = plane_model
                norm_len = math.sqrt(a * a + b * b + c * c) + 1e-12
                a, b, c, d = a / norm_len, b / norm_len, c / norm_len, d / norm_len
                plane_normal = np.array([a, b, c], dtype=float)

                # 将局部 inlier 映射回全局索引
                voxel_inliers_global = pts_indices[inliers]

                # # 对 inlier 做 bitmap 最大连通域 + 外包闭合边界筛选(允许空洞), 并用边界内点重估 plane_model
                # connected_inliers_global, leftover_inliers_global, refined_plane_model = self._bitmapMCCfilter(
                #     coords=coords,
                #     inliers_global=voxel_inliers_global,
                #     plane_model=(a, b, c, d),
                #     grid_size=self.sample_spacing
                # )
                # filtered_inliers_global = connected_inliers_global
                # a, b, c, d = refined_plane_model
                # plane_normal = np.array([a, b, c], dtype=float)
                filtered_inliers_global = voxel_inliers_global

                inlier_set = set(filtered_inliers_global.tolist())
                remain_set = set(pts_indices.tolist()) - inlier_set

                # Debug: Step2 voxel 内 patch vs other 的法向/曲率/到平面距离分布诊断（每 voxel 一条）
                if debug_enabled:
                    try:
                        stats_patch = self._ComputeVoxelDistributionStats(
                            coords=coords,
                            normals=normals,
                            curvature=curvature,
                            indices=list(inlier_set),
                            plane_normal=plane_normal,
                            plane_d=float(d),
                            normals_valid=normals_valid
                        )
                        stats_other = self._ComputeVoxelDistributionStats(
                            coords=coords,
                            normals=normals,
                            curvature=curvature,
                            indices=list(remain_set),
                            plane_normal=plane_normal,
                            plane_d=float(d),
                            normals_valid=normals_valid
                        )
                        rec = {
                            "voxel_id": str(voxel_id),
                            "n_total": int(pts_indices.size),
                            "n_patch": int(len(inlier_set)),
                            "n_other": int(len(remain_set)),
                            "plane_a": float(a),
                            "plane_b": float(b),
                            "plane_c": float(c),
                            "plane_d": float(d),
                        }
                        # patch stats
                        for k, v in stats_patch.items():
                            rec[f"patch_{k}"] = v
                        # other stats
                        for k, v in stats_other.items():
                            rec[f"other_{k}"] = v
                        # differences: other - patch
                        for k in stats_patch.keys():
                            pv = stats_patch.get(k, float("nan"))
                            ov = stats_other.get(k, float("nan"))
                            if (pv is None) or (ov is None):
                                rec[f"diff_{k}"] = float("nan")
                            else:
                                try:
                                    rec[f"diff_{k}"] = float(ov) - float(pv)
                                except Exception:
                                    rec[f"diff_{k}"] = float("nan")

                        step2_voxel_diag_records.append(rec)
                    except Exception as _e:
                        pass

                if len(inlier_set) < self.min_plane_points:
                    # 平面内点太少, 整体视为剩余点
                    remain_points[voxel_id] = set(pts_indices.tolist())
                else:
                    clusters.append({
                        "voxel_ids": [voxel_id],
                        "points": set(filtered_inliers_global.tolist()),
                        "orig_points": set(filtered_inliers_global.tolist()),
                        "normal": (float(a), float(b), float(c)),
                        "d": float(d),
                        "seed_type": "voxel"
                    })
                    remain_points[voxel_id] = remain_set

            for vid in voxel_map.keys():
                remain_points.setdefault(vid, set())

        n_voxel_seeds_step2 = int(len(clusters))

        self.logger.info(
            f"Initial voxel-plane extraction: {len(clusters)} plane seeds, "
            f"{sum(len(s) for s in remain_points.values())} remain points."
        )

        # Monitoring: seed cluster size & remain stats
        _MonLogSeedClustersAndRemain(self.logger, clusters=clusters, remain_points=remain_points,
                                     tag="RemainAfterStep2")
        # -------------------------
        # Debug export (Step2): voxel-patch 可视化 / voxel 内分布诊断表 / voxel 级图件
        # -------------------------
        if debug_enabled:
            _DebugExportStep2(
                detector=self,
                debug_dir=debug_dir,
                debug_base=debug_base,
                num_points=num_points,
                coords=coords,
                normals=normals,
                rgb=rgb,
                curvature=curvature,
                clusters=clusters,
                voxel_map=voxel_map,
                step2_voxel_diag_records=step2_voxel_diag_records,
            )

        # -------------------------
        # Step 3: 边缘平面提取 (edge-patch) 与拼接
        # -------------------------
        # -------------------------
        # Debug bookkeeping (Step3): edge-patch 新建/并入 与 阈值拦截记录
        # -------------------------
        edge_patch_new_points: Set[int] = set()
        edge_patch_merged_points: Set[int] = set()
        edge_merge_records: List[Dict] = []
        with Timer("Supervoxel: Edge patch extraction", self.logger):
            new_clusters: List[Dict] = []

            for voxel_id, pts_set in list(remain_points.items()):
                if pts_set is None or len(pts_set) == 0:
                    continue
                if len(pts_set) <= self.min_edge_points:
                    # 剩余点过少, 不做边缘提取
                    continue

                testable = True
                while len(pts_set) > self.min_edge_points and testable:
                    pts_indices = np.array(list(pts_set), dtype=int)
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(coords[pts_indices])
                    plane_model, inliers = pcd.segment_plane(
                        distance_threshold=self.ransac_distance,
                        ransac_n=3,
                        num_iterations=500
                    )
                    inliers = np.array(inliers, dtype=int)
                    if inliers.size == 0:
                        break

                    edge_inliers_global = pts_indices[inliers]

                    '''
                    # 对 edge-patch 做 bitmap 最大连通域 + 外包闭合边界筛选(允许空洞), 并用边界内点重估 plane_model
                    pa, pb, pc, pd = plane_model
                    norm_len = math.sqrt(pa * pa + pb * pb + pc * pc) + 1e-12
                    pa, pb, pc, pd = pa / norm_len, pb / norm_len, pc / norm_len, pd / norm_len
                    patch_normal = np.array([pa, pb, pc], dtype=float)
                    connected_edge_inliers_global, leftover_edge_inliers_global, refined_plane_model = self._bitmapMCCfilter(
                        coords=coords,
                        inliers_global=edge_inliers_global,
                        plane_model=(pa, pb, pc, pd),
                        grid_size=self.sample_spacing
                    )
                    filtered_edge_inliers_global = connected_edge_inliers_global
                    pa, pb, pc, pd = refined_plane_model
                    patch_normal = np.array([pa, pb, pc], dtype=float)
                    inlier_set = set(filtered_edge_inliers_global.tolist())
                    edge_remain_set = pts_set - inlier_set
                    '''

                    pa, pb, pc, pd = plane_model
                    norm_len = math.sqrt(pa * pa + pb * pb + pc * pc) + 1e-12
                    pa, pb, pc, pd = pa / norm_len, pb / norm_len, pc / norm_len, pd / norm_len
                    patch_normal = np.array([pa, pb, pc], dtype=float)
                    inlier_set = set(edge_inliers_global.tolist())
                    edge_remain_set = pts_set - inlier_set

                    if len(inlier_set) > self.min_edge_patch_points:
                        neighbor_voxels = self._get_neighbor_voxels(voxel_id)
                        best_pass_sim = float("inf")
                        best_pass_idx = None

                        best_any_score = float("inf")
                        best_any_idx = None
                        best_any_mw = float("inf")
                        best_any_nw = float("inf")

                        # 在邻域体素中查找已有平面 cluster (记录“最优候选”与“最优可并入候选”)
                        pts_edge = coords[list(inlier_set)]
                        if pts_edge.shape[0] == 0:
                            best_pass_idx = None
                        else:
                            for neighbor_vid in neighbor_voxels:
                                for idx, cluster in enumerate(clusters):
                                    if neighbor_vid not in cluster["voxel_ids"]:
                                        continue

                                    plane_normal = np.array(cluster["normal"], dtype=float)
                                    plane_d = float(cluster["d"])

                                    distances = np.abs(np.dot(pts_edge, plane_normal) + plane_d)
                                    mw = float(distances.mean()) if distances.size > 0 else float("inf")

                                    # 计算 patch 平面与 cluster 平面的法向夹角
                                    cos_angle = float(np.clip(
                                        np.abs(np.dot(patch_normal, plane_normal)), -1.0, 1.0
                                    ))
                                    nw = float(math.degrees(math.acos(cos_angle)))

                                    # 归一化综合指标（即 Pw）
                                    dist_norm = self.edge_distance if self.edge_distance > 0 else 1e-6
                                    ang_norm = self.edge_angle if self.edge_angle > 0 else 1e-6
                                    score = float(math.sqrt((mw / dist_norm) ** 2 + (nw / ang_norm) ** 2))

                                    if score < best_any_score:
                                        best_any_score = score
                                        best_any_idx = idx
                                        best_any_mw = mw
                                        best_any_nw = nw

                                    if (mw < self.edge_distance) and (nw < self.edge_angle):
                                        if score < best_pass_sim:
                                            best_pass_sim = score
                                            best_pass_idx = idx

                        merged = best_pass_idx is not None
                        merge_reason = ""
                        if merged:
                            # 拼接到已有平面 cluster
                            clusters[best_pass_idx]["points"].update(inlier_set)
                            # 关键修正: 需要同步更新 orig_points，否则 Step4 的 seed 初始化会忽略这些并入点
                            clusters[best_pass_idx]["orig_points"].update(inlier_set)
                            if voxel_id not in clusters[best_pass_idx]["voxel_ids"]:
                                clusters[best_pass_idx]["voxel_ids"].append(voxel_id)

                            edge_patch_merged_points.update(inlier_set)
                            merge_reason = "merged"
                        else:
                            # 新建一个 edge-patch cluster
                            new_clusters.append({
                                "voxel_ids": [voxel_id],
                                "points": set(inlier_set),
                                "orig_points": set(inlier_set),
                                "normal": (float(pa), float(pb), float(pc)),
                                "d": float(pd),
                                "seed_type": "edge",
                            })
                            edge_patch_new_points.update(inlier_set)

                            if best_any_idx is None:
                                merge_reason = "no_neighbor_cluster"
                            else:
                                dist_fail = (best_any_mw >= float(self.edge_distance))
                                ang_fail = (best_any_nw >= float(self.edge_angle))
                                if dist_fail and ang_fail:
                                    merge_reason = "blocked_by_distance_and_angle"
                                elif dist_fail:
                                    merge_reason = "blocked_by_edge_distance"
                                elif ang_fail:
                                    merge_reason = "blocked_by_edge_angle"
                                else:
                                    merge_reason = "blocked_unknown"

                        # 记录 Step3 的“为什么没拼接”信息（每个 edge-patch 一条）
                        try:
                            best_candidate_voxel_id = ""
                            if best_any_idx is not None and 0 <= best_any_idx < len(clusters):
                                cand_voxels = clusters[best_any_idx].get("voxel_ids", None)
                                if cand_voxels and len(cand_voxels) > 0:
                                    best_candidate_voxel_id = str(cand_voxels[0])
                            edge_merge_records.append({
                                "voxel_id": str(voxel_id),
                                "edge_patch_points": int(len(inlier_set)),
                                "merged": int(1 if merged else 0),
                                "merge_reason": merge_reason,
                                "best_candidate_cluster_idx": int(best_any_idx) if best_any_idx is not None else -1,
                                "best_candidate_voxel_id": best_candidate_voxel_id,
                                "best_mw": float(best_any_mw) if np.isfinite(best_any_mw) else float("inf"),
                                "best_nw": float(best_any_nw) if np.isfinite(best_any_nw) else float("inf"),
                                "best_score_Pw": float(best_any_score) if np.isfinite(best_any_score) else float("inf"),
                                "pass_score_Pw": float(best_pass_sim) if np.isfinite(best_pass_sim) else float("inf"),
                                "best_pass_cluster_idx": int(best_pass_idx) if best_pass_idx is not None else -1,
                                "pass_edge_distance": int(1 if (np.isfinite(best_any_mw) and (
                                        best_any_mw < float(self.edge_distance))) else 0),
                                "pass_edge_angle": int(
                                    1 if (np.isfinite(best_any_nw) and (best_any_nw < float(self.edge_angle))) else 0),
                                "edge_distance": float(self.edge_distance),
                                "edge_angle": float(self.edge_angle),
                            })
                        except Exception:
                            pass
                        # 更新该体素剩余点
                        pts_set = edge_remain_set
                    else:
                        # patch 点太少, 认为无法成面, 停止对该 voxel 的边缘提取
                        testable = False

                remain_points[voxel_id] = pts_set

            clusters.extend(new_clusters)

        self.logger.info(f"Edge patch extraction: clusters extended to {len(clusters)} seeds.")

        # Monitoring: remain stats after edge extraction
        _MonLogSeedClustersAndRemain(self.logger, clusters=[], remain_points=remain_points, tag="RemainAfterStep3")

        # -------------------------

        # Debug export (Step3): edge-patch 可视化 + “为什么没拼接”阈值诊断表
        # -------------------------
        if debug_enabled:
            _DebugExportStep3(
                detector=self,
                debug_dir=debug_dir,
                debug_base=debug_base,
                num_points=num_points,
                coords=coords,
                normals=normals,
                rgb=rgb,
                curvature=curvature,
                clusters=clusters,
                n_voxel_seeds_step2=n_voxel_seeds_step2,
                edge_patch_new_points=edge_patch_new_points,
                edge_patch_merged_points=edge_patch_merged_points,
                edge_merge_records=edge_merge_records,
            )

        # -------------------------
        # Step 4: 超体素分割 (Supervoxel segmentation)
        # -------------------------
        supervoxels: List[Dict] = []

        # Debug bookkeeping (Step4): 记录“被吸纳点”与每个 seed 的阈值收缩过程摘要
        absorbed_points_total: Set[int] = set()
        absorbed_sets_with_color: List[Tuple[Set[int], Tuple[int, int, int]]] = []
        supervoxel_growth_records: List[Dict] = []

        # Monitoring: supervoxel growth summary (aggregate)
        mon_sv_seeds = len(clusters)
        mon_sv_iters_total = 0
        mon_sv_absorb_total = 0
        mon_sv_best_pts_total = 0
        mon_sv_success = 0
        mon_verbose = os.environ.get("SUPERVOXEL_MON_VERBOSE", "0").strip() in ("1", "true", "True")

        with Timer("Supervoxel: Supervoxel segmentation", self.logger):
            for ci, cluster in enumerate(clusters):
                record_normal = np.array(cluster["normal"], dtype=float)
                record_d = float(cluster["d"])

                # Monitoring: per-seed stats
                mon_seed_iters = 0
                mon_seed_absorbed = 0
                mon_seed_cand = 0

                # 该 patch 覆盖的所有体素
                seed_voxels: Set[Tuple[int, int, int]] = set(cluster["voxel_ids"])

                # 构建候选体素区域: seed_voxels + 每个 seed_voxel 的 26 邻域
                neighbor_voxels: Set[Tuple[int, int, int]] = set()
                for sv in seed_voxels:
                    neighbor_voxels.add(sv)
                    neighbor_voxels.update(self._get_neighbor_voxels(sv))

                neighbor_remains: Dict[Tuple[int, int, int], np.ndarray] = {}
                for nv in neighbor_voxels:
                    pts_set = remain_points.get(nv)
                    if pts_set:
                        neighbor_remains[nv] = np.array(list(pts_set), dtype=int)
                    else:
                        neighbor_remains[nv] = np.array([], dtype=int)

                dist_th = self.super_distance
                ang_th = self.super_angle
                orientation_diff = float("inf")
                best_points_set: Set[int] = set()
                best_normal = record_normal.copy()
                best_d = record_d

                seed_points_base: Set[int] = set(cluster.get("points", set(cluster.get("orig_points", []))))

                with Timer(f"Supervoxel growth for seed {ci}", self.logger):
                    while orientation_diff > self.max_refit_error and (dist_th > 0 or ang_th > 0):

                        mon_seed_iters += 1
                        seed_points: Set[int] = set(seed_points_base)
                        current_points: Set[int] = set(seed_points)
                        current_normal = record_normal.copy()
                        current_d = record_d
                        cos_ang_th = math.cos(math.radians(ang_th))

                        # 从候选区域中吸收符合当前阈值的 remain 点
                        for nv, nv_indices in neighbor_remains.items():
                            if nv_indices.size == 0:
                                continue

                            mon_seed_cand += int(nv_indices.size)

                            dists = np.abs(np.dot(coords[nv_indices], current_normal) + current_d)

                            valid_mask = normals_valid[nv_indices]
                            if not np.any(valid_mask):
                                continue
                            cosines = np.abs(np.dot(normals[nv_indices], current_normal))

                            mask = (dists < dist_th) & (cosines >= cos_ang_th) & valid_mask
                            if np.any(mask):
                                selected_indices = nv_indices[mask]
                                current_points.update(selected_indices.tolist())

                                mon_seed_absorbed += int(selected_indices.size)

                        if len(current_points) < 3:
                            break

                        pts_array = coords[list(current_points)]
                        centroid = pts_array.mean(axis=0)
                        cov = np.cov(pts_array - centroid, rowvar=False)
                        eig_vals, eig_vecs = np.linalg.eigh(cov)
                        new_normal = eig_vecs[:, int(np.argmin(eig_vals))]

                        if new_normal[2] < 0:
                            new_normal = -new_normal
                        new_d = -float(new_normal.dot(centroid))

                        cos_angle = float(np.clip(
                            np.abs(new_normal.dot(record_normal)), -1.0, 1.0
                        ))
                        orientation_diff = math.degrees(math.acos(cos_angle))

                        if orientation_diff <= self.max_refit_error:
                            best_points_set = set(current_points)
                            best_normal = new_normal
                            best_d = new_d
                            break

                        dist_th -= self.distance_step
                        ang_th -= self.angle_step
                        if dist_th < 0:
                            dist_th = 0.0
                        if ang_th < 0:
                            ang_th = 0.0

                # 构建 supervoxel patch 结果
                orientation_diff_last = float(orientation_diff)
                fallback_used = False
                fallback_reason = ""
                if len(best_points_set) == 0:
                    # 未能在阈值收缩过程中达到 max_refit_error（或中途点数不足），沿用 seed 自身点集
                    best_points_set = set(cluster["points"])
                    best_normal = record_normal
                    best_d = record_d
                    fallback_used = True
                    if not np.isfinite(orientation_diff_last):
                        fallback_reason = "no_valid_refit"
                    else:
                        fallback_reason = "blocked_by_max_refit_error"
                    # 注意：为保持原算法行为，这里仍将 orientation_diff 置为 0（仅用于后续排序/输出的 patch error）。
                    orientation_diff = 0.0

                # Debug: 统计本 seed 的“新增吸纳点”（用于 Step4 可视化）
                try:
                    absorbed_this_seed = set(best_points_set) - set(seed_points_base)
                    absorbed_points_total.update(absorbed_this_seed)
                    if len(absorbed_this_seed) > 0:
                        absorbed_sets_with_color.append((set(absorbed_this_seed), GenerateDebugColorFromId(ci)))
                except Exception:
                    absorbed_this_seed = set()

                # Monitoring: aggregate update
                mon_sv_iters_total += int(mon_seed_iters)
                mon_sv_absorb_total += int(mon_seed_absorbed)
                mon_sv_best_pts_total += int(len(best_points_set))
                if (('fallback_used' in locals()) and (not fallback_used) and (
                        orientation_diff <= self.max_refit_error)):
                    mon_sv_success += 1
                if mon_verbose:
                    self.logger.info(
                        f"[MON] SeedGrow ci={ci}: iters={mon_seed_iters}, candPts={mon_seed_cand}, "
                        f"absorbedPts={mon_seed_absorbed}, bestPts={len(best_points_set)}, "
                        f"finalDistTh={dist_th:.3f}, finalAngTh={ang_th:.1f}, orientDiff={orientation_diff:.2f}"
                    )

                # Debug: 记录阈值收缩/成功情况，便于定位“没被吸纳/没拼接”的拦截阈值
                try:
                    supervoxel_growth_records.append({
                        "seed_idx": int(ci),
                        "seed_type": str(cluster.get("seed_type", "")),
                        "seed_points": int(len(seed_points_base)),
                        "best_points": int(len(best_points_set)),
                        "absorbed_points": int(len(absorbed_this_seed)) if 'absorbed_this_seed' in locals() else 0,
                        "cand_points_seen": int(mon_seed_cand),
                        "iters": int(mon_seed_iters),
                        "init_dist_th": float(self.super_distance),
                        "init_ang_th": float(self.super_angle),
                        "distance_step": float(self.distance_step),
                        "angle_step": float(self.angle_step),
                        "final_dist_th": float(dist_th),
                        "final_ang_th": float(ang_th),
                        "orient_diff": float(orientation_diff),
                        "orient_diff_last": float(
                            orientation_diff_last) if 'orientation_diff_last' in locals() else float("inf"),
                        "max_refit_error": float(self.max_refit_error),
                        "fallback_used": int(1 if ('fallback_used' in locals() and fallback_used) else 0),
                        "fallback_reason": str(fallback_reason) if 'fallback_reason' in locals() else "",
                        "blocked_by_max_refit_error": int(1 if (
                                'fallback_used' in locals() and fallback_used and np.isfinite(
                            orientation_diff_last) and (orientation_diff_last > self.max_refit_error)) else 0),
                        "success": int(1 if (('fallback_used' in locals()) and (not fallback_used) and (
                                orientation_diff <= self.max_refit_error)) else 0),
                    })
                except Exception:
                    pass

                # 根据最终点集重新统计体素覆盖范围
                best_voxel_ids: Set[Tuple[int, int, int]] = set()
                for pid in best_points_set:
                    vid = point_voxel_map[pid]
                    best_voxel_ids.add(vid)
                    # 从 remain_points 中移除这些点
                    if vid in remain_points and pid in remain_points[vid]:
                        remain_points[vid].discard(pid)

                supervoxels.append({
                    "points": best_points_set,
                    "voxel_ids": best_voxel_ids,
                    "normal": tuple(best_normal.tolist()),
                    "d": float(best_d),
                    "error": float(orientation_diff),
                })

        self.logger.info(f"Supervoxel segmentation: generated {len(supervoxels)} supervoxel patches.")

        # Monitoring: supervoxel summary & remain stats
        _MonLogSupervoxels(self.logger, supervoxels=supervoxels, remain_points=remain_points,
                           absorbed_points_total=absorbed_points_total)

        # Debug export (Step4): 多方案可视化 + seed 生长阈值摘要表
        # -------------------------
        if debug_enabled:
            _DebugExportStep4(
                detector=self,
                debug_dir=debug_dir,
                debug_base=debug_base,
                num_points=num_points,
                coords=coords,
                normals=normals,
                rgb=rgb,
                curvature=curvature,
                clusters=clusters,
                n_voxel_seeds_step2=n_voxel_seeds_step2,
                edge_patch_new_points=edge_patch_new_points,
                edge_patch_merged_points=edge_patch_merged_points,
                absorbed_sets_with_color=absorbed_sets_with_color,
                absorbed_points_total=absorbed_points_total,
                supervoxels=supervoxels,
                supervoxel_growth_records=supervoxel_growth_records,
            )

        _MonLogSuperGrowAggAndBitmap(
            logger=self.logger,
            mon_sv_seeds=mon_sv_seeds,
            mon_sv_success=mon_sv_success,
            mon_sv_iters_total=mon_sv_iters_total,
            mon_sv_absorb_total=mon_sv_absorb_total,
            mon_sv_best_pts_total=mon_sv_best_pts_total,
            mon_bitmap_calls=int(getattr(self, "_mon_bitmap_calls", 0)),
            mon_bitmap_inliers=int(getattr(self, "_mon_bitmap_inliers", 0)),
            mon_bitmap_connected=int(getattr(self, "_mon_bitmap_connected", 0)),
            mon_bitmap_leftover=int(getattr(self, "_mon_bitmap_leftover", 0)),
        )

        # -------------------------
        # Step 5: Patch-based 区域生长
        # -------------------------
        discontinuities: List[Discontinuity] = []
        patch_merge_records: List[Dict] = []
        with Timer("Supervoxel: Patch-based region growing", self.logger):
            n_patches = len(supervoxels)
            if n_patches == 0:
                return []

            patch_neighbors: Dict[int, Set[int]] = {i: set() for i in range(n_patches)}
            voxel_to_patches: Dict[Tuple[int, int, int], List[int]] = {}

            # 体素到 patch 的映射
            for i, patch in enumerate(supervoxels):
                for vid in patch["voxel_ids"]:
                    voxel_to_patches.setdefault(vid, []).append(i)

            # 建立 patch 邻接关系
            for vid, plist in voxel_to_patches.items():
                # 同体素内 patch 互为邻居
                for p in plist:
                    for q in plist:
                        if p != q:
                            patch_neighbors[p].add(q)
                # 邻域体素内的 patch 互为邻居
                for nbr in self._get_neighbor_voxels(vid):
                    if nbr not in voxel_to_patches:
                        continue
                    for p in plist:
                        for q in voxel_to_patches[nbr]:
                            if p != q:
                                patch_neighbors[p].add(q)

            unmerged: Set[int] = set(range(n_patches))
            patch_order = sorted(unmerged, key=lambda i: supervoxels[i]["error"])

            unmerged: Set[int] = set(range(n_patches))
            patch_order = sorted(unmerged, key=lambda i: supervoxels[i]["error"])

            # 关键修正(防误合并): 仅用“质心到平面距离”无法约束面内(切向)分离，
            # 这里增加一个温和的空间邻近约束，避免“同一平面但相距很远”的片段被直接合并。
            # 阈值取 max(3*voxel_size, 2*patch_distance)，既兼容 voxel_size=1.5m 的工程设置，
            # 也不让 patch_distance 很小时阈值过于苛刻。
            patch_centroid_th = max(self.voxel_size * 3.0, self.patch_distance * 2.0)

            while patch_order:
                seed_id = patch_order.pop(0)
                if seed_id not in unmerged:
                    continue

                # -------------------------
                # 关键修正1：seed 一旦作为当前生长簇的起点，必须立即从 unmerged 中移除。
                # 否则会出现：
                #   (a) seed 被后续其他 seed 二次吸收，导致跨簇“串联”远距离误合并；
                #   (b) seed 被加入 neighbor_list 产生 self-merge (seed==neighbor) 的错误记录。
                # -------------------------
                unmerged.discard(seed_id)

                seed_patch = supervoxels[seed_id]
                seed_normal = np.array(seed_patch["normal"], dtype=float)
                seed_d = float(seed_patch["d"])

                neighbor_set = {nid for nid in patch_neighbors[seed_id] if (nid in unmerged and nid != seed_id)}
                neighbor_list = sorted(list(neighbor_set), key=lambda j: supervoxels[j]["error"])

                merged_patch_ids = [seed_id]

                while neighbor_list:
                    j = neighbor_list.pop(0)
                    if j not in unmerged:
                        continue
                    # 关键修正2：严禁 self-merge / 重复 merge
                    if (j == seed_id) or (j in merged_patch_ids):
                        continue

                    j_patch = supervoxels[j]
                    j_normal = np.array(j_patch["normal"], dtype=float)

                    # 法向夹角
                    cos_angle = float(np.clip(np.abs(seed_normal.dot(j_normal)), -1.0, 1.0))
                    ang_diff = math.degrees(math.acos(cos_angle))

                    # 距离差: 邻居 patch 质心到 seed 平面的距离 (法向方向)
                    j_points = coords[list(j_patch["points"])]
                    if j_points.size > 0:
                        j_centroid = j_points.mean(axis=0)
                        dist_diff = abs(seed_normal.dot(j_centroid) + seed_d)
                    else:
                        j_centroid = np.array([0.0, 0.0, 0.0], dtype=float)
                        dist_diff = 0.0

                    # 额外的空间邻近约束：两片段质心欧氏距离
                    seed_points_now = coords[list(seed_patch["points"])]
                    seed_centroid = seed_points_now.mean(axis=0) if seed_points_now.size > 0 else j_centroid
                    centroid_diff = float(np.linalg.norm(seed_centroid - j_centroid))

                    pass_ang = bool(ang_diff < self.patch_angle)
                    pass_dist = bool(dist_diff < self.patch_distance)
                    pass_centroid = bool(centroid_diff < patch_centroid_th)

                    # Debug: Patch-based 区域生长阈值诊断（到底被哪个阈值拦截）
                    if debug_enabled:
                        try:
                            if pass_ang and pass_dist and pass_centroid:
                                reason = "merged"
                            else:
                                blocks = []
                                if not pass_ang:
                                    blocks.append("patch_angle")
                                if not pass_dist:
                                    blocks.append("patch_distance")
                                if not pass_centroid:
                                    blocks.append("patch_centroid")
                                reason = "blocked_by_" + "_and_".join(blocks) if blocks else "blocked_unknown"
                            patch_merge_records.append({
                                "seed_voxel_id": seed_id,
                                "neighbor_voxel_id": j,
                                "ang_diff": float(ang_diff),
                                "dist_diff": float(dist_diff),
                                "pass_patch_angle": int(1 if pass_ang else 0),
                                "pass_patch_distance": int(1 if pass_dist else 0),
                                # 新增(不破坏旧字段): 面内距离约束诊断
                                "centroid_diff": float(centroid_diff),
                                "pass_patch_centroid": int(1 if pass_centroid else 0),
                                "patch_centroid_th": float(patch_centroid_th),
                                "patch_angle": float(self.patch_angle),
                                "patch_distance": float(self.patch_distance),
                                "decision": reason,
                            })
                        except Exception:
                            pass

                    if pass_ang and pass_dist and pass_centroid:
                        # 合并 patch j 至 seed
                        unmerged.discard(j)
                        merged_patch_ids.append(j)

                        seed_patch["points"].update(j_patch["points"])
                        seed_patch["voxel_ids"].update(j_patch["voxel_ids"])

                        # 重新拟合 seed_patch 平面
                        pts_all = coords[list(seed_patch["points"])]
                        centroid_all = pts_all.mean(axis=0)
                        cov_all = np.cov(pts_all - centroid_all, rowvar=False)
                        eig_vals_all, eig_vecs_all = np.linalg.eigh(cov_all)
                        normal_all = eig_vecs_all[:, int(np.argmin(eig_vals_all))]
                        if normal_all[2] < 0:
                            normal_all = -normal_all
                        seed_normal = normal_all
                        seed_d = -float(normal_all.dot(centroid_all))
                        seed_patch["normal"] = tuple(seed_normal.tolist())
                        seed_patch["d"] = seed_d

                        # 将 j 的邻居加入待检查队列
                        for k in patch_neighbors[j]:
                            if (k == seed_id) or (k in merged_patch_ids):
                                continue
                            if k in unmerged and k not in neighbor_set:
                                neighbor_set.add(k)
                                neighbor_list.append(k)
                        neighbor_list = sorted(neighbor_list, key=lambda x: supervoxels[x]["error"])
                # 生成最终平面 Discontinuity
                # 收集合并的所有片段的点索引
                all_points = set()
                segments_list: List[Segment] = []
                for pid in merged_patch_ids:
                    all_points |= supervoxels[pid]["points"]
                    # 为每个片段创建 Segment 对象 (使用合并前各自的平面参数)
                    plane_normal = supervoxels[pid]["normal"]
                    plane_d = supervoxels[pid]["d"]
                    # 计算片段平面的质心和RMSE
                    pts = coords[list(supervoxels[pid]["points"])]
                    centroid = pts.mean(axis=0) if pts.shape[0] > 0 else np.array([0, 0, 0], dtype=float)
                    # 计算RMSE: 点到该片段平面的平均距离
                    distances = np.abs(np.dot(pts, plane_normal) + plane_d)
                    rmse = math.sqrt((distances ** 2).mean()) if distances.size > 0 else 0.0
                    plane_obj = Plane(plane_normal, plane_d, tuple(centroid.tolist()),
                                      inlier_indices=list(supervoxels[pid]["points"]), rmse=rmse)
                    segments_list.append(Segment(plane_obj, list(supervoxels[pid]["points"])))

                # 统一整个平面的平面参数：对all_points拟合
                pts_all = coords[list(all_points)]
                centroid_all = pts_all.mean(axis=0)
                cov_all = np.cov(pts_all - centroid_all, rowvar=False) if pts_all.shape[0] >= 3 else None
                if cov_all is not None:
                    eig_vals_all, eig_vecs_all = np.linalg.eigh(cov_all)
                    plane_normal = eig_vecs_all[:, np.argmin(eig_vals_all)]
                else:
                    # 点数过少无法拟合平面，用第一个segment的平面
                    plane_normal = np.array(segments_list[0].plane.normal)

                # 假设 normals 有效
                patch_normals = normals[list(all_points)]
                valid_mask = ~np.isnan(patch_normals).any(axis=1)
                if np.any(valid_mask):
                    mean_normal = patch_normals[valid_mask].mean(axis=0)
                    if mean_normal.dot(plane_normal) < 0:
                        plane_normal = -plane_normal

                d_val = -float(plane_normal.dot(centroid_all))
                # 计算整体平面的 dip 和 dip_direction
                norm_len = math.sqrt(plane_normal.dot(plane_normal)) + 1e-12
                plane_norm_unit = plane_normal / norm_len
                nx, ny, nz = plane_norm_unit.tolist()
                # # 将法向统一到上半球
                # if nz < 0:
                #     nx, ny, nz = -nx, -ny, -nz
                horizontal = math.sqrt(nx * nx + ny * ny)
                dip = math.degrees(math.atan2(horizontal, abs(nz)))
                azimuth = math.degrees(math.atan2(nx, ny))
                if azimuth < 0:
                    azimuth += 360.0
                dip_dir = azimuth
                # 计算整体平面粗糙度 (RMSE)
                distances_all = np.abs(np.dot(pts_all, [nx, ny, nz]) + d_val)
                overall_rmse = math.sqrt((distances_all ** 2).mean()) if distances_all.size > 0 else 0.0
                plane_obj = Plane((nx, ny, nz), d_val, tuple(centroid_all.tolist()), inlier_indices=list(all_points),
                                  rmse=overall_rmse)
                # 构造 Discontinuity 对象
                discontinuity = Discontinuity(segments_list, plane_obj, dip, dip_dir, roughness=overall_rmse,
                                              algorithm_name=self.name)
                discontinuities.append(discontinuity)

        self.logger.info(f"Patch-based region growing complete: {len(discontinuities)} planes detected.")

        # Monitoring: plane size stats & final unassigned points
        _MonLogPlaneSizesAndUnassigned(self.logger, discontinuities=discontinuities, remain_points=remain_points)
        # -------------------------
        # Debug export (Step5): Patch-based 区域生长阈值诊断表
        # -------------------------
        if debug_enabled:
            _DebugExportStep5(
                self,
                debug_dir=debug_dir,
                debug_base=debug_base,
                patch_merge_records=patch_merge_records,
            )
        return discontinuities

    def _get_neighbor_voxels(self, voxel_id: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """
        功能简介:
            给定体素索引 (vx, vy, vz), 返回其 3x3x3 邻域中除自身外的 26 个邻居体素索引。
        实现思路:
            - 遍历 dx, dy, dz ∈ {-1, 0, 1} 的所有组合。
            - 排除 (0, 0, 0), 其余 (vx+dx, vy+dy, vz+dz) 视为邻居。
        输入变量:
            voxel_id: Tuple[int, int, int]
                体素索引 (vx, vy, vz)。
        输出变量:
            neighbors: List[Tuple[int, int, int]]
                邻域体素索引列表。
        """
        vx, vy, vz = voxel_id
        neighbors: List[Tuple[int, int, int]] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    neighbors.append((vx + dx, vy + dy, vz + dz))
        return neighbors

    def _bitmapMCCfilter(
            self,
            coords: np.ndarray,
            inliers_global: np.ndarray,
            plane_model: Tuple[float, float, float, float],
            grid_size: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float, float]]:
        """
        功能简介:
            在 pcd.segment_plane 得到 (plane_model, inliers) 之后, 按如下逻辑筛选连通的平面 patch 内点:
                1) 将 inliers 投影到 plane_model 平面上
                2) 以点云采样间隔 s 作为 grid size 构建 bitmap grids
                3) 在 bitmap 上找到最大连通域 C (允许 C 内部存在空洞)
                4) 求 C 的外包闭合边界(二维外包边界), 将边界内的 inliers 作为最终 connected inliers
                5) 用 connected inliers 重新估计 plane_model 作为该 patch 的平面参数
            其余的 inliers 作为 leftover_inliers 返回, 由调用方归入 remain_set 参与后续计算。

        实现思路(详细):
            - 平面投影:
                p_proj = p - (n·p + d) * n, 其中 n 为单位法向
            - 平面局部坐标系:
                在平面上构造正交基(u,v), 将投影点转为2D坐标(U,V)
            - bitmap:
                将(U,V)按 grid_size 量化为整型格网索引(ix,iy), 占据格网置1
            - 最大连通域:
                对占据格网做8邻域 BFS/DFS 找最大连通域
            - 外包闭合边界:
                对最大连通域格网中心点求二维凸包(外包边界), 该步骤天然允许内部存在空洞
            - 边界内点:
                用点在多边形内(point-in-polygon)判断将所有 inliers 归入边界内/外
            - 平面重估:
                对边界内点做PCA最小二乘拟合, 输出单位法向平面参数

        输入变量及类型:
            coords: np.ndarray, shape=(N,3)
                全局点坐标
            inliers_global: np.ndarray, shape=(K,)
                本次RANSAC inliers 对应的全局索引
            plane_model: Tuple[float,float,float,float]
                本次RANSAC输出平面参数 (a,b,c,d), 满足 ax+by+cz+d=0 (不保证归一化)
            grid_size: Optional[float]
                bitmap 网格大小; 若为 None 则使用 self.sample_spacing

        输出变量及类型:
            connected_inliers_global: np.ndarray, shape=(Kc,)
                外包闭合边界内的 inliers(全局索引)
            leftover_inliers_global: np.ndarray, shape=(Kl,)
                外包闭合边界外的原 inliers(全局索引), 应回收进 remain
            refined_plane_model: Tuple[float,float,float,float]
                用 connected_inliers 重新估计得到的平面参数 (a,b,c,d), 单位法向
        """
        with Timer("Supervoxel: _bitmapMCCfilter", self.logger):
            t0 = time.perf_counter()

            # -------------------------
            # 0. 参数与输入检查
            # -------------------------
            if grid_size is None:
                grid_size = self.sample_spacing
            if grid_size <= 0:
                raise ValueError(f"grid_size 必须为正数, 当前={grid_size}")

            if inliers_global is None:
                return np.array([], dtype=int), np.array([], dtype=int), plane_model
            inliers_global = np.asarray(inliers_global, dtype=int)
            if inliers_global.size == 0:
                return np.array([], dtype=int), np.array([], dtype=int), plane_model

            pts = coords[inliers_global]
            if pts.ndim != 2 or pts.shape[1] != 3:
                raise ValueError("coords 必须为 shape=(N,3) 的数组")

            # -------------------------
            # 1. 平面归一化 + 点投影
            # -------------------------
            a, b, c, d = plane_model
            n = np.array([a, b, c], dtype=float)
            n_norm = float(np.linalg.norm(n) + 1e-12)
            n = n / n_norm
            d = float(d / n_norm)

            # p_proj = p - (n·p + d) * n
            signed_dist = np.dot(pts, n) + d
            pts_proj = pts - signed_dist[:, None] * n[None, :]

            # -------------------------
            # 2. 构建平面局部2D坐标系(u,v), 计算(U,V)
            # -------------------------
            # 选取不与 n 平行的参考向量
            if abs(n[2]) < 0.9:
                ref = np.array([0.0, 0.0, 1.0], dtype=float)
            else:
                ref = np.array([0.0, 1.0, 0.0], dtype=float)

            u = np.cross(ref, n)
            u = u / (np.linalg.norm(u) + 1e-12)
            v = np.cross(n, u)  # 已单位化

            origin = pts_proj.mean(axis=0)
            rel = pts_proj - origin[None, :]
            uv = np.empty((pts_proj.shape[0], 2), dtype=float)
            uv[:, 0] = np.dot(rel, u)
            uv[:, 1] = np.dot(rel, v)

            # -------------------------
            # 3. bitmap 栅格化
            # -------------------------
            u_min = float(uv[:, 0].min())
            v_min = float(uv[:, 1].min())

            ix = np.floor((uv[:, 0] - u_min) / grid_size).astype(np.int64)
            iy = np.floor((uv[:, 1] - v_min) / grid_size).astype(np.int64)

            occupied = set(zip(ix.tolist(), iy.tolist()))
            if len(occupied) == 0:
                refined = (float(n[0]), float(n[1]), float(n[2]), float(d))
                return inliers_global.copy(), np.array([], dtype=int), refined

            # -------------------------
            # 4. 最大连通域(MCC)提取: 8邻域
            # -------------------------
            nbr8 = [(-1, -1), (-1, 0), (-1, 1),
                    (0, -1), (0, 1),
                    (1, -1), (1, 0), (1, 1)]
            visited = set()
            mcc_cells = []
            mcc_size = 0

            for cell in occupied:
                if cell in visited:
                    continue
                stack = [cell]
                visited.add(cell)
                comp = [cell]

                while stack:
                    cx, cy = stack.pop()
                    for dx, dy in nbr8:
                        nb = (cx + dx, cy + dy)
                        if nb in occupied and nb not in visited:
                            visited.add(nb)
                            stack.append(nb)
                            comp.append(nb)

                if len(comp) > mcc_size:
                    mcc_size = len(comp)
                    mcc_cells = comp

            if mcc_size < 3:
                refined = (float(n[0]), float(n[1]), float(n[2]), float(d))
                return inliers_global.copy(), np.array([], dtype=int), refined

            # -------------------------
            # 5. 外包闭合边界: MCC 栅格中心点二维凸包(允许空洞)
            # -------------------------
            mcc_uv = np.empty((len(mcc_cells), 2), dtype=float)
            for i, (cx, cy) in enumerate(mcc_cells):
                mcc_uv[i, 0] = u_min + (float(cx) + 0.5) * grid_size
                mcc_uv[i, 1] = v_min + (float(cy) + 0.5) * grid_size

            # Andrew monotone chain convex hull (无额外依赖)
            def _cross(o, a, b) -> float:
                return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

            pts2d = sorted({(float(p[0]), float(p[1])) for p in mcc_uv})
            if len(pts2d) < 3:
                refined = (float(n[0]), float(n[1]), float(n[2]), float(d))
                return inliers_global.copy(), np.array([], dtype=int), refined

            lower = []
            for p in pts2d:
                while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
                    lower.pop()
                lower.append(p)

            upper = []
            for p in reversed(pts2d):
                while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
                    upper.pop()
                upper.append(p)

            hull = lower[:-1] + upper[:-1]  # CCW
            hull_np = np.asarray(hull, dtype=float)

            # -------------------------
            # 6. 点在多边形内判断: hull 边界内的 inliers 作为 connected_inliers
            # -------------------------
            def _point_in_poly(x: float, y: float, poly: np.ndarray) -> bool:
                inside = False
                nverts = poly.shape[0]
                j = nverts - 1
                for i in range(nverts):
                    xi, yi = poly[i, 0], poly[i, 1]
                    xj, yj = poly[j, 0], poly[j, 1]

                    # 边界容差判断: 点落在边界上视为 inside
                    dx1, dy1 = x - xi, y - yi
                    dx2, dy2 = xj - xi, yj - yi
                    crossp = dx1 * dy2 - dy1 * dx2
                    if abs(crossp) < 1e-10:
                        dotp = dx1 * (x - xj) + dy1 * (y - yj)
                        if dotp <= 1e-10:
                            return True

                    # Ray casting
                    intersect = ((yi > y) != (yj > y)) and (
                            x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi
                    )
                    if intersect:
                        inside = not inside
                    j = i
                return inside

            inside_mask = np.zeros((uv.shape[0],), dtype=bool)
            for i in range(uv.shape[0]):
                inside_mask[i] = _point_in_poly(float(uv[i, 0]), float(uv[i, 1]), hull_np)

            connected_inliers_global = inliers_global[inside_mask]
            leftover_inliers_global = inliers_global[~inside_mask]

            if connected_inliers_global.size < 3:
                refined = (float(n[0]), float(n[1]), float(n[2]), float(d))
                return inliers_global.copy(), np.array([], dtype=int), refined

            # -------------------------
            # 7. 用 connected_inliers 重估平面参数 (PCA最小二乘)
            # -------------------------
            pts_conn = coords[connected_inliers_global]
            centroid = pts_conn.mean(axis=0)
            X = pts_conn - centroid[None, :]
            cov = (X.T @ X) / float(max(pts_conn.shape[0], 1))

            eigvals, eigvecs = np.linalg.eigh(cov)
            normal_refit = eigvecs[:, 0]  # 最小特征值对应法向
            normal_refit = normal_refit / (np.linalg.norm(normal_refit) + 1e-12)

            # 与原法向一致性
            if float(np.dot(normal_refit, n)) < 0:
                normal_refit = -normal_refit

            d_refit = -float(np.dot(normal_refit, centroid))
            refined_plane_model = (
                float(normal_refit[0]), float(normal_refit[1]), float(normal_refit[2]), float(d_refit))

            # -------------------------
            # 8. log / 耗时 / 可视化(可选, 无额外依赖)
            # -------------------------
            t1 = time.perf_counter()
            self.logger.debug(
                f"_bitmapMCCfilter: inliers={inliers_global.size}, occupied={len(occupied)}, "
                f"MCC={mcc_size}, hull_verts={len(hull)}, connected={connected_inliers_global.size}, "
                f"leftover={leftover_inliers_global.size}, grid_size={grid_size} | "
                f"耗时 {(t1 - t0):.4f}s"
            )

            # 累计监测计数（不影响算法逻辑）
            try:
                self._mon_bitmap_calls += 1
                self._mon_bitmap_inliers += int(inliers_global.size)
                self._mon_bitmap_connected += int(connected_inliers_global.size)
                self._mon_bitmap_leftover += int(leftover_inliers_global.size)
            except Exception:
                pass

            # 可视化输出: 通过环境变量开启, 避免侵入式修改其它代码
            # - BITMAP_MCC_DEBUG_DIR: 若设置则输出 npz + (可选) PLY
            debug_dir = os.environ.get("BITMAP_MCC_DEBUG_DIR", "").strip()
            if debug_dir:
                try:
                    os.makedirs(debug_dir, exist_ok=True)
                    tag = f"{int(time.time() * 1000)}"
                    np.savez_compressed(
                        os.path.join(debug_dir, f"bitmap_mcc_{tag}.npz"),
                        uv=uv,
                        ix=ix,
                        iy=iy,
                        mcc_cells=np.asarray(mcc_cells, dtype=int),
                        hull=hull_np,
                        inside_mask=inside_mask,
                        inliers_global=inliers_global,
                        connected_inliers_global=connected_inliers_global,
                        leftover_inliers_global=leftover_inliers_global,
                        origin=origin,
                        u=u,
                        v=v,
                        plane_model=np.asarray([n[0], n[1], n[2], d], dtype=float),
                        refined_plane_model=np.asarray(refined_plane_model, dtype=float),
                        grid_size=np.asarray([grid_size], dtype=float)
                    )

                    # 若 Open3D 可用则导出投影点云与connected点云(便于在 CloudCompare/3D 软件中查看)
                    if o3d is not None:
                        pcd_proj = o3d.geometry.PointCloud()
                        pcd_proj.points = o3d.utility.Vector3dVector(pts_proj)
                        o3d.io.write_point_cloud(os.path.join(debug_dir, f"bitmap_mcc_{tag}_proj.ply"), pcd_proj)

                        pcd_conn = o3d.geometry.PointCloud()
                        pcd_conn.points = o3d.utility.Vector3dVector(pts_conn)
                        o3d.io.write_point_cloud(os.path.join(debug_dir, f"bitmap_mcc_{tag}_connected.ply"), pcd_conn)

                except Exception as e:
                    self.logger.warning(f"_bitmapMCCfilter debug export failed: {e}")

            return connected_inliers_global, leftover_inliers_global, refined_plane_model

    # ---------------------------------------------------------
    # Debug export: Step2-4 可视化点云 / 阈值拦截诊断
    # ---------------------------------------------------------
    def _GetDebugExportConfig(self) -> Tuple[Optional[str], str, bool]:
        """读取调试导出配置(外部常量)，避免侵入外部工程代码。

        外部常量:
            SUPERVOXEL_DEBUG_EXPORT_DIR: 调试导出目录；为空则不导出。
            SUPERVOXEL_DEBUG_BASENAME:   文件名前缀(可选)。
            SUPERVOXEL_DEBUG_EXPORT:     1/true 开启；否则关闭(若 DIR 非空也可视为开启)。
        """
        debug_dir = SUPERVOXEL_DEBUG_EXPORT_DIR
        flag = SUPERVOXEL_DEBUG_EXPORT
        enabled = flag or (len(debug_dir) > 0)
        if not enabled:
            return None, "supervoxel_debug", False

        if len(debug_dir) == 0:
            # 若仅开关开启但未指定目录，则默认在当前工作目录下创建子目录
            debug_dir = os.path.join(os.getcwd(), "supervoxel_debug")

        base_name = SUPERVOXEL_DEBUG_BASENAME
        if len(base_name) == 0:
            base_name = "supervoxel_debug"

        os.makedirs(debug_dir, exist_ok=True)
        return debug_dir, base_name, True

    def _ExportDebugPointLevelCsv(
            self,
            csv_path: str,
            coords: np.ndarray,
            normals: np.ndarray,
            rgb: np.ndarray,
            curvature: np.ndarray,
            drdgdb: np.ndarray
    ) -> None:
        """导出“全点云点级CSV”，将可视化颜色写入 DR/DG/DB，不覆盖原始 R/G/B。

        字段对齐 results_exporter.py 的 ExportPointLevelCsv：fileciteturn8file6
        未提供的字段统一用 0 占位（按用户要求）。
        """
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
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

            n = int(coords.shape[0])
            for i in range(n):
                x, y, z = coords[i]
                nx, ny, nz = normals[i]
                r_val, g_val, b_val = rgb[i]
                dr, dg, db = drdgdb[i]
                curv = curvature[i] if not np.isnan(curvature[i]) else 0.0

                writer.writerow([
                    float(x), float(y), float(z),
                    float(nx), float(ny), float(nz),
                    int(r_val), int(g_val), int(b_val),
                    int(dr), int(dg), int(db),
                    0, 0, 0,  # Discontinuity_id / Cluster_id / Segment_id
                    0, 0, 0, 0,  # A,B,C,D
                    0,  # RMS
                    float(curv),
                    0,  # DistToPlane
                ])

    def _ExportDebugPointLevelCsvSubset(
            self,
            csv_path: str,
            indices: np.ndarray,
            coords: np.ndarray,
            normals: np.ndarray,
            rgb: np.ndarray,
            curvature: np.ndarray
    ) -> None:
        """导出点级 CSV 的子集（用于 Step2 voxel 级绘图）。

        - 字段与 _ExportDebugPointLevelCsv 保持一致；
        - DR/DG/DB 默认填 0（不覆盖原始色，也不额外编码）。
        """
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)

        idx = np.asarray(indices, dtype=np.int64).reshape(-1)
        idx = idx[(idx >= 0) & (idx < coords.shape[0])]
        if idx.size == 0:
            return

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
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

            for i in idx.tolist():
                x, y, z = coords[i]
                nx, ny, nz = normals[i]
                r_val, g_val, b_val = rgb[i]
                curv = curvature[i] if (i < curvature.shape[0] and not np.isnan(curvature[i])) else 0.0

                writer.writerow([
                    float(x), float(y), float(z),
                    float(nx), float(ny), float(nz),
                    int(r_val), int(g_val), int(b_val),
                    0, 0, 0,  # DR,DG,DB
                    0, 0, 0,  # Discontinuity_id / Cluster_id / Segment_id
                    0, 0, 0, 0,  # A,B,C,D
                    0,  # RMS
                    float(curv),
                    0,  # DistToPlane
                ])

    def _ComputeVoxelDistributionStats(
            self,
            coords: np.ndarray,
            normals: np.ndarray,
            curvature: np.ndarray,
            indices: List[int],
            plane_normal: np.ndarray,
            plane_d: float,
            normals_valid: np.ndarray
    ) -> Dict[str, float]:
        """计算给定点集在一个 voxel 内的分布统计量（用于 Step2 诊断表）。

        统计项（均以 float 输出，空集返回 NaN）:
            1) normal_angle_deg_* : 点法向与 plane_normal 的夹角 (deg)，使用 |dot| 处理上下半球问题；
            2) curvature_*        : 曲率；
            3) dist_to_plane_*    : 点到平面的距离 |n·p + d|。

        返回字段:
            n_points, n_valid_normals, n_valid_curvature, n_valid_dist,
            normal_angle_deg_mean/std/p50/p90,
            curvature_mean/std/p50/p90,
            dist_to_plane_mean/std/p50/p90
        """
        idx = np.asarray(indices, dtype=np.int64).reshape(-1)
        if idx.size == 0:
            return {
                "n_points": 0,
                "n_valid_normals": 0,
                "n_valid_curvature": 0,
                "n_valid_dist": 0,
                "normal_angle_deg_mean": float("nan"),
                "normal_angle_deg_std": float("nan"),
                "normal_angle_deg_p50": float("nan"),
                "normal_angle_deg_p90": float("nan"),
                "curvature_mean": float("nan"),
                "curvature_std": float("nan"),
                "curvature_p50": float("nan"),
                "curvature_p90": float("nan"),
                "dist_to_plane_mean": float("nan"),
                "dist_to_plane_std": float("nan"),
                "dist_to_plane_p50": float("nan"),
                "dist_to_plane_p90": float("nan"),
            }

        idx = idx[(idx >= 0) & (idx < coords.shape[0])]
        if idx.size == 0:
            return {
                "n_points": 0,
                "n_valid_normals": 0,
                "n_valid_curvature": 0,
                "n_valid_dist": 0,
                "normal_angle_deg_mean": float("nan"),
                "normal_angle_deg_std": float("nan"),
                "normal_angle_deg_p50": float("nan"),
                "normal_angle_deg_p90": float("nan"),
                "curvature_mean": float("nan"),
                "curvature_std": float("nan"),
                "curvature_p50": float("nan"),
                "curvature_p90": float("nan"),
                "dist_to_plane_mean": float("nan"),
                "dist_to_plane_std": float("nan"),
                "dist_to_plane_p50": float("nan"),
                "dist_to_plane_p90": float("nan"),
            }

        n_points = int(idx.size)

        # dist to plane
        pts = coords[idx]
        dist = np.abs(pts.dot(plane_normal) + float(plane_d))
        dist = dist[np.isfinite(dist)]
        n_valid_dist = int(dist.size)

        # normals angle
        n_valid_normals = 0
        ang = np.array([], dtype=float)
        try:
            valid_n_mask = normals_valid[idx]
            if np.any(valid_n_mask):
                nn = normals[idx[valid_n_mask]]
                # 单位化 plane_normal
                pn = plane_normal.astype(float).reshape(3)
                pn_norm = float(np.linalg.norm(pn)) + 1e-12
                pn = pn / pn_norm
                # 单位化 nn
                nn_norm = np.linalg.norm(nn, axis=1)
                m = nn_norm > 1e-12
                nn = nn[m] / nn_norm[m][:, None]
                if nn.shape[0] > 0:
                    n_valid_normals = int(nn.shape[0])
                    cosv = np.clip(np.abs(nn.dot(pn)), 0.0, 1.0)
                    ang = np.degrees(np.arccos(cosv))
        except Exception:
            ang = np.array([], dtype=float)
            n_valid_normals = 0

        # curvature
        curv = np.array([], dtype=float)
        try:
            curv = curvature[idx]
            curv = curv[np.isfinite(curv)]
        except Exception:
            curv = np.array([], dtype=float)
        n_valid_curv = int(curv.size)

        def _stat(x: np.ndarray) -> Tuple[float, float, float, float]:
            if x.size == 0:
                return float("nan"), float("nan"), float("nan"), float("nan")
            return float(np.mean(x)), float(np.std(x)), float(np.percentile(x, 50)), float(np.percentile(x, 90))

        ang_mean, ang_std, ang_p50, ang_p90 = _stat(ang)
        curv_mean, curv_std, curv_p50, curv_p90 = _stat(curv)
        dist_mean, dist_std, dist_p50, dist_p90 = _stat(dist)

        return {
            "n_points": float(n_points),
            "n_valid_normals": float(n_valid_normals),
            "n_valid_curvature": float(n_valid_curv),
            "n_valid_dist": float(n_valid_dist),
            "normal_angle_deg_mean": ang_mean,
            "normal_angle_deg_std": ang_std,
            "normal_angle_deg_p50": ang_p50,
            "normal_angle_deg_p90": ang_p90,
            "curvature_mean": curv_mean,
            "curvature_std": curv_std,
            "curvature_p50": curv_p50,
            "curvature_p90": curv_p90,
            "dist_to_plane_mean": dist_mean,
            "dist_to_plane_std": dist_std,
            "dist_to_plane_p50": dist_p50,
            "dist_to_plane_p90": dist_p90,
        }

    def _ExportDebugRecordsCsv(self, csv_path: str, fieldnames: List[str], records: List[Dict]) -> None:
        if not records:
            return
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in records:
                writer.writerow(r)

    @staticmethod
    def _FillDrDgDbBySets(
            num_points: int,
            default_rgb: Tuple[int, int, int],
            colored_sets: List[Tuple[Set[int], Tuple[int, int, int]]]
    ) -> np.ndarray:
        """根据多个点索引集合为 DR/DG/DB 赋色（后者覆盖前者）。"""
        drdgdb = np.zeros((num_points, 3), dtype=np.int32)
        drdgdb[:, 0] = int(default_rgb[0])
        drdgdb[:, 1] = int(default_rgb[1])
        drdgdb[:, 2] = int(default_rgb[2])

        for idx_set, color in colored_sets:
            if not idx_set:
                continue
            rr, gg, bb = color
            ids = np.fromiter(idx_set, dtype=np.int64)
            ids = ids[(ids >= 0) & (ids < num_points)]
            drdgdb[ids, 0] = int(rr)
            drdgdb[ids, 1] = int(gg)
            drdgdb[ids, 2] = int(bb)

        return drdgdb


# ---------------------------------------------------------
# Debug colors (module-level helpers)
# ---------------------------------------------------------
def GenerateDebugColorFromId(idx: int):
    """生成稳定且区分度较好的调试颜色（用于 Step2/Step3/Step4 的 DR/DG/DB 可视化）。

    输入:
        idx: int
            patch/cluster 的编号（通常为 enumerate 的索引）。
    输出:
        (r, g, b): Tuple[int,int,int]
            0-255 的 RGB。
    """
    import colorsys

    # 黄金分割步进，能让相邻 idx 的色相差异更均匀
    h = (idx * 0.61803398875) % 1.0
    s = 0.65
    v = 0.95
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    rr, gg, bb = int(r * 255), int(g * 255), int(b * 255)

    # 显式规避纯红/纯绿/纯蓝（便于与“结构色”编码区分）
    if (rr, gg, bb) in [(255, 0, 0), (0, 255, 0), (0, 0, 255)]:
        rr = max(0, rr - 10)
        gg = max(0, gg - 10)
        bb = max(0, bb - 10)

    return rr, gg, bb


if __name__ == "__main__":
    # 简单命令行测试: python detector_supervoxel.py <point_cloud_file>
    import sys
    from ..io_pointcloud import PointCloudIO

    if len(sys.argv) < 2:
        print("Usage: python detector_supervoxel.py <point_cloud_file>")
        sys.exit(0)

    pc_path = sys.argv[1]
    pc = PointCloudIO.ReadPointCloud(pc_path)
    detector = SupervoxelDetector()
    discontinuities = detector.DetectPlanes(pc)
    print(f"Detected {len(discontinuities)} planes.")
