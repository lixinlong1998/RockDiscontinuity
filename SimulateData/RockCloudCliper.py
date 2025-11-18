import os
import math
import argparse
from pathlib import Path
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import open3d as o3d
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances


def load_pcd(ply_path: Path) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(str(ply_path))
    if pcd.is_empty():
        raise ValueError(f"Empty point cloud: {ply_path}")
    return pcd


def ensure_normals(
    pcd: o3d.geometry.PointCloud,
    normal_k: int = 30,
    normal_radius: float = None,
) -> None:
    """
    若无法向，则估计法向；若已有法向，直接保留。
    建议 normal_radius 约为 cube_size 的 0.1~0.2。
    """
    has_normals = np.asarray(pcd.normals).shape[0] == np.asarray(pcd.points).shape[0]
    if has_normals:
        # 归一化，避免后续角度计算受幅值影响
        n = np.asarray(pcd.normals, dtype=np.float64)
        n_norm = np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        pcd.normals = o3d.utility.Vector3dVector(n / n_norm)
        return

    if normal_radius is not None and normal_radius > 0:
        param = o3d.geometry.KDTreeSearchParamHybrid(radius=float(normal_radius), max_nn=int(normal_k))
        pcd.estimate_normals(search_param=param)
    else:
        param = o3d.geometry.KDTreeSearchParamKNN(knn=int(normal_k))
        pcd.estimate_normals(search_param=param)

    # 可选：一致化法向朝向（以质心为参考）
    pcd.orient_normals_towards_camera_location(camera_location=pcd.get_center())


def compute_cube_grid(bmin: np.ndarray, bmax: np.ndarray, cube_size: float, stride: float):
    """
    返回所有小立方体的左下近角（x0,y0,z0）列表。
    """
    xs = np.arange(bmin[0], bmax[0], stride)
    ys = np.arange(bmin[1], bmax[1], stride)
    zs = np.arange(bmin[2], bmax[2], stride)
    for ix, x0 in enumerate(xs):
        for iy, y0 in enumerate(ys):
            for iz, z0 in enumerate(zs):
                yield ix, iy, iz, float(x0), float(y0), float(z0)


def crop_points(pts: np.ndarray, x0: float, y0: float, z0: float, cube_size: float) -> np.ndarray:
    m = (
        (pts[:, 0] >= x0) & (pts[:, 0] < x0 + cube_size) &
        (pts[:, 1] >= y0) & (pts[:, 1] < y0 + cube_size) &
        (pts[:, 2] >= z0) & (pts[:, 2] < z0 + cube_size)
    )
    idx = np.nonzero(m)[0]
    return idx


def flip_normals_to_hemisphere(normals: np.ndarray) -> np.ndarray:
    """
    将法向统一到同一半球（±n 等价）。我们用主方向作为参考后翻转。
    """
    # PCA 第一主方向
    u, s, vh = np.linalg.svd(normals, full_matrices=False)
    ref = vh[0]  # (3,)
    dots = normals @ ref
    flipped = normals.copy()
    flipped[dots < 0] *= -1.0
    return flipped


def kmeans_cosine(normals: np.ndarray, k: int, random_state: int = 0):
    """
    在单位球面上做 KMeans（欧氏 KMeans + 归一化 + 余弦指标评估）。
    """
    # KMeans 在欧氏空间做，但我们事先单位化；评估用余弦距离
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(normals)
    centers = km.cluster_centers_
    centers /= (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12)
    return labels, centers


def cluster_normals_1to3(normals_raw: np.ndarray) -> Dict[str, Any]:
    """
    在 k=1,2,3 中自适应选择最佳簇数。
    评价：
      - 对 k>=2 用 silhouette_score(metric='cosine')
      - 对 k=1 用到簇离散度（平均与中心的余弦距离）
    """
    normals = normals_raw.astype(np.float64)
    norms = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    normals = normals / norms

    normals = flip_normals_to_hemisphere(normals)

    results = []
    # k=1
    centroid = np.mean(normals, axis=0)
    centroid /= (np.linalg.norm(centroid) + 1e-12)
    dists = cosine_distances(normals, centroid[None, :]).reshape(-1)  # 余弦距离 ~ 角度的单调函数
    dispersion = float(np.mean(dists))
    results.append(dict(k=1, score=1.0 - dispersion, kind="dispersion", labels=np.zeros(len(normals), dtype=int),
                        centers=centroid[None, :]))

    # k=2,3
    for k in (2, 3):
        labels, centers = kmeans_cosine(normals, k=k)
        try:
            sil = silhouette_score(normals, labels, metric="cosine")
        except Exception:
            sil = -1.0
        results.append(dict(k=k, score=float(sil), kind="silhouette", labels=labels, centers=centers))

    # 选择得分最高者；打平时优先更小的 k（更简洁）
    results.sort(key=lambda r: (r["score"], -r["k"]), reverse=True)
    best = results[0]

    # 统计簇占比
    labels = best["labels"]
    k = best["k"]
    counts = np.bincount(labels, minlength=k).astype(int)
    props = (counts / max(1, len(labels))).tolist()

    # 簇中心两两夹角（度）
    def angle_deg(a, b):
        v = np.clip(float(np.dot(a, b)), -1.0, 1.0)
        return math.degrees(math.acos(v))

    centers = best["centers"]
    pair_angles = []
    for i in range(k):
        for j in range(i + 1, k):
            pair_angles.append(angle_deg(centers[i], centers[j]))

    # 判定是否“聚类到 1–3 类”
    # 阈值可调：k>=2 用 silhouette >= 0.35；k=1 用 dispersion<=0.15（等价 score>=0.85）
    is_clustered = False
    if k == 1:
        is_clustered = (1.0 - np.mean(dists)) >= 0.85
    else:
        is_clustered = best["score"] >= 0.35

    return dict(
        best_k=k,
        score=best["score"],
        kind=best["kind"],
        labels=labels,
        centers=centers,
        counts=counts.tolist(),
        proportions=props,
        pair_angles_deg=pair_angles,
        is_clustered=bool(is_clustered),
    )


def color_by_labels(npts: int, labels: np.ndarray, k: int) -> np.ndarray:
    """
    为不同簇着色（RGB, 0-1）。最多 3 类，循环使用三种对比色。
    """
    palette = np.array([
        [0.95, 0.35, 0.35],  # red-ish
        [0.35, 0.75, 0.95],  # blue-ish
        [0.40, 0.85, 0.55],  # green-ish
    ], dtype=np.float64)
    colors = np.zeros((npts, 3), dtype=np.float64) + 0.7  # 默认灰
    if k >= 1:
        for cid in range(k):
            colors[labels == cid] = palette[cid % len(palette)]
    return colors


def process_one_file(
    ply_path: Path,
    out_dir: Path,
    cube_size: float,
    stride: float,
    min_points: int,
    estimate_normals_if_missing: bool,
    normal_k: int,
    normal_radius: float,
    voxel_down: float = None
) -> List[Dict[str, Any]]:
    stem = ply_path.stem
    pcd = load_pcd(ply_path)

    # 选配：先体素下采样，提速与抑制噪声
    if voxel_down and voxel_down > 0:
        pcd = pcd.voxel_down_sample(voxel_size=float(voxel_down))

    if estimate_normals_if_missing:
        ensure_normals(pcd, normal_k=normal_k, normal_radius=normal_radius)

    pts = np.asarray(pcd.points)
    has_normals = np.asarray(pcd.normals).shape[0] == pts.shape[0]
    if not has_normals:
        # 若用户不估计法向，但又需要评价，则跳过评价仅裁切
        pass

    bmin = pts.min(axis=0)
    bmax = pts.max(axis=0)

    rows = []
    for ix, iy, iz, x0, y0, z0 in compute_cube_grid(bmin, bmax, cube_size, stride):
        sel_idx = crop_points(pts, x0, y0, z0, cube_size)
        if sel_idx.size < min_points:
            continue

        # 构造子点云
        sub = o3d.geometry.PointCloud()
        sub.points = o3d.utility.Vector3dVector(pts[sel_idx])

        eval_res = None
        if has_normals:
            n_sub = np.asarray(pcd.normals)[sel_idx]
            sub.normals = o3d.utility.Vector3dVector(n_sub)
            eval_res = cluster_normals_1to3(n_sub)

            # 上色
            colors = color_by_labels(len(sel_idx), np.array(eval_res["labels"]), eval_res["best_k"])
            sub.colors = o3d.utility.Vector3dVector(colors)

        # 保存子 PLY
        fname = f"{stem}_Cube_x{ix}_y{iy}_z{iz}_X{round(x0,3)}_Y{round(y0,3)}_Z{round(z0,3)}_S{cube_size}_T{stride}_N{sel_idx.size}"
        if eval_res is not None:
            fname += f"_K{eval_res['best_k']}"
        out_path = out_dir / f"{fname}.ply"
        o3d.io.write_point_cloud(str(out_path), sub, write_ascii=False, compressed=False)

        # 记录清单
        row = dict(
            src_file=str(ply_path),
            cube_file=str(out_path),
            ix=ix, iy=iy, iz=iz,
            x0=x0, y0=y0, z0=z0,
            cube_size=cube_size, stride=stride,
            n_points=int(sel_idx.size),
        )
        if eval_res is not None:
            row.update(dict(
                best_k=int(eval_res["best_k"]),
                score=float(eval_res["score"]),
                score_kind=str(eval_res["kind"]),
                is_clustered=bool(eval_res["is_clustered"]),
                proportions=";".join([f"{p:.4f}" for p in eval_res["proportions"]]),
                pair_angles_deg=";".join([f"{a:.2f}" for a in eval_res["pair_angles_deg"]]),
                centers=";".join([",".join([f"{c:.5f}" for c in ctr]) for ctr in eval_res["centers"]]),
            ))
        rows.append(row)

    return rows


def main():
    parser = argparse.ArgumentParser(description="Batch-cut PLY into cubes and evaluate normal clustering (1–3 groups).")
    parser.add_argument("--input_dir", type=str, required=True, help="包含 PLY 的目录（递归）")
    parser.add_argument("--output_dir", type=str, required=True, help="输出小块 PLY 的目录")
    parser.add_argument("--cube_size", type=float, required=True, help="立方体边长（与点云坐标单位一致）")
    parser.add_argument("--stride", type=float, default=None, help="滑窗步长；默认等于 cube_size（无重叠）")
    parser.add_argument("--min_points", type=int, default=500, help="小块保存的最小点数阈值")
    parser.add_argument("--estimate_normals_if_missing", action="store_true", help="当源数据无法向时，估计法向")
    parser.add_argument("--normal_k", type=int, default=30, help="估计法向的 KNN 数")
    parser.add_argument("--normal_radius", type=float, default=0.0, help="估计法向的邻域半径（0 表示不用半径）")
    parser.add_argument("--voxel_down", type=float, default=0.0, help="预下采样体素（0 表示不下采样）")
    parser.add_argument("--manifest", type=str, default="manifest.csv", help="汇总 CSV 文件名")
    args = parser.parse_args()

    cube_size = float(args.cube_size)
    stride = float(args.stride) if args.stride and args.stride > 0 else cube_size
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []

    # 递归搜集所有 .ply
    ply_files = [Path(p) for p in Path(args.input_dir).rglob("*.ply")]
    if not ply_files:
        print(f"[WARN] No PLY files found in {args.input_dir}")
        return

    for i, ply_path in enumerate(ply_files, 1):
        print(f"[{i}/{len(ply_files)}] Processing: {ply_path}")
        try:
            rows = process_one_file(
                ply_path=ply_path,
                out_dir=out_dir,
                cube_size=cube_size,
                stride=stride,
                min_points=args.min_points,
                estimate_normals_if_missing=args.estimate_normals_if_missing,
                normal_k=args.normal_k,
                normal_radius=args.normal_radius if args.normal_radius > 0 else None,
                voxel_down=args.voxel_down if args.voxel_down > 0 else None
            )
            all_rows.extend(rows)
        except Exception as e:
            print(f"[ERROR] {ply_path}: {e}")

    # 写 manifest
    if all_rows:
        df = pd.DataFrame(all_rows)
        manifest_path = out_dir / args.manifest
        df.to_csv(manifest_path, index=False, encoding="utf-8-sig")
        print(f"[DONE] Saved {len(all_rows)} cubes. Manifest: {manifest_path}")
    else:
        print("[DONE] No cubes produced (check min_points/stride/cube_size).")


if __name__ == "__main__":
    main()
