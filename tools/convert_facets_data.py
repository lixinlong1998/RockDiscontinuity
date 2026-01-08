# -*- coding: utf-8 -*-
"""
convert_facets_data.py
将 CloudCompare 导出的点云（含 scalar_facet_id）与聚类标签/平面参数合并，
输出制表符分隔的 txt：X,Y,Z,cluster_id,subcluster_id,facet_id,A,B,C,D

规则补充：
- 对于 cluster_id = -1 的点：cluster_id 保持 -1，且 subcluster_id 也统一为 -1（不参与子排序）
- 对于未匹配到标签（NaN）的点：也统一置为 -1，并令 subcluster_id = -1
"""

import os
import pandas as pd
from plyfile import PlyData


# ---------- 小工具：不区分大小写寻找列名 ----------
def find_col(df, candidates, required=True, df_name=""):
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    if required:
        raise KeyError(f"[列缺失] 在 {df_name} 中找不到列：{candidates}")
    return None


def load_ply_as_df(ply_path):
    """读取 PLY 顶点表为 DataFrame，兼容 scalar_facet_id 命名。"""
    ply = PlyData.read(ply_path)
    v = ply["vertex"].data  # numpy structured array
    df = pd.DataFrame(v)

    # 兼容坐标列
    x_col = find_col(df, ["x"], True, "PLY")
    y_col = find_col(df, ["y"], True, "PLY")
    z_col = find_col(df, ["z"], True, "PLY")
    df.rename(columns={x_col: "X", y_col: "Y", z_col: "Z"}, inplace=True)

    # 兼容 facet_id 列（CloudCompare 常见为 scalar_facet_id）
    facet_col = find_col(
        df,
        ["scalar_facet_id", "facet_id", "facetindex", "facet_index", "facetid"],
        True,
        "PLY"
    )
    df.rename(columns={facet_col: "facet_id"}, inplace=True)

    # 若存在 scalar_facet_rms，保留但本任务不输出（如需可扩展）
    rms_col = find_col(df, ["scalar_facet_rms", "facet_rms"], required=False, df_name="PLY")
    if rms_col is not None and rms_col != "scalar_facet_rms":
        df.rename(columns={rms_col: "scalar_facet_rms"}, inplace=True)

    # CloudCompare 常把 facet_id 存为 float，这里转 Int64（可空整型）
    df["facet_id"] = pd.to_numeric(df["facet_id"], errors="coerce").round().astype("Int64")

    return df


def create_output_txt(ply_file, labels_file, metrics_file, output_file):
    # 1) 读取
    df_points = load_ply_as_df(ply_file)
    df_labels = pd.read_csv(labels_file)
    df_metrics = pd.read_csv(metrics_file)

    # 2) 统一关键列名
    # labels: 需要 facet_id & cluster_id（有时列名叫 label/labels）
    lbl_facet = find_col(df_labels, ["facet_id", "FacetID", "facetid"], True, "LABELS")
    lbl_cluster = find_col(df_labels, ["cluster_id", "cluster", "label", "labels"], True, "LABELS")
    df_labels = df_labels.rename(columns={lbl_facet: "facet_id", lbl_cluster: "cluster_id"})

    # metrics: 需要 facet_id & A,B,C,D
    met_facet = find_col(df_metrics, ["facet_id", "FacetID", "facetid"], True, "METRICS")
    met_A = find_col(df_metrics, ["A"], True, "METRICS")
    met_B = find_col(df_metrics, ["B"], True, "METRICS")
    met_C = find_col(df_metrics, ["C"], True, "METRICS")
    met_D = find_col(df_metrics, ["D"], True, "METRICS")
    df_metrics = df_metrics.rename(columns={met_facet: "facet_id", met_A: "A", met_B: "B", met_C: "C", met_D: "D"})

    # 3) 类型对齐
    df_points["facet_id"] = df_points["facet_id"].astype("Int64")
    df_labels["facet_id"] = pd.to_numeric(df_labels["facet_id"], errors="coerce").round().astype("Int64")
    df_metrics["facet_id"] = pd.to_numeric(df_metrics["facet_id"], errors="coerce").round().astype("Int64")

    # 4) 将 cluster_id 映射到点（通过 facet_id）
    df_points = df_points.merge(df_labels[["facet_id", "cluster_id"]], on="facet_id", how="left")

    # 4.1) 统一将未匹配/缺失的 cluster 设为 -1（占位）
    df_points["cluster_id"] = pd.to_numeric(df_points["cluster_id"], errors="coerce")
    df_points["cluster_id"] = df_points["cluster_id"].fillna(-1).astype(int)

    # 5) 计算 subcluster_id：
    #     仅对 cluster_id != -1 的簇，在同一 cluster 内对“唯一 facet_id”按升序编号；
    #     对 cluster_id == -1 的点，subcluster_id 统一为 -1。
    # 5.1) 先对有效簇构建编号表
    valid_facets = (
        df_labels[["cluster_id", "facet_id"]]
        .drop_duplicates()
        .copy()
    )
    # 保留有效簇（cluster_id != -1）
    valid_facets = valid_facets[pd.to_numeric(valid_facets["cluster_id"], errors="coerce") != -1]
    valid_facets = valid_facets.sort_values(["cluster_id", "facet_id"])
    valid_facets["subcluster_id"] = valid_facets.groupby("cluster_id").cumcount() + 1  # 从1开始

    # 5.2) 合并 subcluster_id 到点数据
    df_points = df_points.merge(valid_facets, on=["cluster_id", "facet_id"], how="left")

    # 5.3) 对 cluster_id = -1 的点 或 未匹配到编号的点，subcluster_id 置为 -1
    df_points["subcluster_id"] = pd.to_numeric(df_points["subcluster_id"], errors="coerce")
    df_points.loc[df_points["cluster_id"] == -1, "subcluster_id"] = -1
    df_points["subcluster_id"] = df_points["subcluster_id"].fillna(-1).astype(int)

    # 6) 挂上 A,B,C,D（平面参数）
    df_points = df_points.merge(df_metrics[["facet_id", "A", "B", "C", "D"]], on="facet_id", how="left")

    # 7) 选列与导出
    out_cols = ["X", "Y", "Z", "cluster_id", "subcluster_id", "facet_id", "A", "B", "C", "D"]
    df_points = df_points[out_cols].copy()

    # 数值规范（坐标与 ABCD 转为数值；facet_id、cluster_id、subcluster_id 保持整数）
    for c in ["X", "Y", "Z", "A", "B", "C", "D"]:
        df_points[c] = pd.to_numeric(df_points[c], errors="coerce")

    df_points["facet_id"] = pd.to_numeric(df_points["facet_id"], errors="coerce").fillna(-1).astype(int)
    df_points["cluster_id"] = pd.to_numeric(df_points["cluster_id"], errors="coerce").fillna(-1).astype(int)
    df_points["subcluster_id"] = pd.to_numeric(df_points["subcluster_id"], errors="coerce").fillna(-1).astype(int)

    # 输出
    df_points.to_csv(output_file, sep="\t", index=False)  # float_format="%.12f"去掉 float_format 参数（默认保留全部有效数字）


if __name__ == "__main__":
    workspace = r"D:\Research\20250313_RockFractureSeg\Code\qfacet_gpu\data\facet_export\TSDK_Rockfall_13_P1_ORG_facets_Kd_E0.3A10"
    filebasename = "TSDK_Rockfall_13_P1_ORG_facets"
    ply_file = os.path.join(workspace, f"{filebasename}_points.ply")
    labels_file = os.path.join(workspace, f"{filebasename}_stereographic_labels.csv")
    metrics_file = os.path.join(workspace, f"{filebasename}_metrics.csv")
    output_file = os.path.join(workspace, f"{filebasename}_converted.txt")
    # peaks_file = os.path.join(workspace, "TSDK_Rockfall_1_P2_0.05m_facets_stereographic_KDE_peaks.csv")
    create_output_txt(ply_file, labels_file, metrics_file, output_file)
