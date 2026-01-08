# -*- coding: utf-8 -*-
# CloudCompare PythonRuntime (pycc)
# QFACETS 导出 v1.2：命名/目录按“Cloud [facets]”结点名自动解析；
# 写 facet_id / facet_rms；导出 contour/polygon 为 ASCII PLY；生成统计表。

import os, re, csv, math
import numpy as np
import pycc

# ================= 配置（可改） =================
# 为空则用用户主目录；也可填 D:\exports 之类
PARENT_OUT_DIR = r"D:\Research\20250313_RockFractureSeg\Code\qfacet_gpu\data\facet_export"

# 生成合并的 origin points PLY
MERGE_ALL = True

# 若也要逐 facet 导出 origin points，可设 True
SAVE_PER_FACET = False

# 线/面导出格式：推荐 "SHP"（CC 可见）；也可用 "PLY"
POLYLINE_EXPORT_FORMAT = "PLY"

# 当没有 polygon 结点时，是否用“闭合 contour 或 origin-points 的凸包”来生成 polygon
FALLBACK_BUILD_POLYGON = True

# ==============================================

CC = pycc.GetInstance()
sel = CC.getSelectedEntities()
if not sel:
    raise RuntimeError("请在 DB 树选中包含 QFACETS 的父组或“Cloud [facets]”组后再运行。")
root = sel[0]


# ---------- 工具函数 ----------
def polyline_is_closed(poly: pycc.ccPolyline, tol_ratio: float = 1e-6) -> bool:
    """优先用 isClosed；否则用首尾距离与包围盒对角线比值判断"""
    try:
        if poly.isClosed():
            return True
    except Exception:
        pass
    V = poly_vertices_numpy(poly)
    if V.shape[0] < 3:
        return False
    diag = np.linalg.norm(V.max(0) - V.min(0)) + 1e-15
    return (np.linalg.norm(V[0] - V[-1]) / diag) < tol_ratio


def convex_hull_2d(UV: np.ndarray) -> np.ndarray:
    """二维单调链凸包，返回按逆/顺时针排序的顶点下标"""
    pts = UV.astype(float)
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    P = pts[order]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for i in range(len(P)):
        while len(lower) >= 2 and cross(P[lower[-2]], P[lower[-1]], P[i]) <= 0:
            lower.pop()
        lower.append(i)
    upper = []
    for i in range(len(P) - 1, -1, -1):
        while len(upper) >= 2 and cross(P[upper[-2]], P[upper[-1]], P[i]) <= 0:
            upper.pop()
        upper.append(i)
    idx = lower[:-1] + upper[:-1]
    return order[idx]


def project_to_plane(P3: np.ndarray, ctr: np.ndarray, n: np.ndarray) -> np.ndarray:
    """3D→平面局部坐标 (u,v)"""
    u, v, _ = ortho_basis(n)
    UV = np.stack([(P3 - ctr) @ u, (P3 - ctr) @ v], axis=1)
    return UV


def polygon_area_from_vertices3d(V3: np.ndarray, ctr: np.ndarray, n: np.ndarray) -> float:
    """3D 闭合多边形的面积"""
    if V3.shape[0] < 3: return float("nan")
    UV = project_to_plane(V3, ctr, n)
    x, y = UV[:, 0], UV[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def build_polygon_from_contour_or_points(oc, ct, ctr, n) -> np.ndarray:
    """
    返回 polygon 顶点的 3D 坐标序列：
    1) 若 contour 闭合：直接用它的顶点（若未闭合但首尾很近，自动闭合）
    2) 否则：用 origin points 的 2D 凸包（在平面上取 hull，回投影到 3D）
    若都不可用，返回 None
    """
    # 1) 闭合 contour
    if ct is not None and ct.size() >= 3 and polyline_is_closed(ct):
        V = poly_vertices_numpy(ct)
        # 若首尾没完全重合，这里手动闭合一下
        if np.linalg.norm(V[0] - V[-1]) > 0:
            V = np.vstack([V, V[0]])
        return V

    # 2) origin points 凸包
    if oc is not None and oc.size() >= 3:
        P = cloud_points_numpy(oc)
        UV = project_to_plane(P, ctr, n)
        idx = convex_hull_2d(UV)
        hull2d = UV[idx]
        # 回到 3D： ctr + u*x + v*y
        u, v, _ = ortho_basis(n)
        V3 = ctr + np.outer(hull2d[:, 0], u) + np.outer(hull2d[:, 1], v)
        # 闭合
        if np.linalg.norm(V3[0] - V3[-1]) > 0:
            V3 = np.vstack([V3, V3[0]])
        return V3

    return None


def write_polygon_vertices_as_ascii_ply(V3: np.ndarray, path: str):
    """把闭合 polygon 写成 vertex+edge 的 ASCII PLY"""
    n = V3.shape[0]
    edges = [(i, i + 1) for i in range(n - 1)]
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element edge {len(edges)}\n")
        f.write("property int vertex1\nproperty int vertex2\n")
        f.write("end_header\n")
        for p in V3: f.write(f"{p[0]} {p[1]} {p[2]}\n")
        for a, b in edges: f.write(f"{a} {b}\n")


def iter_children(n):
    for i in range(n.getChildrenNumber()):
        yield n.getChild(i)


def find_cloud_facets_node(n):
    """返回名称包含 'Cloud [facets]' 的第一个结点；若自己就是则返回自己"""
    if "Cloud [facets]" in n.getName():
        return n
    for ch in iter_children(n):
        r = find_cloud_facets_node(ch)
        if r: return r
    return None


def sanitize_float_token(xstr):
    try:
        v = float(xstr)
        s = f"{v:.6f}".rstrip('0').rstrip('.')
        return s
    except Exception:
        return xstr


def parse_workspace_from_cloud_facets_name(name: str):
    """
    例：
    TSDK_Rockfall_1_P2_0.05m - Cloud [facets] [Kd-tree][error < 0.2][angle < 20 deg.]
    -> base='TSDK_Rockfall_1_P2_0.05m', algo='Kd', E='0.2', A='20'
    """
    base = name.split(" - Cloud")[0].strip()
    algo = "Alg"
    if re.search(r"\[Kd[-\s]?tree\]", name, re.I):
        algo = "Kd"
    elif re.search(r"\[Octree\]", name, re.I):
        algo = "Oct"

    mE = re.search(r"\[error\s*[<≤]\s*([0-9+.]+)\s*\]", name, re.I)
    mA = re.search(r"\[angle\s*[<≤]\s*([0-9+.]+)\s*(deg\.?|°)?\s*\]", name, re.I)
    E = sanitize_float_token(mE.group(1)) if mE else "NA"
    A = sanitize_float_token(mA.group(1)) if mA else "NA"

    ws = f"{base}_facets_{algo}_E{E}A{A}"
    return base, ws


def ensure_dir(p):
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def home_dir():
    return os.path.expanduser("~")


def get_or_add_sf(cloud: pycc.ccPointCloud, name: str):
    idx = cloud.getScalarFieldIndexByName(name)
    if idx >= 0:
        sf = cloud.getScalarField(idx)
        return idx, sf.asArray()
    if cloud.size() == 0:
        raise RuntimeError(f"点云“{cloud.getName()}”为空，无法创建标量场 {name}")
    idx = cloud.addScalarField(name)
    if idx < 0:
        raise RuntimeError(f"无法在点云“{cloud.getName()}”创建标量场 {name}")
    return idx, cloud.getScalarField(idx).asArray()


def find_origin_cloud(facet_obj):
    for ch in iter_children(facet_obj):
        if isinstance(ch, pycc.ccPointCloud) and "Origin" in ch.getName():
            return ch
    return None


def find_polylines(facet_obj):
    contour = None
    polygon = None
    stack = [facet_obj]
    while stack:
        n = stack.pop()
        for ch in iter_children(n):
            stack.append(ch)
            if isinstance(ch, pycc.ccPolyline):
                nm = ch.getName().lower()
                if "polygon" in nm:
                    polygon = ch
                elif "contour" in nm:
                    contour = ch
    return contour, polygon


def collect_facets(node):
    items = []
    nm = node.getName()
    if re.search(r"^facet\s+\d+", nm, re.I):
        oc = find_origin_cloud(node)
        ct, pg = find_polylines(node)
        m = re.search(r"facet\s+(\d+)", nm, re.I)
        fid = int(m.group(1)) if m else 0
        m2 = re.search(r"RMS=([0-9.+\-Ee]+)", nm)
        rms = float(m2.group(1)) if m2 else float("nan")
        items.append((fid, rms, oc, ct, pg, node))
    for ch in iter_children(node):
        items.extend(collect_facets(ch))
    return items


def cloud_points_numpy(cloud: pycc.ccPointCloud):
    P = cloud.points()
    if isinstance(P, np.ndarray):
        return P.copy()
    arr = np.zeros((cloud.size(), 3), dtype=np.float64)
    for i in range(cloud.size()):
        p = cloud.getPoint(i)
        arr[i] = [float(p.x), float(p.y), float(p.z)]
    return arr


def poly_vertices_numpy(poly: pycc.ccPolyline):
    n = poly.size()
    V = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        p = poly.getPoint(i)
        V[i] = [float(p.x), float(p.y), float(p.z)]
    return V


def fit_plane_svd(pts: np.ndarray):
    c = pts.mean(axis=0)
    Q = pts - c
    _, _, Vt = np.linalg.svd(Q, full_matrices=False)
    n = Vt[-1];
    n = n / (np.linalg.norm(n) + 1e-15)
    A, B, C = n.tolist()
    D = -float(np.dot(n, c))
    return A, B, C, D, c, n


def ortho_basis(n):
    n = n / (np.linalg.norm(n) + 1e-15)
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, a);
    u = u / (np.linalg.norm(u) + 1e-15)
    v = np.cross(n, u)
    return u, v, n


def polygon_area_on_plane(V3, ctr, n):
    if V3.shape[0] < 3: return float("nan")
    u, v, _ = ortho_basis(n)
    UV = np.stack([(V3 - ctr) @ u, (V3 - ctr) @ v], axis=1)
    x, y = UV[:, 0], UV[:, 1]
    s = 0.0
    for i in range(len(x)):
        j = (i + 1) % len(x)
        s += x[i] * y[j] - x[j] * y[i]
    return abs(s) * 0.5


def polyline_length(V3):
    if V3.shape[0] < 2: return 0.0
    L = np.linalg.norm(V3[1:] - V3[:-1], axis=1).sum()
    return float(L)


def max_diameter_endpoints(V3):
    """求顶点集最大两点距离与端点（polygon 无 contour 时做迹长近似）"""
    if V3.shape[0] < 2: return 0.0, (np.nan,) * 6
    # O(n^2)；顶点数通常不大
    dmax = -1.0;
    a = b = 0
    for i in range(len(V3)):
        d = np.linalg.norm(V3[i + 1:] - V3[i], axis=1)
        if d.size == 0: continue
        jrel = int(np.argmax(d))
        if d[jrel] > dmax:
            dmax = float(d[jrel])
            a, b = i, i + 1 + jrel
    s = V3[a];
    e = V3[b]
    return dmax, (s[0], s[1], s[2], e[0], e[1], e[2])


def write_polyline_as_ascii_ply(poly: pycc.ccPolyline, path: str):
    """把 ccPolyline 写成含 edge 元素的 ASCII PLY"""
    V = poly_vertices_numpy(poly)
    n = V.shape[0]
    if n == 0:
        return
    edges = []
    for i in range(n - 1):
        edges.append((i, i + 1))
    closed = False
    try:
        closed = bool(poly.isClosed())
    except Exception:
        closed = False
    if closed and n >= 3:
        edges.append((n - 1, 0))

    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element edge {len(edges)}\n")
        f.write("property int vertex1\nproperty int vertex2\n")
        f.write("end_header\n")
        for i in range(n):
            f.write(f"{V[i, 0]} {V[i, 1]} {V[i, 2]}\n")
        for (a, b) in edges:
            f.write(f"{a} {b}\n")


# ---------- 定位 Cloud [facets] 名称并生成 workspace ----------
cf_node = find_cloud_facets_node(root) or root
base_name, workspace_name = parse_workspace_from_cloud_facets_name(cf_node.getName())

parent = PARENT_OUT_DIR.strip() or home_dir()
workspace = os.path.join(parent, workspace_name)
cont_dir = os.path.join(workspace, f"{base_name}_facets_contours")
poly_dir = os.path.join(workspace, f"{base_name}_facets_polygons")
ensure_dir(cont_dir);
ensure_dir(poly_dir);
ensure_dir(workspace)

points_merged_path = os.path.join(workspace, f"{base_name}_facets_points.ply")
table_csv_path = os.path.join(workspace, f"{base_name}_facets_metrics.csv")

# ---------- 收集 facets ----------
facets = collect_facets(root)
if not facets:
    raise RuntimeError("未找到任何 facet。请确认选择的分组正确。")

# ---------- 写标量场 facet_id / facet_rms ----------
for fid, rms, oc, ct, pg, node in facets:
    if oc is None or oc.size() == 0:
        continue
    idx_id, arr_id = get_or_add_sf(oc, "facet_id")
    arr_id[:] = float(fid)
    oc.getScalarField(idx_id).computeMinAndMax()

    idx_r, arr_r = get_or_add_sf(oc, "facet_rms")
    arr_r[:] = float(rms) if not math.isnan(rms) else np.nan
    oc.getScalarField(idx_r).computeMinAndMax()

    oc.setCurrentDisplayedScalarField(idx_id)
    oc.showSF(True)

# ---------- 导出 contour / polygon ----------
params = pycc.FileIOFilter.SaveParameters()
params.alwaysDisplaySaveDialog = False  # 不弹保存对话框（沿用上次 GUI 选择）
for fid, rms, oc, ct, pg, node in facets:
    # contour
    if ct is not None and ct.size() > 0:
        if POLYLINE_EXPORT_FORMAT.upper() == "SHP":
            outp = os.path.join(cont_dir, f"facet_contour_{fid:03d}.shp")
            pycc.FileIOFilter.SaveToFile(ct, outp, params)
        else:
            outp = os.path.join(cont_dir, f"facet_contour_{fid:03d}.ply")
            write_polyline_as_ascii_ply(ct, outp)

    # polygon（优先现成的 polygon；否则按开关回退构造）
    if pg is not None and pg.size() > 0:
        if POLYLINE_EXPORT_FORMAT.upper() == "SHP":
            outp = os.path.join(poly_dir, f"facet_polygon_{fid:03d}.shp")
            pycc.FileIOFilter.SaveToFile(pg, outp, params)
        else:
            outp = os.path.join(poly_dir, f"facet_polygon_{fid:03d}.ply")
            write_polyline_as_ascii_ply(pg, outp)
    elif FALLBACK_BUILD_POLYGON:
        # 用 contour/points 构造 polygon 并导出
        # 先用 origin/contour 拟合平面
        Pfit = None
        if oc is not None and oc.size() >= 3:
            Pfit = cloud_points_numpy(oc)
        elif ct is not None and ct.size() >= 3:
            Pfit = poly_vertices_numpy(ct)
        if Pfit is None:
            continue
        A, B, C, D, ctr, n = fit_plane_svd(Pfit)
        V3 = build_polygon_from_contour_or_points(oc, ct, ctr, n)
        if V3 is None:
            continue
        if POLYLINE_EXPORT_FORMAT.upper() == "SHP":
            # 先把 V3 临时转成 ccPolyline（闭合）
            tmp = pycc.ccPointCloud(V3[:, 0].astype(pycc.PointCoordinateType),
                                    V3[:, 1].astype(pycc.PointCoordinateType),
                                    V3[:, 2].astype(pycc.PointCoordinateType))
            poly = pycc.ccPolyline(tmp)
            poly.addPointIndex(0, V3.shape[0])
            poly.setClosed(True)
            outp = os.path.join(poly_dir, f"facet_polygon_{fid:03d}.shp")
            pycc.FileIOFilter.SaveToFile(poly, outp, params)
            del poly;
            del tmp
        else:
            outp = os.path.join(poly_dir, f"facet_polygon_{fid:03d}.ply")
            write_polygon_vertices_as_ascii_ply(V3, outp)

# ---------- 计算几何量并输出表 ----------
rows = []
for fid, rms, oc, ct, pg, node in facets:
    # 平面：优先用 origin points
    P = None
    if oc is not None and oc.size() >= 3:
        P = cloud_points_numpy(oc)
    elif ct is not None and ct.size() >= 3:
        P = poly_vertices_numpy(ct)
    elif pg is not None and pg.size() >= 3:
        P = poly_vertices_numpy(pg)

    if P is not None:
        A, B, C, D, ctr, n = fit_plane_svd(P)
    else:
        A = B = C = D = float("nan");
        ctr = np.zeros(3);
        n = np.array([0, 0, 1.0])

    # 面积：优先 polygon；否则闭合 contour；再否则 origin-points 的凸包
    Area = float("nan")
    if pg is not None and pg.size() >= 3:
        Vpg = poly_vertices_numpy(pg)
        Area = polygon_area_from_vertices3d(Vpg, ctr, n)
    elif FALLBACK_BUILD_POLYGON:
        Vbuilt = build_polygon_from_contour_or_points(oc, ct, ctr, n)
        if Vbuilt is not None and Vbuilt.shape[0] >= 3:
            Area = polygon_area_from_vertices3d(Vbuilt, ctr, n)

    # 迹长与端点：优先 contour；若无，则用 polygon 顶点对的最大直径近似
    if ct is not None and ct.size() >= 2:
        Vct = poly_vertices_numpy(ct)
        TraceLength = polyline_length(Vct)
        sx, sy, sz = Vct[0].tolist()
        ex, ey, ez = Vct[-1].tolist()
    elif pg is not None and pg.size() >= 2:
        Vpg = poly_vertices_numpy(pg)
        TraceLength, (sx, sy, sz, ex, ey, ez) = max_diameter_endpoints(Vpg)
    else:
        TraceLength = float("nan")
        sx = sy = sz = ex = ey = ez = float("nan")

    facet_points_number = int(oc.size()) if oc is not None else 0

    rows.append([
        int(fid),
        float(rms),
        facet_points_number,
        float(A), float(B), float(C), float(D),
        float(Area),
        float(TraceLength),
        float(sx), float(sy), float(sz),
        float(ex), float(ey), float(ez),
    ])

# 写 CSV
header = [
    "facet_id", "facet_rms", "facet_points_number",
    "A", "B", "C", "D", "Area", "TraceLength",
    "TraceStartX", "TraceStartY", "TraceStartZ",
    "TraceEndX", "TraceEndY", "TraceEndZ",
]
with open(table_csv_path, "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows([header, *rows])

# ---------- 合并 origin points 并导出 ----------
if MERGE_ALL:
    pts_all = [];
    fid_all = [];
    rms_all = []
    for fid, rms, oc, ct, pg, node in facets:
        if oc is None or oc.size() == 0: continue
        P = cloud_points_numpy(oc)
        pts_all.append(P)
        fid_all.append(np.full((oc.size(),), float(fid), dtype=np.float32))
        rms_all.append(np.full((oc.size(),), float(rms), dtype=np.float32))
    if pts_all:
        P = np.vstack(pts_all)
        xs = P[:, 0].astype(pycc.PointCoordinateType)
        ys = P[:, 1].astype(pycc.PointCoordinateType)
        zs = P[:, 2].astype(pycc.PointCoordinateType)
        merged = pycc.ccPointCloud(xs, ys, zs)
        merged.setName(f"{base_name}_facets_points")
        idx_f = merged.addScalarField("facet_id", np.concatenate(fid_all))
        merged.getScalarField(idx_f).computeMinAndMax()
        idx_r = merged.addScalarField("facet_rms", np.concatenate(rms_all))
        merged.getScalarField(idx_r).computeMinAndMax()
        merged.setCurrentDisplayedScalarField(idx_f);
        merged.showSF(True)
        CC.addToDB(merged);
        CC.updateUI()
        pycc.FileIOFilter.SaveToFile(merged, points_merged_path, pycc.FileIOFilter.SaveParameters())

print("完成：")
print(f"  Workspace: {workspace}")
print(f"  合并点云:  {points_merged_path}")
print(f"  轮廓目录:  {cont_dir}")
print(f"  多边形目录:{poly_dir}")
print(f"  统计表:    {table_csv_path}")
