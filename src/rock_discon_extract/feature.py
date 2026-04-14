import numpy as np

def _BuildKdTree(points: np.ndarray):
    """
    功能简介:
        为点云构建KDTree，支持scipy或sklearn二选一。
    实现思路:
        优先使用scipy.spatial.cKDTree；若不可用则回退到sklearn.neighbors.KDTree。
    输入:
        points: (N,3) float
    输出:
        tree: KDTree对象
        query_ball: callable(points, r) -> list[list[int]]
    """
    try:
        from scipy.spatial import cKDTree  # 更常见、更快
        tree = cKDTree(points)
        def query_ball(P, r):
            return tree.query_ball_point(P, r)
        return tree, query_ball
    except Exception:
        from sklearn.neighbors import KDTree
        tree = KDTree(points, leaf_size=40)
        def query_ball(P, r):
            ind = tree.query_radius(P, r, return_distance=False)
            return [x.tolist() for x in ind]
        return tree, query_ball

def _TriangleArea(p1, p2, p3):
    # 三角形面积
    return 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))

def _TetraVolume(p1, p2, p3, p4):
    # 四面体体积
    return abs(np.dot(p2 - p1, np.cross(p3 - p1, p4 - p1))) / 6.0

def _Angle(p1, p2, p3):
    # 以p1为顶点的夹角 ∠(p2-p1, p3-p1)，弧度
    u = p2 - p1
    v = p3 - p1
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return 0.0
    c = np.dot(u, v) / (nu * nv)
    c = np.clip(c, -1.0, 1.0)
    return float(np.arccos(c))

def _SampleMeasure(neigh_pts: np.ndarray, centroid: np.ndarray, measure: str, rng: np.random.Generator):
    """
    功能简介:
        从一个邻域点集采样一次指定几何量（D1/D2/D3/D4/A3）。
    实现思路:
        按需要随机抽取1/2/3/4个点并计算对应几何量。
    输入:
        neigh_pts: (M,3)
        centroid: (3,)
        measure: 'D1'/'D2'/'D3'/'D4'/'A3'
        rng: numpy随机数生成器
    输出:
        val: float
    """
    m = neigh_pts.shape[0]
    if measure == "D1":
        p = neigh_pts[rng.integers(0, m)]
        return float(np.linalg.norm(p - centroid))
    if measure == "D2":
        i, j = rng.integers(0, m, size=2)
        return float(np.linalg.norm(neigh_pts[i] - neigh_pts[j]))
    if measure == "D3":
        i, j, k = rng.integers(0, m, size=3)
        area = _TriangleArea(neigh_pts[i], neigh_pts[j], neigh_pts[k])
        return float(np.sqrt(area))
    if measure == "D4":
        i, j, k, l = rng.integers(0, m, size=4)
        vol = _TetraVolume(neigh_pts[i], neigh_pts[j], neigh_pts[k], neigh_pts[l])
        return float(np.cbrt(vol))
    if measure == "A3":
        i, j, k = rng.integers(0, m, size=3)
        return _Angle(neigh_pts[i], neigh_pts[j], neigh_pts[k])
    raise ValueError(f"Unknown measure: {measure}")

def _AdaptiveBinEdges(points, neighbors, r, measure, bins, rng,
                      max_points=2000, max_vals=8000, min_neigh=10):
    """
    功能简介:
        在尺度r下，为某个measure估计自适应分箱边界（用分位数近似CDF均衡分箱）。
    实现思路:
        从若干点的邻域中累积采样几何量，使用quantile生成等概率bins边界。
    输入:
        points: (N,3)
        neighbors: list[list[int]] 对应尺度r的邻域索引
        r: float
        measure: str
        bins: int
        rng: random generator
    输出:
        edges: (bins+1,) float, 单调不减
    """
    N = points.shape[0]
    idx = np.arange(N)
    rng.shuffle(idx)
    idx = idx[:min(max_points, N)]

    vals = []
    for i in idx:
        nb = neighbors[i]
        if len(nb) < min_neigh:
            continue
        neigh_pts = points[nb]
        centroid = neigh_pts.mean(axis=0)
        # 每个点采样少量，快速攒够vals
        for _ in range(8):
            vals.append(_SampleMeasure(neigh_pts, centroid, measure, rng))
        if len(vals) >= max_vals:
            break

    if len(vals) < max(50, bins * 5):
        # 样本太少时，退化为线性边界（避免崩）
        v = np.array(vals) if len(vals) > 0 else np.array([0.0, 1.0])
        vmin, vmax = float(v.min()), float(v.max())
        if vmax == vmin:
            vmax = vmin + 1e-6
        return np.linspace(vmin, vmax, bins + 1)

    v = np.array(vals, dtype=float)
    qs = np.linspace(0.0, 1.0, bins + 1)
    edges = np.quantile(v, qs)
    # 去重/单调性保障（分位数可能重复）
    for k in range(1, edges.size):
        if edges[k] <= edges[k - 1]:
            edges[k] = edges[k - 1] + 1e-12
    return edges

def ComputeShapeDistributionFeatures(points: np.ndarray,
                                     radii,
                                     bins: int = 10,
                                     samples_per_point: int = 255,
                                     measures=("D1", "D2", "D3", "D4", "A3"),
                                     min_neigh: int = 10,
                                     seed: int = 0):
    """
    功能简介:
        计算论文中的 shape distribution 多尺度几何直方图特征（D1/D2/D3/D4/A3）。
    实现思路:
        对每个尺度r:
          1) 半径邻域
          2) 对每个measure用分位数估计自适应分箱边界(近似CDF均衡)
          3) 对每个点在邻域内随机采样samples_per_point次measure，做归一化直方图
        拼接得到最终特征矩阵。
    输入:
        points: (N,3) float
        radii: list/tuple/ndarray of float，多尺度半径
        bins: int，每个measure的直方图bin数
        samples_per_point: int，每点每measure的随机采样次数（论文常用255）
        measures: iterable[str]，默认('D1','D2','D3','D4','A3')
        min_neigh: int，邻域最小点数，不足则该点该尺度特征置零
        seed: int，随机种子
    输出:
        feat: (N, len(radii)*len(measures)*bins) float
    """
    points = np.asarray(points, dtype=float)
    radii = list(radii)
    rng = np.random.default_rng(seed)

    _, query_ball = _BuildKdTree(points)

    N = points.shape[0]
    all_feats = []

    for r in radii:
        # 1) 邻域
        neighbors = query_ball(points, r)

        # 2) 每个measure的自适应bin边界
        edges_dict = {}
        for m in measures:
            edges_dict[m] = _AdaptiveBinEdges(points, neighbors, r, m, bins, rng, min_neigh=min_neigh)

        # 3) 每点直方图
        feat_r = np.zeros((N, len(measures) * bins), dtype=float)
        for i in range(N):
            nb = neighbors[i]
            if len(nb) < min_neigh:
                continue
            neigh_pts = points[nb]
            centroid = neigh_pts.mean(axis=0)

            col0 = 0
            for m in measures:
                edges = edges_dict[m]
                vals = np.empty(samples_per_point, dtype=float)
                for t in range(samples_per_point):
                    vals[t] = _SampleMeasure(neigh_pts, centroid, m, rng)
                hist, _ = np.histogram(vals, bins=edges)
                hist = hist.astype(float)
                s = hist.sum()
                if s > 0:
                    hist /= s
                feat_r[i, col0:col0 + bins] = hist
                col0 += bins

        all_feats.append(feat_r)

    feat = np.concatenate(all_feats, axis=1)
    return feat


# ---------------------- 示例 ----------------------
if __name__ == "__main__":
    # 假设 points 是 (N,3)
    points = np.random.rand(5000, 3).astype(float)

    radii = [0.25, 0.5, 1.0, 2.0]  # 多尺度
    feat = ComputeShapeDistributionFeatures(points, radii, bins=10, samples_per_point=255)

    print(feat.shape)  # (N, len(radii)*len(measures)*bins) = (5000, 4*5*10=200)
