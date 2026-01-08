from typing import List

import numpy as np
from scipy.spatial import cKDTree

from .logging_utils import LoggerManager, Timer
from .geometry import Point, BoundingBox, Discontinuity
from typing import List, Dict, Tuple, Set
import math
import numpy as np

# 尝试导入 Open3D 库
try:
    import open3d as o3d
except ImportError:
    o3d = None

'''
# Efficient RANSAC for Point-Cloud Shape Detection（仅“平面”提取工作流伪代码）
# 说明：原论文同时支持 plane/sphere/cylinder/cone/torus，这里只保留 plane 分支。
# 关键机制：局部化采样(Octree) + 懒惰评分(随机子集置信区间) + 评分含连通性(2D bitmap) + 接受后重拟合(3ε)。
# 依据：算法1、局部化采样4.3、评分4.4、连通性4.4.1、评分加速4.5、重拟合4.6。:contentReference[oaicite:0]{index=0}

输入:
    P                # 当前剩余点集
    n(p)             # 每点法向(预先估计)
    ε, α             # 距离带与法向偏差阈值（compatibility）
    β                # 连通性bitmap像元尺寸（≈采样分辨率）
    pt               # 置信阈值（论文示例用99%）
    τ                # 最小可接受形状规模（点数）
    k = 3            # 平面最小采样点数（p1,p2,p3）
    d               # Octree最大层数
    x               # level-weighting 的混合系数（更新层采样分布用）
输出:
    Ψ = {ψ1, ψ2, ...}    # 抽取到的平面集合（含各自支持点集）

------------------------------------------------------------
Preprocess():
    1) 为每个点估计法向 n(p)                                   :contentReference[oaicite:1]{index=1}
    2) 构建全局 Octree(P) 以支持“局部化采样”                     :contentReference[oaicite:2]{index=2}
    3) 将 P 随机划分为 r 个互不相交子集 {S1..Sr}（用于懒惰评分）  :contentReference[oaicite:3]{index=3}
       对每个子集 Sj 构建 Octree(Sj)（评分时只访问距形状≤ε的cell）:contentReference[oaicite:4]{index=4}
    4) 初始化层采样分布 Pl = 1/d，并初始化每层累计得分 σl = 0     :contentReference[oaicite:5]{index=5}

------------------------------------------------------------
Main():
    Ψ ← ∅
    C ← ∅                 # 候选集合
    s ← 0                 # 已生成/评估的候选计数（用于概率判据）

    repeat
        # A) newCandidates(): 局部化采样生成若干“平面候选”            :contentReference[oaicite:6]{index=6}
        for t = 1..T_batch:
            ψ ← SamplePlaneCandidate_Localized(P, Octree(P), Pl, α)
            if ψ != NULL:
                C ← C ∪ {ψ}
                s ← s + 1

        # B) bestCandidate(): 懒惰评分找当前最优候选                  :contentReference[oaicite:7]{index=7}
        m ← BestCandidate_LazyScore(C, {S1..Sr}, {Octree(Sj)}, ε, α, β)

        # C) 置信接受：若认为“没漏掉更大形状”，则抽取 m               :contentReference[oaicite:8]{index=8}
        if ProbNoBetterCandidate(|Pm|, s, k, SamplingModel="localized") > pt:
            # C1) 重拟合与最终支持集确定（3ε带，减少杂散点）            :contentReference[oaicite:9]{index=9}
            (m_refit, Pm_final) ← RefitAndFinalize(m, P, n(p), 3ε, α, β)

            # C2) 从点集删除、并清理无效候选
            P ← P \ Pm_final
            Ψ ← Ψ ∪ {m_refit}
            C ← RemoveCandidatesTouchingPoints(C, Pm_final)

            # C3) 更新层采样权重（level weighting，用累计得分σl）      :contentReference[oaicite:10]{index=10}
            Pl ← UpdateLevelDistribution(Pl, σl, x)

        end if

    until ProbNoBetterCandidate(τ, s, k, SamplingModel="localized") > pt

    return Ψ

------------------------------------------------------------
SamplePlaneCandidate_Localized(P, Octree(P), Pl, α):
    # 4.3 局部化采样：先选p1，再随机选包含p1的某层cell C，只在C内选其余点 :contentReference[oaicite:11]{index=11}
    p1 ← UniformSample(P)
    l  ← SampleLevelByDistribution(Pl)      # level weighting                       :contentReference[oaicite:12]{index=12}
    C  ← RandomCellOnLevelContainingPoint(Octree(P), l, p1)
    p2,p3 ← UniformSampleTwoDistinctPoints(C.points)

    # 平面估计：由(p1,p2,p3)拟合出 plane normal nψ 与 plane参数ψ
    ψ ← FitPlaneFrom3Points(p1,p2,p3)

    # 候选快速验证：plane法向与采样点法向偏差均 < α 才接受              :contentReference[oaicite:13]{index=13}
    if Angle(nψ, n(p1))<α AND Angle(nψ, n(p2))<α AND Angle(nψ, n(p3))<α:
        return ψ
    else:
        return NULL

------------------------------------------------------------
BestCandidate_LazyScore(C, {S1..Sr}, {Octree(Sj)}, ε, α, β):
    # 4.5 懒惰评分：只评估部分子集，利用超几何分布给出总分置信区间并逐步收敛 :contentReference[oaicite:14]{index=14}

    # 每个候选 ψ 维护：
    #   evaluated_subsets(ψ)   # 已评估到第几个子集
    #   score_on_subsets(ψ)    # 各子集得分σ_Sj(ψ)
    #   CI(ψ)=[a,b], E(ψ)      # 由超几何分布外推的置信区间与期望          :contentReference[oaicite:15]{index=15}
    #   compat_points_union(ψ) # 已发现的兼容点并集（用于连通性最大分量）    :contentReference[oaicite:16]{index=16}

    初始化：对所有 ψ∈C 先评估一个子集（如 S1），得到σ_S1(ψ)，并计算CI(ψ), E(ψ)

    while True:
        ψm ← argmax_{ψ∈C} E(ψ)               # 当前期望最高的候选            :contentReference[oaicite:17]{index=17}
        ψc ← NextCompetitorWhoseCIOverlaps(ψm, C)   # 找与ψm区间仍重叠者

        if ψc == NULL:
            return ψm                        # 区间不再重叠，ψm可判定最优

        # 对 ψm 与 ψc 再各追加评估一个子集（或只评估更“有希望”的那个）
        for ψ in {ψm, ψc}:
            j ← evaluated_subsets(ψ) + 1
            if j > r: continue
            # 评分时：用子集Octree仅遍历距plane≤ε的cells，降低计算量         :contentReference[oaicite:18]{index=18}
            σ_Sj(ψ), new_compat_points ← ScoreOnSubset_WithConnectivity(
                                            ψ, Sj, Octree(Sj), ε, α, β, fraction_x=j_points/|P|)
            compat_points_union(ψ) ← compat_points_union(ψ) ∪ new_compat_points
            UpdateCIandExpectation(ψ)        # 超几何分布外推CI与E             :contentReference[oaicite:19]{index=19}
        end for
    end while

------------------------------------------------------------
ScoreOnSubset_WithConnectivity(ψ, Sj, Octree(Sj), ε, α, β, fraction_x):
    # 4.4 评分 = ε距离带 + 法向偏差α + “最大连通分量”                  :contentReference[oaicite:20]{index=20}
    # 兼容点集合：
    #   P̂ψ = { p∈Sj | d(ψ,p)<ε AND arccos(|n(p)·n(ψ,p)|)<α }         :contentReference[oaicite:21]{index=21}
    # 其中 n(ψ,p) 是 p 投影到平面ψ处的平面法向（对平面即常量nψ）         :contentReference[oaicite:22]{index=22}
    # 连通性：
    #   Pψ = maxcomponent(ψ, P̂ψ)                                    :contentReference[oaicite:23]{index=23}

    # 子集抽样率降低时，bitmap像元应按 xβ 调整（论文明确给出）           :contentReference[oaicite:24]{index=24}
    β' ← fraction_x * β

    # 仅遍历 Octree(Sj) 中距平面≤ε的cells，收集候选兼容点               :contentReference[oaicite:25]{index=25}
    P̂ψ ← CollectCompatiblePoints(ψ, Octree(Sj), ε, α)

    # 将 P̂ψ 投影到平面2D坐标系，按像元β'栅格化，做连通域，取最大组件
    Pψ ← LargestConnectedComponent_OnPlaneBitmap(P̂ψ, ψ, β')

    return |Pψ|, P̂ψ

------------------------------------------------------------
RefitAndFinalize(m, P, n(p), 3ε, α, β):
    # 4.6 接受候选后重拟合，并把 3ε 带内兼容点纳入（减少clutter）         :contentReference[oaicite:26]{index=26}
    Q ← { p∈P | d(m,p)<3ε AND arccos(|n(p)·n(m)|)<α }   # 兼容点（更宽带）
    m_refit ← LeastSquaresFitPlane(Q)
    # 最终支持集仍建议按 4.4 的连通性规则取最大连通分量（避免跨裂隙粘连）
    P_final_hat ← { p∈P | d(m_refit,p)<ε AND angle_norm<α }
    P_final ← LargestConnectedComponent_OnPlaneBitmap(P_final_hat, m_refit, β)

    return (m_refit, P_final)

'''
def EfficientRANSAC(coords, normals, pts_indices) -> List:
    """
    功能简介:


    实现思路:
        - 在一定范围内随机生成平面上的 (x, y) 坐标;
        - z 由一个简单平面方程给出, 并叠加少量高斯噪声;
        - 将这些点包装为 Point 对象, 构造 PointCloud.

    输入:
        coords, normals = get_point_cloud_data(point_cloud)  # (N, 3)
        pts_indices = np.array(pts_indices_list, dtype=int) # 点集的全局id
    输出:
        pts_patches
            List[int],长度与pts_indices一致，记录每个点所属的局部patch_id, 0表示该点不属于任何patch.
    """
    #


    # 在 [-5, 5] 范围内随机生成 x, y
    xs = np.random.uniform(-5.0, 5.0, size=num_points)
    ys = np.random.uniform(-5.0, 5.0, size=num_points)

    # 设定一个简单平面: z = 0.2 * x + 0.1 * y + 1.0 + 噪声
    noise = np.random.normal(loc=0.0, scale=0.01, size=num_points)
    zs = 0.2 * xs + 0.1 * ys + 1.0 + noise

    points = []
    for i in range(num_points):
        p = Point(
            x=float(xs[i]),
            y=float(ys[i]),
            z=float(zs[i]),
            point_id=i
        )
        points.append(p)

    return PointCloud(points)
