import os
import sys

# 修改点 1: 将 src 加入搜索路径, 仅对当前脚本生效
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np  # 修改点 2: 用于合成简单平面点云

from src.rock_discon_extract.io_pointcloud import PointCloudIO
from src.rock_discon_extract.geometry import Point
from src.rock_discon_extract.pointcloud import PointCloud
from src.rock_discon_extract.pipeline import RockDiscontinuityPipeline
from src.rock_discon_extract.results_exporter import ResultsExporter
from src.rock_discon_extract.visualizer import ResultsVisualizer
from src.rock_discon_extract.logging_utils import LoggerManager

from src.rock_discon_extract.algorithms.ransac import RansacDetector
from src.rock_discon_extract.algorithms.region_growing import RegionGrowingDetector
from src.rock_discon_extract.algorithms.moe import MoeDetector
from src.rock_discon_extract.algorithms.supervoxel import SupervoxelDetector
from src.rock_discon_extract.algorithms.dbfcm_cluster import DBFCMCluster


def GenerateSyntheticPlanePointCloud(num_points=1000) -> PointCloud:
    """
    功能简介:
        生成一个简单的人工平面点云, 用于测试 RANSAC 算法是否正常工作.

    实现思路:
        - 在一定范围内随机生成平面上的 (x, y) 坐标;
        - z 由一个简单平面方程给出, 并叠加少量高斯噪声;
        - 将这些点包装为 Point 对象, 构造 PointCloud.

    输入:
        num_points: int
            生成的点数量.

    输出:
        point_cloud: PointCloud
            含有 num_points 个近似共面的点云对象.
    """
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


if __name__ == "__main__":
    # point_path = r"E:\Database\_RockPoints\TSDK_Rockfall_RegularClip\TSDK_Rockfall_1_P1_0.05m.ply"
    point_path = r"D:\Research\20250313_RockFractureSeg\Code\RockDiscontinuity\data\rock_data\Rock_GLS4_part1_localize_0.05m.ply"
    result_path = r"D:\Research\20250313_RockFractureSeg\Code\RockDiscontinuity\result"

    # 0) 创建log文件
    LoggerManager.CreatLogFile(result_path)

    # 1) 读取真实点云
    # cloud = GenerateSyntheticPlanePointCloud(num_points=1000)    # 使用合成平面点云, 而不是单个点
    cloud = PointCloudIO.ReadPointCloudAsObjects(point_path, attach_extra_attrs=False)  # 大数据建议先关掉

    # 【可选】2) 估计法向与曲率, 便于后续可视化(CloudCompare 曲率/法向热力图等)
    cloud.EstimateNormals(k_neighbor=30, est_normals=False, est_curvature=True)  # 若输入 PLY 已带法向, 可以暂时注释掉本行

    # 【可选】3) 人工给定的结构面 dip/dir 列表, 每项为 (dip_deg, dip_dir_deg), 用于作为额外初始簇中心。
    manual_dip_dirs = []

    # 4) 指定使用的算法和参数
    detector_algorithms = [
        # # Open3D 后端 RANSAC
        # RansacDetector(
        #     distance_threshold=0.25,
        #     angle_threshold=15,
        #     min_inliers=100,
        #     max_iterations=1000,
        #     impl_mode="open3d"  # "sklearn" "manual"
        # ),
        #
        # RegionGrowingDetector(
        #     normal_angle_threshold=15.0,
        #     distance_threshold=0.25,
        #     min_region_size=100
        # ),

        # MoeDetector(
        #     voxel_size=0.5,
        #     num_major_orientations=0,
        #     Sa=30.0,
        #     epsilon=0.05,
        #     min_points_big_voxel=300,
        #     min_points_sub_voxel=150,
        #     t1=10.0,
        #     t2=15.0,
        #     t3=None,
        #     max_distance=0.3
        # ),

        # 弱风化岩体
        SupervoxelDetector(
            voxel_size=1,  # 基于点云平均间距实际情况,不要过大
            ransac_distance=0.2,  # 需要足够精确,但需要顾及测量误差
            min_plane_points=30,  # RANSAC后的内点数只要大于此值则认为是成立的,应该取决于voxel内的平均点数
            edge_distance=0.2,  # edge拼接,需要和平面估计同样严格
            edge_angle=10,  # edge拼接,需要和平面估计同样严格
            min_edge_points=30,  # 启动edge patch detect的最小点数
            min_edge_patch_points=25,  # edge patch detect后的内点数
            super_distance=0.3,  # 吸纳周围散点的初始阈值
            super_angle=30,  # 吸纳周围散点的初始阈值
            max_refit_error=5.0,  # 吸纳周围散点时至多可接受的误差
            distance_step=0.01,  # 2个step保持同步
            angle_step=1.0,  # 2个step保持同步
            patch_distance=0.25,
            patch_angle=15
        ),

        # 强风化岩体
        # SupervoxelDetector(
        #     voxel_size=2,  # 基于点云平均间距实际情况,不要过大
        #     ransac_distance=0.25,  # 需要足够精确,但需要顾及测量误差
        #     min_plane_points=30,  # RANSAC后的内点数只要大于此值则认为是成立的,应该取决于voxel内的平均点数
        #     edge_distance=0.25,  # edge拼接,需要和平面估计同样严格
        #     edge_angle=15,  # edge拼接,需要和平面估计同样严格
        #     min_edge_points=30,  # 启动edge patch detect的最小点数
        #     min_edge_patch_points=15,  # edge patch detect后的内点数
        #     super_distance=0.3,  # 吸纳周围散点的初始阈值
        #     super_angle=30,  # 吸纳周围散点的初始阈值
        #     max_refit_error=5.0,  # 吸纳周围散点时至多可接受的误差
        #     distance_step=0.01,  # 2个step保持同步
        #     angle_step=1.0,  # 2个step保持同步
        #     patch_distance=0.25,
        #     patch_angle=20
        # ),
        # TODO: 其它检测算法同样在此添加
    ]

    cluster_algorithms = [
        DBFCMCluster(
            rmse_weight=0.4,
            length_weight=0.6,
            first_layer_ratio=0.6,
            dbscan_eps_deg=10.0,
            dbscan_min_samples_ratio=0.05,
            quality_mode="piecewise",
            quality_gamma=1.5,
            quality_min_weight=0.1,
            low_quality_ratio=0.2,
            rmse_gamma=1.0,
            fcm_m=2.0,
            fcm_max_iter=100,
            fcm_tol_deg=1e-2,
            noise_membership_threshold=0.4,
            merge_angle_eps_deg=5.0,
            max_merge_loops=3,
            enable_plot=False
        ),
        # TODO: 其它聚类算法同样在此添加
    ]

    # 5) 执行pipeline
    pipeline = RockDiscontinuityPipeline(cloud, detector_algorithms, cluster_algorithms, manual_dip_dirs)
    results = pipeline.RunAll()
    # print(results)

    # 6) 导出结果
    paths_list = []
    for algorithm_name, result in results.items():
        discon_algoname, cluster_algoname = algorithm_name
        discon_list, cluster_list = result
        print(discon_algoname, "=>", len(discon_list), "discontinuities")
        print(cluster_algoname, "=>", len(cluster_list), "clusters")
        exporter = ResultsExporter(
            point_cloud=cloud,
            discontinuities=discon_list,
            clusters=cluster_list,
            algorithm_name=algorithm_name[0]
        )
        paths = exporter.ExportAll(result_path, point_path)
        # print(paths)
        paths_list.append((paths["dir"], paths["point_cloud_path"]))

    # 7) 结果可视化
    plots_list = [
        "dipdir_rose",  # 倾向玫瑰图
        "stereonet_kde",  # stereonet + KDE + 聚类表
        "distributions_combined",  # 三个分布图合并
        "slope_cluster_stereonet",  # 优势结构面+整体坡面 stereonet
        # "area_distribution",  # 面积分布图
        # "trace_length_distribution",  # 迹长分布图
        # "roughness_distribution",  # 粗糙度分布图
    ]
    vis = ResultsVisualizer(paths_list=paths_list)
    # 单一结果分析 + 批处理: 每个 out_dir 下各出一张 dipdir 玫瑰图
    vis.ExportAllSingleAnalysis(plots_name=plots_list, output_formats=("png", "svg"), show=False)
    # 多结果对比分析 + 批处理: 对同一 basename 下的多个结果叠加对比
    # vis.ExportAllCompareAnalysis(plots_name=plots_list, output_formats=("png", "svg"), show=False)
