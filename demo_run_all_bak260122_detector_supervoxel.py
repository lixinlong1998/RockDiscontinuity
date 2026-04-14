# coding:gbk
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # 将 src 加入搜索路径, 仅对当前脚本生效
from src.rock_discon_extract.io_pointcloud import PointCloudIO
from src.rock_discon_extract.geometry import Point
from src.rock_discon_extract.pointcloud import PointCloud, GenerateSyntheticPlanePointCloud
from src.rock_discon_extract.pipeline import RockDiscontinuityPipeline
from src.rock_discon_extract.results_exporter import ResultsExporter
from src.rock_discon_extract.visualizer import ResultsVisualizer
from src.rock_discon_extract.logging_utils import LoggerManager
from src.rock_discon_extract.algorithms.detector_ransac import RansacDetector
from src.rock_discon_extract.algorithms.detector_rg import RegionGrowingDetector
from src.rock_discon_extract.algorithms.detector_moe import MoeDetector
from src.rock_discon_extract.algorithms.detector_supervoxel import SupervoxelDetector
from src.rock_discon_extract.algorithms.cluster_dbfcm import DBFCMCluster

result_path = r"D:\Research\20250313_RockFractureSeg\Code\RockDiscontinuity\result"

point_path_list = [
    r"D:\Research\20250313_RockFractureSeg\Code\RockDiscontinuity\data\rock_data\Rock_GLS4_part1_localize_0.05m.ply",
]

# 指定使用的算法和参数
detector_algorithms = [
    # 强风化岩体
    SupervoxelDetector(
        voxel_size=1.5,
        sample_spacing=0.05,
        ransac_distance=0.03,
        ransac_angle=10,
        min_plane_points=300,
        edge_distance=0.1,
        edge_angle=15,
        min_edge_points=50,  # 剩余点低于这个数, 不做边缘提取
        min_edge_patch_points=50,  # 作为edge-patch的最少点
        super_distance=0.20,
        super_angle=20.0,
        max_refit_error=8,
        distance_step=0.01,
        angle_step=1.0,
        patch_distance=0.3,
        patch_angle=20
    )
    # TODO: 其它检测算法同样在此添加
]

cluster_algorithms = [
    DBFCMCluster(
        rmse_weight=0.4,
        length_weight=0.6,
        first_layer_ratio=0.6,
        dbscan_eps_deg=10,
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

plots_list = [
    "dipdir_rose",  # 倾向玫瑰图
    "stereonet_kde",  # stereonet + KDE + 聚类表
    "distributions_combined",  # 合并:面积,迹长,粗糙度分布图
    "slope_cluster_stereonet",  # 优势结构面+整体坡面 stereonet
    # "area_distribution",  # 面积分布图
    # "trace_length_distribution",  # 迹长分布图
    # "roughness_distribution",  # 粗糙度分布图
]

if __name__ == "__main__":
    for point_path in point_path_list:
        # 0) 创建log文件
        LoggerManager.CreatLogFile(result_path)

        # 1) 读取真实点云
        # point_cloud = GenerateSyntheticPlanePointCloud(num_points=1000)    # 使用合成平面点云
        point_cloud = PointCloudIO.ReadPointCloudAsObjects(point_path, attach_extra_attrs=False)  # 大数据建议先关掉

        # 【可选】2) 估计法向与曲率, 便于后续可视化(CloudCompare 曲率/法向热力图等)
        point_cloud.EstimateNormals(k_neighbor=30, est_normals=True, est_curvature=True)  # 若输入 PLY 已带法向, 可以暂时注释掉本行

        # 【可选】3) 人工给定的结构面 dip/dir 列表, 每项为 (dip_deg, dip_dir_deg), 用于作为额外初始簇中心。
        manual_dip_dirs = []

        # 5) 执行pipeline
        pipeline = RockDiscontinuityPipeline(point_cloud, detector_algorithms, cluster_algorithms, manual_dip_dirs)
        results = pipeline.RunAll()  # (discontinuities, clusters, parameters)
        # print(results)

        # 6) 导出结果
        paths_list = []
        for algorithm_name, result in results.items():
            discon_algoname, cluster_algoname = algorithm_name
            discon_list, cluster_list, parameters_dict = result
            print(discon_algoname, "=>", len(discon_list), "discontinuities")
            print(cluster_algoname, "=>", len(cluster_list), "clusters")
            exporter = ResultsExporter(
                point_cloud=point_cloud,
                discontinuities=discon_list,
                clusters=cluster_list,
                algorithm_name=algorithm_name[0],
                parameters=parameters_dict,
            )
            paths = exporter.ExportAll(result_path, point_path)
            # print(paths)
            paths_list.append((paths["dir"], paths["point_cloud_path"]))

        # 7) 结果可视化
        vis = ResultsVisualizer(paths_list=paths_list)
        # 单一结果分析 + 批处理: 每个 out_dir 下各出一张 dipdir 玫瑰图
        vis.ExportAllSingleAnalysis(plots_name=plots_list, output_formats=("png", "svg"), show=False)
        # 多结果对比分析 + 批处理: 对同一 basename 下的多个结果叠加对比
        # vis.ExportAllCompareAnalysis(plots_name=plots_list, output_formats=("png", "svg"), show=False)
