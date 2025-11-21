import os
import sys

# 修改点 1: 将 src 加入搜索路径, 仅对当前脚本生效
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np  # 修改点 2: 用于合成简单平面点云

from src.rock_discon_extract.io_pointcloud import PointCloudIO
from src.rock_discon_extract.geometry import Point
from src.rock_discon_extract.pointcloud import PointCloud
from src.rock_discon_extract.algorithms.ransac import RansacDetector
from src.rock_discon_extract.algorithms.region_growing import RegionGrowingDetector
from src.rock_discon_extract.pipeline import RockDiscontinuityPipeline
from src.rock_discon_extract.results_exporter import ResultsExporter


def GenerateSyntheticPlanePointCloud(num_points: int = 1000) -> PointCloud:
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
    point_path = r"D:\Research\20250313_RockFractureSeg\Code\RockDiscontinuity\data\rock_data\Rock_GLS4_part1_localize_0.05m.ply"
    result_path = r"D:\Research\20250313_RockFractureSeg\Code\RockDiscontinuity\result"

    # 1) 读取真实点云
    # cloud = GenerateSyntheticPlanePointCloud(num_points=1000)    # 使用合成平面点云, 而不是单个点
    cloud = PointCloudIO.ReadPointCloudAsObjects(point_path, attach_extra_attrs=False)  # 大数据建议先关掉

    algorithms = [
        # Open3D 后端 RANSAC
        RansacDetector(
            distance_threshold=0.05,
            angle_threshold=10.0,
            min_inliers=100,
            max_iterations=1000,
            impl_mode="open3d"  # "sklearn" "manual"
        ),

        RegionGrowingDetector(
            normal_angle_threshold=10.0,
            distance_threshold=0.02,
            min_region_size=50
        ),
        # TODO: 其它算法同样在此添加
    ]

    pipeline = RockDiscontinuityPipeline(cloud, algorithms)
    results = pipeline.RunAll()
    print(results)
    disc_list = results["RANSAC-open3d"]  # 或你实际的算法名

    # 4) 导出结果
    exporter = ResultsExporter(
        point_cloud=cloud,
        discontinuities=disc_list,
        cluster_labels=None,
        algorithm_name="RANSAC_open3d"
    )

    paths = exporter.ExportAll(result_path, point_path)
    print(paths)

    for name, dis_list in results.items():
        print(name, "=>", len(dis_list), "discontinuities")
