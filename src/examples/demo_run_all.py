from rock_discon_extract.geometry import Point
from rock_discon_extract.pointcloud import PointCloud
from rock_discon_extract.algorithms.ransac import RansacDetector
from rock_discon_extract.algorithms.region_growing import RegionGrowingDetector
from rock_discon_extract.pipeline import RockDiscontinuityPipeline


def main():
    # 构造一个简单的空点云占位
    points = [Point(0.0, 0.0, 0.0)]
    cloud = PointCloud(points)

    algorithms = [
        RansacDetector(distance_threshold=0.05, angle_threshold=10.0, min_inliers=100),
        RegionGrowingDetector(
            normal_angle_threshold=10.0,
            distance_threshold=0.02,
            min_region_size=50
        ),
        # TODO: 其它算法同样在此添加
    ]

    pipeline = RockDiscontinuityPipeline(cloud, algorithms)
    results = pipeline.RunAll()

    for name, dis_list in results.items():
        print(name, "=>", len(dis_list), "discontinuities")


if __name__ == "__main__":
    main()
