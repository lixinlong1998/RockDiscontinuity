import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rock_discon_extract.geometry import Point
from rock_discon_extract.pointcloud import PointCloud
from rock_discon_extract.algorithms.ransac import RansacDetector
from rock_discon_extract.algorithms.region_growing import RegionGrowingDetector
from rock_discon_extract.pipeline import RockDiscontinuityPipeline


def main():
    # 构造一个简单的空点云占位
    points = [Point(0.0, 0.0, 0.0)]
    cloud = PointCloud(points)

    algorithms = []

    # Built-in examples with explicit parameters.
    try:
        algorithms.append(
            RansacDetector(distance_threshold=0.05, angle_threshold=10.0, min_inliers=100)
        )
    except Exception:
        pass

    try:
        algorithms.append(
            RegionGrowingDetector(
                normal_angle_threshold=10.0,
                distance_threshold=0.02,
                min_region_size=50,
            )
        )
    except Exception:
        pass

    # Auto-discover additional algorithms in package and add those with default constructors.
    try:
        import inspect
        import pkgutil
        from importlib import import_module

        from rock_discon_extract.algorithms import __path__ as _algo_pkg_path
        try:
            from rock_discon_extract.algorithms.base import (
                PlaneDetectionAlgorithm as _PDA,
            )
        except Exception:
            _PDA = None

        if _PDA is not None:
            for _, _mod_name, _ in pkgutil.iter_modules(_algo_pkg_path):
                if _mod_name in {"__init__", "base", "ransac", "region_growing"}:
                    continue
                try:
                    _mod = import_module(f"rock_discon_extract.algorithms.{_mod_name}")
                except Exception:
                    continue
                for _, _cls in inspect.getmembers(_mod, inspect.isclass):
                    try:
                        if issubclass(_cls, _PDA) and _cls is not _PDA:
                            try:
                                _inst = _cls()  # only take algorithms with default ctor
                            except TypeError:
                                continue
                            algorithms.append(_inst)
                    except Exception:
                        continue
    except Exception:
        # Best-effort discovery; ignore any issues silently in example script.
        pass

    pipeline = RockDiscontinuityPipeline(cloud, algorithms)
    results = pipeline.RunAll()

    for name, dis_list in results.items():
        print(name, "=>", len(dis_list), "discontinuities")


if __name__ == "__main__":
    main()
