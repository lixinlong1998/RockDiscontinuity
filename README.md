# RockDiscontinuity
Rock discontinuity extraction with multi-methods

##  增加算法参数导出功能
#### 文件结构如下:
- RockDiscontinuity\src\rock_discon_extract\algorithms\detector_supervoxel.py
- RockDiscontinuity\src\rock_discon_extract\algorithms\base.py
- RockDiscontinuity\src\rock_discon_extract\results_exporter.py
- RockDiscontinuity\src\rock_discon_extract\pipeline.py
- RockDiscontinuity\src\rock_discon_extract\visualizer.py
- RockDiscontinuity\src\rock_discon_extract\base.py
- RockDiscontinuity\demo_run_all_bak260122_detector_supervoxel.py
#### 为了能够导出每次运行的算法参数,你需要:
- 补齐base.py中PlaneDetectionAlgorithm对象和ClusteringAlgorithm对象的get_parameters(self,)方法
- 确认pipeline.py下RunAll()中的parameters = detect_algo.get_parameters() + cluster_algo.get_parameters()是否符合字典拼接语法
- 补齐results_exporter.py中ResultsExporter对象的ExportParametersJson(parameters_json)方法,并检查这个方法在ExportAll()方法中的调用是否符合语法

## supervoxel debug 可视化调整需求清单: 
bak260122_detector_supervoxel_bitmap_vb3.py是基于之前可视化修改后的代码,可以正常运行, 在此基础上,根据下列调整需求清单做修改(代码中已经实现的部分可以做调整和结构优化)
#### 参考results_exporter.py中的def ExportPointLevelCsv()函数导出下列可视化赋色点云,赋予的颜色RGB存放在DR,DG,DB中,这样不会覆盖点云原来的颜色,其余未提及的参数可以用整数0占位
- step2 voxel-patch的提取情况, 用基于ID的可复现彩色(排除纯红,纯绿,纯蓝)表示voxel-patch点集, 剩余点标记为灰色
- step2 voxel-patch的提取情况, 用纯绿色(便于叠加后续步骤结构)表示voxel-patch点集, 剩余点标记为灰色
- step3 edge-patch的提取情况, 用纯蓝色表示edge-patch点集, 如果edge-path被拼接到最近的voxel-patch,则标记为对应于voxel-patch的颜色, 剩余点标记为灰色
- step3 edge-patch的提取情况, 保留"纯绿色表示voxel-patch点集",用纯蓝色表示edge-patch点集, 如果edge-path被拼接到最近的voxel-patch,则标记为蓝绿色,剩余点标记为灰色
- step4 超体素分割后, 被吸纳的点标记为对应于voxel-patch的颜色, 剩余点标记为灰色
- step4 超体素分割后, 被吸纳的点标记为纯红色, 保留"纯绿色表示voxel-patch点集,纯蓝色表示edge-patch点集,蓝绿色表示拼接走的edge-patch点集", 剩余点标记为灰色
- step4 超体素分割后, voxel-patch沿用原来基于ID的可复现彩色,给edge-patch顺着voxel-patch的id继续往下排并基于ID赋予可复现彩色, 剩余点标记为灰色
#### 关键参数信息表csv导出,后续我可以在excel中打开自行检查
- Step2 每个voxel内:voxel-patch点的"法向\曲率\到平面距离"分布指标, 其余点的"法向\曲率\到平面距离"分布指标, 2者分布指标的差异.
- Step3 “edge-patch 为什么没拼接”阈值诊断表导出（到底被 edge_distance 还是 edge_angle 拦截）
- Step4 “supervoxel 生长阈值收缩/成功情况摘要表导出”（每个 seed 的最终阈值、迭代次数、吸纳点数、是否因 max_refit_error 卡住）。
- Step5 Patch-based 区域生长时,相邻patch之间的的阈值诊断表(到底被 patch_distance 还是 patch_angle 拦截)
#### 参考curvature_filter_voxel_export.py中大概第179-192行代码,调用visualizer.py来绘制过程中的信息:
- 在step2 为每个voxel绘制其内部点云的dipdir_rose 与 stereonet_kde,输出到一个单独的文件夹中.
#### 给出脚本的文件存放地址如下, 同时部分给出了相关代码段落便于你快速定位:
- RockDiscontinuity\src\rock_discon_extract\algorithms\bak260122_detector_supervoxel_bitmap_vb3.py    [lines:all]
- RockDiscontinuity\src\rock_discon_extract\results_exporter.py   [lines:,17-158;202-310]
- RockDiscontinuity\curvature_filter_voxel_export.py  [lines:179-192]
- RockDiscontinuity\src\rock_discon_extract\visualizer.py [lines:26-155]