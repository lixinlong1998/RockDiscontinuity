import os
from typing import Dict, List, Optional, Sequence, Tuple, Callable

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import gridspec
from .logging_utils import LoggerManager, Timer

PathTuple = Tuple[str, str]  # (out_dir, point_path)

# ==== 关键修复：设置中文字体，按优先级尝试多种字体 ====

# 设置字体族和字体
# 设置全局字体为Times New Roman
mpl.rcParams["font.family"] = "Times New Roman"
# 告诉Matplotlib在渲染时优先使用这些字体
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'Times New Roman']  # 添加SimHei作为中文字体
# 解决负号'-'显示为方块的问题[citation:1][citation:10]
mpl.rcParams['axes.unicode_minus'] = False


class ResultsVisualizer:
    """
    功能简介:
        针对 ResultsExporter 导出的结果(点级/结构面级 CSV 等),
        提供批量的可视化分析接口, 支持:
            1) 单一结果分析 + 批处理: 对每个 (out_dir, point_path) 各自绘制图件;
            2) 多结果对比分析 + 批处理: 对同一数据集的多组结果(如 RANSAC vs RG)
               进行对比绘图。

    实现思路(概要):
        - 在构造函数中接收 paths_list=[(out_dir, point_path), ...],
          其中 out_dir 必须是 ResultsExporter._CreateOutputSubdir 创建的子目录,
          point_path 为对应的输入点云路径(用于确定 basename);
        - 内部维护两张“图件注册表”:
            * single_plot_meta: 单一结果分析绘图函数的注册表;
            * compare_plot_meta: 多结果对比分析绘图函数的注册表;
          映射关系:
            plot_name -> {"func": callable, "short_name": str}
          其中 short_name 用于输出文件名中的“图件简称”;
        - 提供两个统一入口:
            * ExportAllSingleAnalysis(plots_name, output_formats, show)
                对 paths_list 中的每一个 (out_dir, point_path),
                针对 plots_name 中指定的图件逐一绘制;
            * ExportAllCompareAnalysis(plots_name, output_formats, show)
                先按 point_path 的 basename 分组, 对每个数据集内部的多组结果
                (多个 out_dir) 执行对比绘图。
    """

    def __init__(self, paths_list: Sequence[PathTuple]):
        """
        功能简介:
            初始化 ResultsVisualizer, 记录所有结果路径对, 并建立绘图函数注册表。

        实现思路:
            - 将传入的 paths_list 拷贝为内部 list, 并做基础校验;
            - 创建 logger 用于记录信息;
            - 调用 _InitPlotRegistry 注册当前已实现的图件类型。
        输入:
            paths_list: Sequence[Tuple[str, str]]
                每个元素为 (out_dir, point_path):
                    out_dir: ResultsExporter.ExportAll 返回的 paths["dir"];
                    point_path: 对应的输入点云路径。
        输出:
            无显式返回, 但内部保存 self.paths_list。
        """
        self.logger = LoggerManager.GetLogger(self.__class__.__name__)
        self.paths_list: List[PathTuple] = list(paths_list)
        self.dpi = 300
        self.font_family = "Times New Roman"
        self._ValidatePathsList()
        self._InitPlotRegistry()

    # ============================================================
    # 公共入口: 单一结果分析 + 批处理
    # ============================================================

    def ExportAllSingleAnalysis(
            self,
            plots_name: Sequence[str],
            output_formats: Sequence[str] = ("png", "svg"),
            show: bool = False,
    ) -> None:
        """
        功能简介:
            针对 paths_list 中的每一个 (out_dir, point_path),
            执行“单一结果”的批量可视化分析。即:
                for (out_dir, point_path) in paths_list:
                    对 plots_name 中指定的每种图件, 各自绘制一张图,
                    并存储到 out_dir 下, 文件名格式为:
                        <basename>_图件简称.扩展名
                    其中 basename = point_path 的文件名去扩展名。

        实现思路:
            1) 将 plots_name 归一化为 list, 并检查是否为空;
            2) 将 output_formats 归一化为 ['.png', '.svg'] 形式;
            3) 对每个 path_tuple=(out_dir, point_path):
                 - 对每个 plot_name:
                     * 在 single_plot_meta 中查找对应绘图函数及 short_name;
                     * 若不存在, 记录 warning 并跳过;
                     * 调用对应绘图函数, 绘制并保存图件。
        输入:
            plots_name: Sequence[str]
                需要输出的图件名称列表, 例如 ["dipdir_rose"]。
            output_formats: Sequence[str]
                输出文件格式列表, 支持 "png"、"svg" 等,
                可写为 "png"/".png" 混用, 内部统一加上 "."。
            show: bool
                是否在绘制完成后调用 plt.show() 显示图像。
                在批处理脚本中通常设为 False。
        输出:
            无显式返回。图件将写入到各自的 out_dir 目录中。
        """
        plot_names = list(plots_name)
        if not plot_names:
            self.logger.warning("ExportAllSingleAnalysis: plots_name 为空, 不执行绘图。")
            return

        formats = self._NormalizeFormats(output_formats)

        with Timer(
                f"ExportAllSingleAnalysis(n_paths={len(self.paths_list)}, plots={plot_names})",
                self.logger,
        ):
            for out_dir, point_path in self.paths_list:
                base_name = self._GetBaseName(point_path)
                for plot_name in plot_names:
                    meta = self.single_plot_meta.get(plot_name)
                    if meta is None:
                        self.logger.warning(
                            f"ExportAllSingleAnalysis: 未注册的单一分析图件 '{plot_name}', 跳过。"
                        )
                        continue

                    func: Callable = meta["func"]
                    short_name: str = meta["short_name"]

                    try:
                        func(
                            out_dir=out_dir,
                            point_path=point_path,
                            base_name=base_name,
                            plot_short_name=short_name,
                            output_formats=formats,
                            show=show,
                        )
                    except Exception as e:
                        self.logger.warning(
                            f"ExportAllSingleAnalysis: 绘制 '{plot_name}' "
                            f"失败, out_dir={out_dir}, point_path={point_path}, 错误: {e}"
                        )

    # ============================================================
    # 公共入口: 多结果对比分析 + 批处理
    # ============================================================

    def ExportAllCompareAnalysis(
            self,
            plots_name: Sequence[str],
            output_formats: Sequence[str] = ("png", "svg"),
            show: bool = False,
    ) -> None:
        """
        功能简介:
            按 point_path 的 basename 对 paths_list 分组, 对同一数据集下的
            多组结果(out_dir 不同, 代表不同算法/参数)进行“多结果对比”的
            批量可视化分析。

            例如:
                paths_list = [
                    (out_dir_RANSAC, point_A),
                    (out_dir_RG,     point_A),
                    (out_dir_RANSAC, point_B),
                    (out_dir_RG,     point_B),
                ]
            则对比分析时:
                - 对 A 组的 (out_dir_RANSAC, point_A) 与 (out_dir_RG, point_A)
                  一起绘制对比图, 存到这两个 out_dir 中;
                - 对 B 组同理。

        实现思路:
            1) 将 plots_name 归一化为 list, 并检查是否为空;
            2) 将 output_formats 归一化为 ['.png', '.svg'] 形式;
            3) 调用 _GroupPathsByBaseName 按 basename 分组;
            4) 对每个 basename 对应的 paths_group:
                 - 对每个 plot_name:
                     * 在 compare_plot_meta 中查找绘图函数及 short_name;
                     * 若不存在, 记录 warning 并跳过;
                     * 调用绘图函数, 在一张图内叠加该组中所有结果。
        输入:
            plots_name: Sequence[str]
                需要输出的图件名称列表, 例如 ["dipdir_rose"]。
            output_formats: Sequence[str]
                输出文件格式列表, 支持 "png"、"svg" 等。
            show: bool
                是否在绘制完成后调用 plt.show() 显示图像。
        输出:
            无显式返回。对比图将写入每个分组中所有 out_dir 目录。
        """
        plot_names = list(plots_name)
        if not plot_names:
            self.logger.warning("ExportAllCompareAnalysis: plots_name 为空, 不执行绘图。")
            return

        formats = self._NormalizeFormats(output_formats)

        groups = self._GroupPathsByBaseName()
        if not groups:
            self.logger.warning("ExportAllCompareAnalysis: 按 basename 分组后为空, 不执行绘图。")
            return

        with Timer(
                f"ExportAllCompareAnalysis(n_groups={len(groups)}, plots={plot_names})",
                self.logger,
        ):
            for base_name, paths_group in groups.items():
                for plot_name in plot_names:
                    meta = self.compare_plot_meta.get(plot_name)
                    if meta is None:
                        self.logger.warning(
                            f"ExportAllCompareAnalysis: 未注册的对比分析图件 '{plot_name}', 跳过。"
                        )
                        continue

                    func: Callable = meta["func"]
                    short_name: str = meta["short_name"]

                    try:
                        func(
                            base_name=base_name,
                            paths_group=paths_group,
                            plot_short_name=short_name,
                            output_formats=formats,
                            show=show,
                        )
                    except Exception as e:
                        self.logger.warning(
                            f"ExportAllCompareAnalysis: 绘制 '{plot_name}' "
                            f"失败, base_name={base_name}, 错误: {e}"
                        )

    # ============================================================
    # 注册表与内部工具函数
    # ============================================================

    def _InitPlotRegistry(self) -> None:
        """
        功能简介:
            初始化图件注册表, 将具体的绘图函数绑定到 plot_name 上。

        实现思路:
            - single_plot_meta: 单一结果分析图件注册表;
            - compare_plot_meta: 多结果对比图件注册表。
            当前已注册:
                plot_name="dipdir_rose":
                    * 单一分析: _PlotDipDirectionRoseSingle, short_name="dipdir_rose";
                    * 对比分析: _PlotDipDirectionRoseCompare, short_name="dipdir_rose_cmp"。
        输入:
            无。
        输出:
            无, 但设置 self.single_plot_meta / self.compare_plot_meta。
        """
        self.single_plot_meta: Dict[str, Dict[str, Callable]] = {
            "dipdir_rose": {
                "func": self._PlotDipDirectionRoseSingle,
                "short_name": "dipdir_rose",
            },
            "stereonet_kde": {
                "func": self._PlotStereonetKdeSingle,
                "short_name": "stereonet_kde",
            },
            "area_distribution": {
                "func": self._PlotAreaDistributionSingle,
                "short_name": "area_distribution",
            },
            "trace_length_distribution": {
                "func": self._PlotTraceLengthDistributionSingle,
                "short_name": "trace_length_distribution",
            },
            "roughness_distribution": {
                "func": self._PlotRoughnessDistributionSingle,
                "short_name": "roughness_distribution",
            },
            "distributions_combined": {  # 新增：三个分布图合并
                "func": self._PlotDistributionsSingle,
                "short_name": "distributions_combined",
            },
            "slope_cluster_stereonet": {
                "func": self._PlotSlopeClusterStereonetSingle,
                "short_name": "slope_cluster_stereonet",
            },
        }

        self.compare_plot_meta: Dict[str, Dict[str, Callable]] = {
            "dipdir_rose": {
                "func": self._PlotDipDirectionRoseCompare,
                "short_name": "dipdir_rose_cmp",
            },
        }

    def _ValidatePathsList(self) -> None:
        """
        功能简介:
            对传入的 paths_list 做基础合法性检查。

        实现思路:
            - 检查每个元素是否为长度为 2 的 tuple 或 list;
            - 不检查目录/文件是否真实存在, 具体检查在绘图时执行。
        输入:
            无(使用 self.paths_list)。
        输出:
            无, 如遇明显错误则抛异常。
        """
        for idx, item in enumerate(self.paths_list):
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ValueError(
                    f"paths_list 第 {idx} 项格式错误, 期望 (out_dir, point_path), "
                    f"实际为: {item}"
                )

    @staticmethod
    def _GetBaseName(point_path: str) -> str:
        return os.path.splitext(os.path.basename(point_path))[0]

    @staticmethod
    def _NormalizeFormats(output_formats: Sequence[str]) -> List[str]:
        """
        功能简介:
            将输出格式列表统一转换为形如 [".png", ".svg"] 的形式。

        实现思路:
            - 对每个格式字符串:
                * 去除前后空白;
                * 若为空则跳过;
                * 若不以 "." 开头则自动加上;
            - 去重后返回。
        输入:
            output_formats: Sequence[str]
                例如 ("png", "svg") 或 (".png", ".svg")。
        输出:
            formats: List[str]
                形如 [".png", ".svg"] 的列表。
        """
        formats: List[str] = []
        for fmt in output_formats:
            s = str(fmt).strip()
            if not s:
                continue
            if not s.startswith("."):
                s = "." + s
            if s not in formats:
                formats.append(s)
        if not formats:
            # 默认至少给一个 png
            formats = [".png"]
        return formats

    def _GroupPathsByBaseName(self) -> Dict[str, List[PathTuple]]:
        """
        功能简介:
            将 self.paths_list 按 point_path 的 basename 分组,
            用于多结果对比分析。

        实现思路:
            - 对每个 (out_dir, point_path):
                * 计算 base_name;
                * 将该二元组追加到 groups[base_name] 列表中。
        输入:
            无(使用 self.paths_list)。
        输出:
            groups: Dict[str, List[PathTuple]]
                key 为 base_name, value 为该数据集下的 (out_dir, point_path) 列表。
        """
        groups: Dict[str, List[PathTuple]] = {}
        for out_dir, point_path in self.paths_list:
            base_name = self._GetBaseName(point_path)
            groups.setdefault(base_name, []).append((out_dir, point_path))
        return groups

    @staticmethod
    def _BuildDiscCsvPath(out_dir: str, base_name: str) -> str:
        return os.path.join(out_dir, f"{base_name}_discontinuitys.csv")

    @staticmethod
    def _BuildClusterCsvPath(out_dir: str, base_name: str) -> str:
        return os.path.join(out_dir, f"{base_name}_clusters.csv")

    @staticmethod
    def _BuildPointsCsvPath(out_dir: str, base_name: str) -> str:
        return os.path.join(out_dir, f"{base_name}_points.csv")

    @staticmethod
    def _ExtractAlgorithmTagFromOutDir(out_dir: str) -> str:
        """
        功能简介:
            从 ResultsExporter._CreateOutputSubdir 生成的 out_dir 中,
            解析出算法名称片段, 用于对比图的图例标签。

        实现思路:
            - out_dir 末级目录名格式约为: "YYYYMMDD_HHMMSS_算法名";
            - 取 os.path.basename(out_dir) 得到目录名;
            - 找到第一个下划线, 将其后的部分作为算法名字符串返回;
            - 若解析失败, 则返回整个目录名。
        输入:
            out_dir: str
                结果输出子目录。
        输出:
            algo_tag: str
                解析得到的算法标签。
        """
        name = os.path.basename(os.path.normpath(out_dir))
        idx = name.find("_")
        if idx >= 0 and idx + 1 < len(name):
            return name[idx + 1:]
        return name

    @staticmethod
    def _SaveFigure(
            fig: plt.Figure,
            fig_dpi: int,
            out_dirs: Sequence[str],
            base_name: str,
            plot_short_name: str,
            output_formats: Sequence[str],
            logger,

    ) -> None:
        """
        功能简介:
            将给定 Figure 对象以多种格式保存到多个 out_dir 中。

        实现思路:
            - 对每个 out_dir:
                * 确保目录存在;
                * 对每个扩展名 fmt:
                    - 构造文件名: <base_name>_<plot_short_name><fmt>;
                    - 调用 fig.savefig() 保存。
        输入:
            fig: matplotlib.figure.Figure
                需要保存的图像对象。
            fig_dpi: int
                设置保存分辨率
            out_dirs: Sequence[str]
                要保存到的目录列表。
            base_name: str
                点云 basename(不含扩展名)。
            plot_short_name: str
                图件简称, 将出现在文件名中。
            output_formats: Sequence[str]
                已统一为 ".png" 等形式的扩展名列表。
            logger:
                Logger 对象, 用于记录保存信息。
        输出:
            无显式返回。
        """
        for out_dir in out_dirs:
            os.makedirs(out_dir, exist_ok=True)
            for fmt in output_formats:
                filename = f"{base_name}_{plot_short_name}{fmt}"
                save_path = os.path.join(out_dir, filename)
                fig.savefig(save_path, dpi=fig_dpi)
                logger.info(f"图像已保存: {save_path}")

    # ============================================================
    # 图件实现 1: Dip Direction Rose (单一结果 + 对比结果)
    # ============================================================

    def _PlotDipDirectionRoseSingle(
            self,
            out_dir: str,
            point_path: str,
            base_name: str,
            plot_short_name: str,
            output_formats: Sequence[str],
            show: bool,
    ) -> None:
        """
        功能简介:
            对单一结果(out_dir, point_path)绘制倾向(dip direction)玫瑰图.
            保持比例协调，字体使用Times New Roman。
        """

        disc_csv_path = self._BuildDiscCsvPath(out_dir, base_name)
        if not os.path.isfile(disc_csv_path):
            self.logger.warning(f"未找到结构面 CSV: {disc_csv_path}")
            return

        try:
            df = pd.read_csv(disc_csv_path)
        except Exception as e:
            self.logger.warning(f"读取 CSV 失败: {disc_csv_path}, 错误: {e}")
            return

        if "Dipdir" not in df.columns:
            self.logger.warning(f"CSV 缺少列 'Dipdir', 跳过绘图")
            return

        dipdir_deg = df["Dipdir"].to_numpy(dtype=float)
        dipdir_deg = np.mod(dipdir_deg, 360.0)

        if dipdir_deg.size == 0:
            self.logger.warning(f"Dipdir 数量为 0, 跳过绘图")
            return

        with Timer(f"_PlotDipDirectionRoseSingle(base={base_name})", self.logger):
            # 2. 创建图形，调整尺寸和DPI，为文字留出更多空间
            fig = plt.figure(figsize=(10, 8.5), dpi=300)  # 调整为更协调的尺寸
            # 使用GridSpec精细控制绘图区和边距
            gs = fig.add_gridspec(1, 1, left=0.12, right=0.88, top=0.90, bottom=0.12)
            ax = fig.add_subplot(gs[0, 0], polar=True)

            # 分箱设置
            bin_size_deg = 10.0
            edges_deg = np.arange(0.0, 360.0 + bin_size_deg, bin_size_deg)
            width_rad = np.deg2rad(bin_size_deg)

            hist, _ = np.histogram(dipdir_deg, bins=edges_deg)
            centers_deg = edges_deg[:-1] + bin_size_deg * 0.5
            angles_rad = np.deg2rad(centers_deg)

            # 3. 绘制玫瑰图 - 优化颜色和边框
            bars = ax.bar(
                angles_rad,
                hist,
                width=width_rad,
                bottom=0.0,
                alpha=0.85,  # 提高透明度，使图形更清晰
                color='#2E86AB',  # 使用更专业的蓝色系
                edgecolor='black',  # 明确的黑色边框
                linewidth=0.8,
                align='center',
                zorder=2  # 确保柱状图在网格线之上
            )

            # 4. 设置极坐标参数
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)

            # 5. 优化刻度标签 - 显著增大并加粗
            theta_ticks = np.arange(0, 360, 30)
            # 方位标注：使用更专业的N, E, S, W
            theta_labels = ['N', '30°', '60°', 'E', '120°', '150°',
                            'S', '210°', '240°', 'W', '300°', '330°']

            # 设置角度网格和标签
            ax.set_thetagrids(theta_ticks, labels=theta_labels,
                              fontsize=11, weight='bold')  # 增大并加粗标签

            # 角度网格线样式
            ax.grid(axis='x', which='major', linestyle='-',
                    linewidth=0.7, alpha=0.7, color='gray')

            # 6. 优化半径（频率）轴
            max_hist = np.max(hist)
            # 智能确定刻度间隔
            if max_hist <= 5:
                r_step = 1
            elif max_hist <= 20:
                r_step = max(2, int(max_hist / 5))
            else:
                r_step = max(5, int(max_hist / 5))

            r_ticks = np.arange(0, max_hist + r_step, r_step)
            r_labels = [f'{int(t)}' for t in r_ticks]

            # 设置半径网格和标签
            ax.set_rgrids(r_ticks, labels=r_labels, angle=22.5,
                          fontsize=10, weight='bold')  # 增大并加粗标签

            # 半径网格线样式
            ax.grid(axis='y', which='major', linestyle='--',
                    linewidth=0.5, alpha=0.5, color='gray')

            # 设置半径轴标签 - 位置和样式优化
            ax.set_ylabel('Frequency', fontsize=12, weight='bold', labelpad=20)
            # 将ylabel移到更合适的位置
            ax.yaxis.set_label_coords(-0.15, 0.5)

            # 7. 设置主标题 - 提升视觉权重
            ax.set_title(
                f'Dip Direction Rose Diagram\n{base_name}',
                fontsize=14,  # 适当增大
                weight='bold',
                pad=25,  # 增加标题与图的间距
                loc='center'  # 居中对齐
            )

            # 8. 添加统计信息框 - 优化样式和位置
            total_count = len(dipdir_deg)
            mean_direction = np.mean(dipdir_deg)
            dominant_sector = centers_deg[np.argmax(hist)]
            max_freq = np.max(hist)

            # 更专业的统计信息文本
            info_text = (f'Total Discontinuities: {total_count:,}\n'
                         f'Mean Direction: {mean_direction:.1f}°\n'
                         f'Dominant Sector: {dominant_sector:.0f}°\n'
                         f'Max Frequency: {max_freq}')

            # 信息框样式优化
            bbox_props = dict(
                boxstyle='round,pad=0.5',
                facecolor='lightgray',
                alpha=0.9,
                edgecolor='black',
                linewidth=1.0
            )

            ax.text(
                0.02, 0.98, info_text,
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='left',
                fontsize=10,
                weight='bold',
                bbox=bbox_props,
                zorder=5
            )

            # 9. 添加图例（如果需要）
            # 创建一个简单的图例说明
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='#2E86AB', alpha=0.85,
                                     edgecolor='black', linewidth=0.8,
                                     label='Dip Direction Frequency')]

            ax.legend(handles=legend_elements,
                      loc='upper right',
                      bbox_to_anchor=(0.98, 0.98),
                      fontsize=10,
                      frameon=True,
                      fancybox=True,
                      shadow=True,
                      borderpad=0.8)

            # 10. 整体美化调整
            # 设置背景色为白色，确保清晰度
            ax.set_facecolor('white')
            fig.patch.set_facecolor('white')

            # 精细调整布局
            # fig.tight_layout(pad=2.0)

            # 保存图像
            self._SaveFigure(
                fig=fig,
                fig_dpi=self.dpi,
                out_dirs=[out_dir],
                base_name=base_name,
                plot_short_name=plot_short_name,
                output_formats=output_formats,
                logger=self.logger
            )

            if show:
                plt.show()
            else:
                plt.close(fig)

    def _PlotDipDirectionRoseCompare(
            self,
            base_name: str,
            paths_group: Sequence[PathTuple],
            plot_short_name: str,
            output_formats: Sequence[str],
            show: bool,
    ) -> None:
        """
        功能简介:
            对同一数据集(base_name)下的多个结果(不同 out_dir)
            绘制倾向(dip direction)玫瑰图对比图。在同一张极坐标图中,
            使用不同颜色的柱状玫瑰叠加, 图例标注算法/参数标签。

            输出文件:
                对每个输出格式 fmt, 在该数据集的所有 out_dir 中保存同一张图:
                    <out_dir_i>/<base_name>_<plot_short_name><fmt>
                例如:
                    Rock_GLS4_part1_localize_0.05m_dipdir_rose_cmp.png

        实现思路:
            1) 对 paths_group 中的每个 (out_dir, point_path):
                 - 构造 disc_csv_path;
                 - 从 CSV 中读取 Dipdir 列, 并统计直方图;
                 - 标签 label 由 _ExtractAlgorithmTagFromOutDir(out_dir) 给出;
            2) 统一采用相同的角度分箱, 在一张极坐标图中叠加多组柱状;
            3) 将生成的图像保存到该组中所有 out_dir 下。
        输入:
            base_name: str
                数据集 basename。
            paths_group: Sequence[PathTuple]
                属于该数据集的所有 (out_dir, point_path)。
            plot_short_name: str
                图件简称, 如 "dipdir_rose_cmp"。
            output_formats: Sequence[str]
                输出格式列表(".png", ".svg" 等)。
            show: bool
                是否调用 plt.show() 显示图像。
        输出:
            无显式返回。
        """
        if not paths_group:
            return

        bin_size_deg = 10.0
        edges_deg = np.arange(0.0, 360.0 + bin_size_deg, bin_size_deg)
        width_rad = np.deg2rad(bin_size_deg)
        centers_deg = edges_deg[:-1] + bin_size_deg * 0.5
        angles_rad = np.deg2rad(centers_deg)

        with Timer(
                f"_PlotDipDirectionRoseCompare(base={base_name}, n_results={len(paths_group)})",
                self.logger,
        ):
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, polar=True)

            any_plotted = False
            for out_dir, point_path in paths_group:
                disc_csv_path = self._BuildDiscCsvPath(out_dir, base_name)
                if not os.path.isfile(disc_csv_path):
                    self.logger.warning(
                        f"_PlotDipDirectionRoseCompare: 未找到 CSV, 跳过: {disc_csv_path}"
                    )
                    continue

                try:
                    df = pd.read_csv(disc_csv_path)
                except Exception as e:
                    self.logger.warning(
                        f"_PlotDipDirectionRoseCompare: 读取 CSV 失败, 跳过: {disc_csv_path}, 错误: {e}"
                    )
                    continue

                if "Dipdir" not in df.columns:
                    self.logger.warning(
                        f"_PlotDipDirectionRoseCompare: 'Dipdir' 列不存在, 跳过: {disc_csv_path}"
                    )
                    continue

                dipdir_deg = df["Dipdir"].to_numpy(dtype=float)
                dipdir_deg = np.mod(dipdir_deg, 360.0)

                if dipdir_deg.size == 0:
                    self.logger.warning(
                        f"_PlotDipDirectionRoseCompare: Dipdir 数量为 0, 跳过: {disc_csv_path}"
                    )
                    continue

                hist, _ = np.histogram(dipdir_deg, bins=edges_deg)
                label = self._ExtractAlgorithmTagFromOutDir(out_dir)

                ax.bar(
                    angles_rad,
                    hist,
                    width=width_rad,
                    bottom=0.0,
                    alpha=0.5,
                    edgecolor="k",
                    label=label,
                    align="center",
                )
                any_plotted = True

                self.logger.info(
                    f"_PlotDipDirectionRoseCompare: 已叠加 {label}, "
                    f"csv={os.path.basename(disc_csv_path)}, count={dipdir_deg.size}"
                )

            if not any_plotted:
                plt.close(fig)
                self.logger.warning(
                    f"_PlotDipDirectionRoseCompare: base_name={base_name} 无有效 Dipdir 数据, 跳过保存。"
                )
                return

            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.set_thetagrids(np.arange(0, 360, 30))

            ax.set_title(f"Dip Direction Rose Compare ({base_name})", fontsize=14)
            ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.2))

            fig.tight_layout()

            out_dirs = [out_dir for out_dir, _ in paths_group]
            self._SaveFigure(
                fig=fig,
                fig_dpi=self.dpi,
                out_dirs=out_dirs,
                base_name=base_name,
                plot_short_name=plot_short_name,
                output_formats=output_formats,
                logger=self.logger,
            )

            if show:
                plt.show()
            else:
                plt.close(fig)

    # 添加 stereonet + KDE + 聚类表的绘图函数
    def _PlotStereonetKdeSingle(
            self,
            out_dir: str,
            point_path: str,
            base_name: str,
            plot_short_name: str,
            output_formats: Sequence[str],
            show: bool,
    ) -> None:
        """
        功能简介:
            绘制结构面极点的上半球极射赤平投影图，背景为 KDE 密度分布，
            右侧为归一化色带，左侧为"结构面优势分组信息"表。
            修复中文显示问题和色带宽度问题。
        """
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        from sklearn.neighbors import KernelDensity

        disc_csv_path = self._BuildDiscCsvPath(out_dir, base_name)
        if not os.path.isfile(disc_csv_path):
            self.logger.warning(f"未找到结构面 CSV: {disc_csv_path}")
            return

        try:
            df = pd.read_csv(disc_csv_path)
        except Exception as e:
            self.logger.warning(f"读取 CSV 失败: {disc_csv_path}, 错误: {e}")
            return

        cluster_csv_path = self._BuildClusterCsvPath(out_dir, base_name)
        if not os.path.isfile(cluster_csv_path):
            self.logger.warning(f"未找到聚类 CSV: {cluster_csv_path}")
            return

        try:
            df_cluster = pd.read_csv(cluster_csv_path)
        except Exception as e:
            self.logger.warning(f"读取 CSV 失败: {cluster_csv_path}, 错误: {e}")
            return

        # 检查必要列
        required_cols = ["Dip", "Dipdir", "Cluster_id"]
        for col in required_cols:
            if col not in df.columns:
                self.logger.warning(f"CSV 缺少列 '{col}'，跳过绘图")
                return

        # 提取数据
        dip_deg = df["Dip"].to_numpy(dtype=float)
        dipdir_deg = df["Dipdir"].to_numpy(dtype=float)
        cluster_ids = df["Cluster_id"].to_numpy(dtype=int)
        clusters_id = df_cluster["Cluster_id"]
        clusters_dip = df_cluster["Dip"]
        clusters_dipdir = df_cluster["Dipdir"]
        clusters_diconnumber = df_cluster["Discontinuity_Number"]
        clusters_confidence = df_cluster["confidence"]

        # 过滤无效数据
        valid_mask = (
                (dip_deg >= 0) & (dip_deg <= 90) &
                (dipdir_deg >= 0) & (dipdir_deg < 360) &
                np.isfinite(dip_deg) & np.isfinite(dipdir_deg)
        )

        if not np.any(valid_mask):
            self.logger.warning("没有有效数据，跳过绘图")
            return

        dip_deg = dip_deg[valid_mask]
        dipdir_deg = dipdir_deg[valid_mask]
        cluster_ids = cluster_ids[valid_mask]

        with Timer(f"_PlotStereonetKdeSingle(base={base_name})", self.logger):
            # 1) 将倾角/倾向转换为 stereonet 极坐标
            def stereographic_r_from_dip(dip_deg):
                """将倾角转换为极射赤平投影半径"""
                return np.tan(np.radians(dip_deg) / 2.0)

            r = stereographic_r_from_dip(dip_deg)
            theta = np.radians(dipdir_deg)

            # 2) 计算 KDE 密度
            try:
                # 创建规则网格
                n_dir, n_dip = 181, 91
                dir_grid = np.linspace(0, 360, n_dir)
                dip_grid = np.linspace(0, 90, n_dip)
                dir_mesh, dip_mesh = np.meshgrid(dir_grid, dip_grid)

                # 转换为极坐标网格
                theta_grid = np.radians(dir_mesh)
                r_grid = stereographic_r_from_dip(dip_mesh)

                # 准备 KDE 数据
                X = np.column_stack([dipdir_deg / 360.0, dip_deg / 90.0])

                # 使用 sklearn KernelDensity
                kde = KernelDensity(bandwidth=0.05, kernel='gaussian')
                kde.fit(X)

                # 评估网格上的密度
                grid_points = np.column_stack([dir_mesh.ravel() / 360.0, dip_mesh.ravel() / 90.0])
                log_density = kde.score_samples(grid_points)
                density = np.exp(log_density).reshape(dir_mesh.shape)

                # 归一化到 [0, 1]
                density_norm = (density - density.min()) / (density.max() - density.min() + 1e-10)

            except Exception as e:
                self.logger.warning(f"KDE 计算失败: {e}")
                density_norm = None

            # 3) 创建图形和子图

            fig = plt.figure(figsize=(16, 8), dpi=self.dpi)

            # 使用 GridSpec 创建 1x3 布局，调整宽度比例
            gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 3.0, 0.2],  # 减少色带宽度
                                  wspace=0.25, hspace=0.1)

            # 左侧: 聚类信息表
            ax_table = fig.add_subplot(gs[0, 0])
            ax_table.axis('off')

            # 计算聚类统计信息
            unique_clusters = np.unique(cluster_ids[cluster_ids >= 0])
            if len(unique_clusters) == 0:
                # 如果没有聚类信息，显示提示
                ax_table.text(0.5, 0.5, "No valid clusters found",
                              ha='center', va='center', fontsize=12)
            else:
                # 准备表格数据
                table_data = []
                colors_legend = []

                # 为每个聚类分配颜色
                cmap_clusters = cm.get_cmap('tab20', len(unique_clusters))
                for i, cluster_id in enumerate(unique_clusters):
                    mask = cluster_ids == cluster_id
                    if np.sum(mask) == 0:
                        continue

                    # 计算平均倾角/倾向
                    cluster_dip = np.mean(dip_deg[mask])
                    cluster_dipdir = np.mean(dipdir_deg[mask])
                    cluster_count = np.sum(mask)

                    # 确保倾向在 [0, 360) 范围内
                    cluster_dipdir = cluster_dipdir % 360.0

                    table_data.append([
                        "●",  # 图例符号
                        f"{cluster_dip:.1f}°/{cluster_dipdir:.1f}°",
                        str(cluster_count)
                    ])
                    colors_legend.append(cmap_clusters(i))

                # 创建表格 - 使用支持中文的字体
                table = ax_table.table(
                    cellText=table_data,
                    colLabels=["cluster", "dip/dip dir", "discontinuities"],
                    colColours=['lightgray', 'lightgray', 'lightgray'],
                    cellLoc='center',
                    loc='center',
                    fontsize=11  # 稍微增大字体
                )

                # 设置表格样式
                table.auto_set_font_size(False)
                table.set_fontsize(11)

                # 设置表头样式 - 加粗
                for j in range(3):
                    table[0, j].set_text_props(weight='bold', fontsize=12)

                # 设置图例列的颜色 - 增大图例符号
                for row_idx in range(len(table_data)):
                    table[row_idx + 1, 0].get_text().set_color(colors_legend[row_idx])
                    table[row_idx + 1, 0].get_text().set_fontsize(16)  # 增大图例符号

                # 调整表格缩放
                table.scale(1.2, 1.8)

            # 设置表标题 - 确保中文字体
            ax_table.set_title("clusters", fontsize=14, pad=25, weight='bold')

            # 中部: stereonet
            ax_stereo = fig.add_subplot(gs[0, 1], projection='polar')

            # 设置 stereonet 参数
            ax_stereo.set_theta_zero_location('N')
            ax_stereo.set_theta_direction(-1)
            ax_stereo.set_rlim(0, 1)

            # 绘制 KDE 背景
            if density_norm is not None:
                # 使用等高线填充
                levels = np.linspace(0, 1, 21)
                contour = ax_stereo.contourf(
                    theta_grid, r_grid, density_norm,
                    levels=levels, cmap='viridis', alpha=0.7
                )

            # 绘制散点 (按聚类着色)
            if len(unique_clusters) > 0:
                cmap_scatter = cm.get_cmap('tab20', len(unique_clusters))
                for i, cluster_id in enumerate(unique_clusters):
                    mask = cluster_ids == cluster_id
                    if np.sum(mask) == 0:
                        continue

                    ax_stereo.scatter(
                        theta[mask], r[mask],
                        s=25, alpha=0.8,
                        color=cmap_scatter(i),
                        edgecolors='k', linewidth=0.5,
                        label=f'Cluster {cluster_id}',
                        zorder=3
                    )
            else:
                # 如果没有聚类，使用单一颜色
                ax_stereo.scatter(
                    theta, r,
                    s=25, alpha=0.8,
                    color='gray',
                    edgecolors='k', linewidth=0.5,
                    zorder=3
                )

            # 设置 stereonet 网格和标签
            ax_stereo.set_thetagrids(
                np.arange(0, 360, 30),
                labels=['N', '30°', '60°', 'E', '120°', '150°',
                        'S', '210°', '240°', 'W', '300°', '330°'],
                fontsize=10
            )

            # 设置半径网格和标签
            r_ticks = [0.30, 0.5, 0.75, 1.0]
            r_labels = ['15°', '30°', '45°', '90°']
            ax_stereo.set_rgrids(
                r_ticks, labels=r_labels,
                angle=135, fontsize=10
            )

            # 设置标题
            ax_stereo.set_title(
                f"Stereographic Projection with KDE Density\n({base_name})",
                fontsize=13, pad=15, weight='bold'
            )

            # 右侧: colorbar - 调整宽度和刻度
            if density_norm is not None:
                ax_cbar = fig.add_subplot(gs[0, 2])

                # 创建colorbar，调整宽度
                cbar = plt.colorbar(contour, cax=ax_cbar, orientation='vertical')
                cbar.set_label('Normalized Density', fontsize=11, weight='bold')

                # 设置colorbar刻度 - 包括顶部和底部的值
                cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
                cbar.set_ticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
                cbar.ax.tick_params(labelsize=9)

                # 确保顶部有1.00刻度
                if 1.0 not in cbar.get_ticks():
                    cbar.set_ticks(list(cbar.get_ticks()) + [1.0])
                    cbar.set_ticklabels([f'{t:.1f}' for t in cbar.get_ticks()])

            # 调整布局
            fig.tight_layout()

            # 保存图像
            self._SaveFigure(
                fig=fig,
                fig_dpi=self.dpi,
                out_dirs=[out_dir],
                base_name=base_name,
                plot_short_name=plot_short_name,
                output_formats=output_formats,
                logger=self.logger
            )

            if show:
                plt.show()
            else:
                plt.close(fig)

    # 添加面积分布图函数
    def _PlotAreaDistributionSingle(
            self,
            out_dir: str,
            point_path: str,
            base_name: str,
            plot_short_name: str,
            output_formats: Sequence[str],
            show: bool,
    ) -> None:
        """
        功能简介:
            绘制结构面面积的概率分布图与PDF曲线.

        实现思路:
            1) 读取结构面级 CSV 获取 Area 数据;
            2) 绘制归一化直方图;
            3) 使用 KDE 估计 PDF 曲线;
            4) 自适应横纵坐标范围, 刻度顶满边框.
        """

        from scipy.stats import gaussian_kde

        disc_csv_path = self._BuildDiscCsvPath(out_dir, base_name)
        if not os.path.isfile(disc_csv_path):
            self.logger.warning(f"未找到结构面 CSV: {disc_csv_path}")
            return

        try:
            df = pd.read_csv(disc_csv_path)
        except Exception as e:
            self.logger.warning(f"读取 CSV 失败: {disc_csv_path}, 错误: {e}")
            return

        if "Area" not in df.columns:
            self.logger.warning(f"CSV 缺少列 'Area', 跳过绘图")
            return

        # 提取面积数据
        areas = df["Area"].to_numpy(dtype=float)
        areas = areas[areas > 0]  # 只保留正面积
        areas = areas[np.isfinite(areas)]  # 移除无穷大/NaN

        if len(areas) < 5:
            self.logger.warning(f"有效面积数据不足 ({len(areas)}个), 跳过绘图")
            return

        with Timer(f"_PlotAreaDistributionSingle(base={base_name})", self.logger):
            fig, ax = plt.subplots(figsize=(8, 5), dpi=self.dpi)

            # 自适应直方图分箱
            n_bins = min(30, int(len(areas) / 10))
            n_bins = max(n_bins, 10)

            # 绘制直方图
            hist, bins, patches = ax.hist(
                areas, bins=n_bins, density=True,
                alpha=0.6, color='steelblue',
                edgecolor='black', linewidth=0.5,
                label='Histogram'
            )

            # 计算 KDE PDF
            try:
                kde = gaussian_kde(areas)
                x_eval = np.linspace(bins[0], bins[-1], 500)
                pdf = kde(x_eval)

                # 绘制 PDF 曲线
                ax.plot(x_eval, pdf, 'r-', linewidth=2, label='PDF')
            except Exception as e:
                self.logger.warning(f"KDE 计算失败: {e}")

            # 设置坐标轴
            ax.set_xlabel('Area (m²)', fontsize=12, weight='bold')
            ax.set_ylabel('Probability Density', fontsize=12, weight='bold')

            # 自适应坐标范围
            x_min = np.min(areas)
            x_max = np.max(areas)
            x_range = x_max - x_min
            ax.set_xlim(max(0, x_min - 0.05 * x_range), x_max + 0.05 * x_range)

            y_max = max(np.max(hist), np.max(pdf) if 'pdf' in locals() else 0)
            ax.set_ylim(0, y_max * 1.1)

            # 设置刻度 - 顶满边框
            n_x_ticks = 5
            n_y_ticks = 5

            x_ticks = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], n_x_ticks)
            y_ticks = np.linspace(0, ax.get_ylim()[1], n_y_ticks)

            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)

            # 格式化刻度标签
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{y:.3f}'))

            # 设置网格
            ax.grid(True, alpha=0.3, linestyle='--')

            # 设置标题和图例
            ax.set_title(f'Area Distribution\n({base_name})', fontsize=14, weight='bold', pad=15)
            ax.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)

            # 调整布局
            fig.tight_layout()

            # 保存图像
            self._SaveFigure(
                fig=fig,
                fig_dpi=self.dpi,
                out_dirs=[out_dir],
                base_name=base_name,
                plot_short_name=plot_short_name,
                output_formats=output_formats,
                logger=self.logger
            )

            if show:
                plt.show()
            else:
                plt.close(fig)

    # 添加迹长分布图函数
    def _PlotTraceLengthDistributionSingle(
            self,
            out_dir: str,
            point_path: str,
            base_name: str,
            plot_short_name: str,
            output_formats: Sequence[str],
            show: bool,
    ) -> None:
        """
        功能简介:
            绘制结构面迹长的概率分布图与PDF曲线.
        """

        from scipy.stats import gaussian_kde

        disc_csv_path = self._BuildDiscCsvPath(out_dir, base_name)
        if not os.path.isfile(disc_csv_path):
            self.logger.warning(f"未找到结构面 CSV: {disc_csv_path}")
            return

        try:
            df = pd.read_csv(disc_csv_path)
        except Exception as e:
            self.logger.warning(f"读取 CSV 失败: {disc_csv_path}, 错误: {e}")
            return

        if "TraceLength" not in df.columns:
            self.logger.warning(f"CSV 缺少列 'TraceLength', 跳过绘图")
            return

        # 提取迹长数据
        trace_lengths = df["TraceLength"].to_numpy(dtype=float)
        trace_lengths = trace_lengths[trace_lengths > 0]  # 只保留正迹长
        trace_lengths = trace_lengths[np.isfinite(trace_lengths)]  # 移除无穷大/NaN

        if len(trace_lengths) < 5:
            self.logger.warning(f"有效迹长数据不足 ({len(trace_lengths)}个), 跳过绘图")
            return

        with Timer(f"_PlotTraceLengthDistributionSingle(base={base_name})", self.logger):
            fig, ax = plt.subplots(figsize=(8, 5), dpi=self.dpi)

            # 自适应直方图分箱
            n_bins = min(30, int(len(trace_lengths) / 10))
            n_bins = max(n_bins, 10)

            # 绘制直方图
            hist, bins, patches = ax.hist(
                trace_lengths, bins=n_bins, density=True,
                alpha=0.6, color='forestgreen',
                edgecolor='black', linewidth=0.5,
                label='Histogram'
            )

            # 计算 KDE PDF
            try:
                kde = gaussian_kde(trace_lengths)
                x_eval = np.linspace(bins[0], bins[-1], 500)
                pdf = kde(x_eval)

                # 绘制 PDF 曲线
                ax.plot(x_eval, pdf, 'r-', linewidth=2, label='PDF')
            except Exception as e:
                self.logger.warning(f"KDE 计算失败: {e}")

            # 设置坐标轴
            ax.set_xlabel('Trace Length (m)', fontsize=12, weight='bold')
            ax.set_ylabel('Probability Density', fontsize=12, weight='bold')

            # 自适应坐标范围
            x_min = np.min(trace_lengths)
            x_max = np.max(trace_lengths)
            x_range = x_max - x_min
            ax.set_xlim(max(0, x_min - 0.05 * x_range), x_max + 0.05 * x_range)

            y_max = max(np.max(hist), np.max(pdf) if 'pdf' in locals() else 0)
            ax.set_ylim(0, y_max * 1.1)

            # 设置刻度 - 顶满边框
            n_x_ticks = 5
            n_y_ticks = 5

            x_ticks = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], n_x_ticks)
            y_ticks = np.linspace(0, ax.get_ylim()[1], n_y_ticks)

            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)

            # 格式化刻度标签
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{y:.3f}'))

            # 设置网格
            ax.grid(True, alpha=0.3, linestyle='--')

            # 设置标题和图例
            ax.set_title(f'Trace Length Distribution\n({base_name})', fontsize=14, weight='bold', pad=15)
            ax.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)

            # 调整布局
            fig.tight_layout()

            # 保存图像
            self._SaveFigure(
                fig=fig,
                fig_dpi=self.dpi,
                out_dirs=[out_dir],
                base_name=base_name,
                plot_short_name=plot_short_name,
                output_formats=output_formats,
                logger=self.logger
            )

            if show:
                plt.show()
            else:
                plt.close(fig)

    # 添加粗糙度分布图函数
    def _PlotRoughnessDistributionSingle(
            self,
            out_dir: str,
            point_path: str,
            base_name: str,
            plot_short_name: str,
            output_formats: Sequence[str],
            show: bool,
            roughness_xlim: Tuple[float, float] = (0, 1),
    ) -> None:
        """
        功能简介:
            绘制结构面粗糙度的概率分布图与PDF曲线, 横坐标采用统一值域.

        输入:
            roughness_xlim: 横坐标范围, 默认 (0, 1)
        """

        from scipy.stats import gaussian_kde

        disc_csv_path = self._BuildDiscCsvPath(out_dir, base_name)
        if not os.path.isfile(disc_csv_path):
            self.logger.warning(f"未找到结构面 CSV: {disc_csv_path}")
            return

        try:
            df = pd.read_csv(disc_csv_path)
        except Exception as e:
            self.logger.warning(f"读取 CSV 失败: {disc_csv_path}, 错误: {e}")
            return

        if "Roughness" not in df.columns:
            self.logger.warning(f"CSV 缺少列 'Roughness', 跳过绘图")
            return

        # 提取粗糙度数据
        roughness = df["Roughness"].to_numpy(dtype=float)
        roughness = roughness[np.isfinite(roughness)]  # 移除无穷大/NaN

        if len(roughness) < 5:
            self.logger.warning(f"有效粗糙度数据不足 ({len(roughness)}个), 跳过绘图")
            return

        with Timer(f"_PlotRoughnessDistributionSingle(base={base_name})", self.logger):
            fig, ax = plt.subplots(figsize=(8, 5), dpi=self.dpi)

            # 自适应直方图分箱
            n_bins = min(30, int(len(roughness) / 10))
            n_bins = max(n_bins, 10)

            # 使用统一横坐标范围
            x_min, x_max = roughness_xlim

            # 绘制直方图
            hist, bins, patches = ax.hist(
                roughness, bins=n_bins, density=True,
                range=(x_min, x_max),
                alpha=0.6, color='darkorange',
                edgecolor='black', linewidth=0.5,
                label='Histogram'
            )

            # 计算 KDE PDF
            try:
                kde = gaussian_kde(roughness)
                x_eval = np.linspace(x_min, x_max, 500)
                pdf = kde(x_eval)

                # 绘制 PDF 曲线
                ax.plot(x_eval, pdf, 'r-', linewidth=2, label='PDF')
            except Exception as e:
                self.logger.warning(f"KDE 计算失败: {e}")

            # 设置坐标轴
            ax.set_xlabel('Roughness', fontsize=12, weight='bold')
            ax.set_ylabel('Probability Density', fontsize=12, weight='bold')

            # 自适应纵坐标范围
            y_max = max(np.max(hist), np.max(pdf) if 'pdf' in locals() else 0)
            ax.set_ylim(0, y_max * 1.1)

            # 设置刻度 - 顶满边框
            n_x_ticks = 5
            n_y_ticks = 5

            x_ticks = np.linspace(x_min, x_max, n_x_ticks)
            y_ticks = np.linspace(0, ax.get_ylim()[1], n_y_ticks)

            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)

            # 格式化刻度标签
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{y:.3f}'))

            # 设置网格
            ax.grid(True, alpha=0.3, linestyle='--')

            # 设置标题和图例
            ax.set_title(f'Roughness Distribution\n({base_name})', fontsize=14, weight='bold', pad=15)
            ax.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)

            # 调整布局
            fig.tight_layout()

            # 保存图像
            self._SaveFigure(
                fig=fig,
                fig_dpi=self.dpi,
                out_dirs=[out_dir],
                base_name=base_name,
                plot_short_name=plot_short_name,
                output_formats=output_formats,
                logger=self.logger
            )

            if show:
                plt.show()
            else:
                plt.close(fig)

    # 统一的分布图
    def _PlotDistributionsSingle(
            self,
            out_dir: str,
            point_path: str,
            base_name: str,
            plot_short_name: str,
            output_formats: Sequence[str],
            show: bool,
    ) -> None:
        """
        功能简介:
            将三个分布图(面积、迹长、粗糙度)作为子图横排罗列放置，
            便于直接插入A4大小的word文档中。

        实现思路:
            1) 创建3个子图，横排布局;
            2) 每个子图有左右双纵轴:
               - 左纵轴: 柱状图(结构面数量)
               - 右纵轴: PDF曲线(密度)
            3) 右纵轴颜色与PDF曲线颜色一致;
            4) 总图宽高比接近2:1，适合A4文档。
        """

        from scipy.stats import gaussian_kde

        disc_csv_path = self._BuildDiscCsvPath(out_dir, base_name)
        if not os.path.isfile(disc_csv_path):
            self.logger.warning(f"未找到结构面 CSV: {disc_csv_path}")
            return

        try:
            df = pd.read_csv(disc_csv_path)
        except Exception as e:
            self.logger.warning(f"读取 CSV 失败: {disc_csv_path}, 错误: {e}")
            return

        # 检查必要列
        required_cols = ["Area", "TraceLength", "Roughness"]
        for col in required_cols:
            if col not in df.columns:
                self.logger.warning(f"CSV 缺少列 '{col}'，跳过绘图")
                return

        with Timer(f"_PlotDistributionsSingle(base={base_name})", self.logger):
            # 准备数据
            areas = df["Area"].to_numpy(dtype=float)
            trace_lengths = df["TraceLength"].to_numpy(dtype=float)
            roughness = df["Roughness"].to_numpy(dtype=float)

            # 过滤有效数据
            areas_valid = areas[(areas > 0) & np.isfinite(areas)]
            trace_valid = trace_lengths[(trace_lengths > 0) & np.isfinite(trace_lengths)]
            rough_valid = roughness[np.isfinite(roughness)]

            # 创建图形，宽高比接近2:1 (A4文档比例)
            fig = plt.figure(figsize=(17, 6), dpi=self.dpi)

            # 创建3个子图
            gs = gridspec.GridSpec(1, 3, wspace=0.5, hspace=0.3)

            # 定义颜色
            hist_color = '#1f77b4'  # 柱状图颜色
            pdf_color = '#d62728'  # PDF曲线颜色

            # 子图1: 面积分布
            if len(areas_valid) >= 5:
                ax1 = fig.add_subplot(gs[0, 0])
                ax1_right = ax1.twinx()

                # 自适应直方图分箱
                n_bins = min(20, max(10, int(len(areas_valid) / 15)))

                # 绘制柱状图 (左纵轴)
                hist, bins, _ = ax1.hist(
                    areas_valid, bins=n_bins,
                    color=hist_color, alpha=0.7,
                    edgecolor='black', linewidth=0.5,
                    label='Count'
                )

                # 计算KDE PDF (右纵轴)
                try:
                    kde = gaussian_kde(areas_valid)
                    x_eval = np.linspace(bins[0], bins[-1], 300)
                    pdf = kde(x_eval)

                    # 绘制PDF曲线 (右纵轴)
                    ax1_right.plot(x_eval, pdf, '-', color=pdf_color,
                                   linewidth=2, label='PDF Density')
                except Exception as e:
                    self.logger.warning(f"面积KDE计算失败: {e}")
                    pdf = np.zeros_like(x_eval)

                # 设置左纵轴
                ax1.set_xlabel('Area (m²)', fontsize=11, weight='bold')
                ax1.set_ylabel('Number of Discontinuities', fontsize=11,
                               weight='bold', color=hist_color)
                ax1.tick_params(axis='y', labelcolor=hist_color)

                # 设置右纵轴
                ax1_right.set_ylabel('Probability Density', fontsize=11,
                                     weight='bold', color=pdf_color)
                ax1_right.tick_params(axis='y', labelcolor=pdf_color)

                # 设置标题
                ax1.set_title('(a) Area Distribution', fontsize=12, weight='bold', pad=10)

                # 添加图例
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax1_right.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2,
                           loc='upper right', fontsize=9)

                # 自适应横坐标范围
                x_min = np.min(areas_valid)
                x_max = np.max(areas_valid)
                x_range = x_max - x_min
                ax1.set_xlim(max(0, x_min - 0.05 * x_range), x_max + 0.05 * x_range)

                # 网格
                ax1.grid(True, alpha=0.3, linestyle='--')

            else:
                ax1 = fig.add_subplot(gs[0, 0])
                ax1.text(0.5, 0.5, 'Insufficient Area Data',
                         ha='center', va='center', fontsize=12)
                ax1.set_title('(a) Area Distribution', fontsize=12, weight='bold', pad=10)

            # 子图2: 迹长分布
            if len(trace_valid) >= 5:
                ax2 = fig.add_subplot(gs[0, 1])
                ax2_right = ax2.twinx()

                # 自适应直方图分箱
                n_bins = min(20, max(10, int(len(trace_valid) / 15)))

                # 绘制柱状图 (左纵轴)
                hist, bins, _ = ax2.hist(
                    trace_valid, bins=n_bins,
                    color=hist_color, alpha=0.7,
                    edgecolor='black', linewidth=0.5,
                    label='Count'
                )

                # 计算KDE PDF (右纵轴)
                try:
                    kde = gaussian_kde(trace_valid)
                    x_eval = np.linspace(bins[0], bins[-1], 300)
                    pdf = kde(x_eval)

                    # 绘制PDF曲线 (右纵轴)
                    ax2_right.plot(x_eval, pdf, '-', color=pdf_color,
                                   linewidth=2, label='PDF Density')
                except Exception as e:
                    self.logger.warning(f"迹长KDE计算失败: {e}")
                    pdf = np.zeros_like(x_eval)

                # 设置左纵轴
                ax2.set_xlabel('Trace Length (m)', fontsize=11, weight='bold')
                ax2.set_ylabel('Number of Discontinuities', fontsize=11,
                               weight='bold', color=hist_color)
                ax2.tick_params(axis='y', labelcolor=hist_color)

                # 设置右纵轴
                ax2_right.set_ylabel('Probability Density', fontsize=11,
                                     weight='bold', color=pdf_color)
                ax2_right.tick_params(axis='y', labelcolor=pdf_color)

                # 设置标题
                ax2.set_title('(b) Trace Length Distribution', fontsize=12, weight='bold', pad=10)

                # 添加图例
                lines1, labels1 = ax2.get_legend_handles_labels()
                lines2, labels2 = ax2_right.get_legend_handles_labels()
                ax2.legend(lines1 + lines2, labels1 + labels2,
                           loc='upper right', fontsize=9)

                # 自适应横坐标范围
                x_min = np.min(trace_valid)
                x_max = np.max(trace_valid)
                x_range = x_max - x_min
                ax2.set_xlim(max(0, x_min - 0.05 * x_range), x_max + 0.05 * x_range)

                # 网格
                ax2.grid(True, alpha=0.3, linestyle='--')

            else:
                ax2 = fig.add_subplot(gs[0, 1])
                ax2.text(0.5, 0.5, 'Insufficient Trace Length Data',
                         ha='center', va='center', fontsize=12)
                ax2.set_title('(b) Trace Length Distribution', fontsize=12, weight='bold', pad=10)

            # 子图3: 粗糙度分布
            if len(rough_valid) >= 5:
                ax3 = fig.add_subplot(gs[0, 2])
                ax3_right = ax3.twinx()

                # 使用统一横坐标范围 (0-1)
                x_min, x_max = 0, 1.0
                n_bins = 20

                # 绘制柱状图 (左纵轴)
                hist, bins, _ = ax3.hist(
                    rough_valid, bins=n_bins,
                    range=(x_min, x_max),
                    color=hist_color, alpha=0.7,
                    edgecolor='black', linewidth=0.5,
                    label='Count'
                )

                # 计算KDE PDF (右纵轴)
                try:
                    kde = gaussian_kde(rough_valid)
                    x_eval = np.linspace(x_min, x_max, 300)
                    pdf = kde(x_eval)

                    # 绘制PDF曲线 (右纵轴)
                    ax3_right.plot(x_eval, pdf, '-', color=pdf_color,
                                   linewidth=2, label='PDF Density')
                except Exception as e:
                    self.logger.warning(f"粗糙度KDE计算失败: {e}")
                    pdf = np.zeros_like(x_eval)

                # 设置左纵轴
                ax3.set_xlabel('Roughness', fontsize=11, weight='bold')
                ax3.set_ylabel('Number of Discontinuities', fontsize=11,
                               weight='bold', color=hist_color)
                ax3.tick_params(axis='y', labelcolor=hist_color)

                # 设置右纵轴
                ax3_right.set_ylabel('Probability Density', fontsize=11,
                                     weight='bold', color=pdf_color)
                ax3_right.tick_params(axis='y', labelcolor=pdf_color)

                # 设置标题
                ax3.set_title('(c) Roughness Distribution', fontsize=12, weight='bold', pad=10)

                # 添加图例
                lines1, labels1 = ax3.get_legend_handles_labels()
                lines2, labels2 = ax3_right.get_legend_handles_labels()
                ax3.legend(lines1 + lines2, labels1 + labels2,
                           loc='upper right', fontsize=9)

                # 设置横坐标范围
                ax3.set_xlim(x_min, x_max)

                # 网格
                ax3.grid(True, alpha=0.3, linestyle='--')

            else:
                ax3 = fig.add_subplot(gs[0, 2])
                ax3.text(0.5, 0.5, 'Insufficient Roughness Data',
                         ha='center', va='center', fontsize=12)
                ax3.set_title('(c) Roughness Distribution', fontsize=12, weight='bold', pad=10)

            # 设置总标题
            fig.suptitle(f'Statistical Distributions of Discontinuities\n({base_name})',
                         fontsize=14, weight='bold', y=0.98)

            # 调整布局
            fig.tight_layout(rect=[0, 0, 1, 0.95])  # 为总标题留出空间

            # 保存图像
            self._SaveFigure(
                fig=fig,
                fig_dpi=self.dpi,
                out_dirs=[out_dir],
                base_name=base_name,
                plot_short_name=plot_short_name,
                output_formats=output_formats,
                logger=self.logger
            )

            if show:
                plt.show()
            else:
                plt.close(fig)

    # 添加优势结构面+整体坡面 stereonet 函数
    def _PlotSlopeClusterStereonetSingle(
            self,
            out_dir: str,
            point_path: str,
            base_name: str,
            plot_short_name: str,
            output_formats: Sequence[str],
            show: bool,
    ) -> None:
        """
        功能简介:
            绘制优势结构面+整体坡面的上半球极射赤平投影关系图。
            聚类产状用弧线表示，用于分析优势产状与坡面的几何关系。
        """

        # 设置全局字体为Times New Roman
        mpl.rcParams["font.family"] = self.font_family

        # 读取聚类CSV
        clusters_csv_path = os.path.join(out_dir, f"{base_name}_clusters.csv")
        if not os.path.isfile(clusters_csv_path):
            self.logger.warning(f"未找到聚类 CSV: {clusters_csv_path}")
            return

        try:
            clusters_df = pd.read_csv(clusters_csv_path)
        except Exception as e:
            self.logger.warning(f"读取聚类 CSV 失败: {clusters_csv_path}, 错误: {e}")
            return

        with Timer(f"_PlotSlopeClusterStereonetSingle(base={base_name})", self.logger):
            fig, ax = plt.subplots(figsize=(10, 10), dpi=self.dpi, subplot_kw=dict(projection='polar'))

            # 设置 stereonet 参数
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1)
            ax.set_rlim(0, 1)

            # 辅助函数: 倾角转半径
            def stereographic_r_from_dip(dip_deg):
                return np.tan(np.radians(dip_deg) / 2.0)

            # 辅助函数: 绘制平面弧线
            def plot_plane_great_circle(ax, dip_deg, dipdir_deg, color='blue', alpha=0.7, label=None):
                """
                在 stereonet 上绘制平面的弧线。
                参数:
                    dip_deg: 倾角
                    dipdir_deg: 倾向
                    color: 弧线颜色
                    alpha: 透明度
                    label: 图例标签
                """
                # 生成弧线点
                n_points = 180
                phi = np.radians(dipdir_deg)
                delta = np.radians(dip_deg)

                # 计算弧线在 stereonet 上的点
                # 对于 stereonet 投影，平面的轨迹是一个小圆
                t = np.linspace(0, 2 * np.pi, n_points)

                # 计算小圆的半径和中心
                # 倾角转换为 stereonet 上的位置
                r0 = stereographic_r_from_dip(dip_deg)
                theta0 = np.radians(dipdir_deg)

                # 计算小圆上各点的极角
                # 对于 stereonet，平面的轨迹是一个以(theta0, r0)为中心的圆
                # 简化：绘制一个通过极点的小圆
                if dip_deg > 0:
                    # 倾角大于0，绘制一个小圆
                    k = 2 * np.tan(np.radians(dip_deg) / 2)
                    r_circle = k / (1 + np.cos(t))
                    theta_circle = t + np.radians(dipdir_deg) - np.pi / 2

                    # 转换为直角坐标并过滤
                    x_circle = r_circle * np.cos(theta_circle)
                    y_circle = r_circle * np.sin(theta_circle)

                    # 只保留在单位圆内的点
                    mask = (x_circle ** 2 + y_circle ** 2) <= 1
                    if np.any(mask):
                        # 绘制弧线
                        ax.plot(theta_circle[mask], r_circle[mask],
                                color=color, linewidth=1.5, alpha=alpha, label=label)
                else:
                    # 水平面，绘制一个通过中心的圆
                    circle = plt.Circle((theta0, 0), r0, transform=ax.transData._b,
                                        fill=False, color=color, linewidth=1.5, alpha=alpha)
                    ax.add_artist(circle)

            # 1) 绘制聚类产状弧线
            if len(clusters_df) > 0:
                # 为每个聚类分配颜色
                colors = plt.cm.tab20(np.linspace(0, 1, len(clusters_df)))

                for idx, row in clusters_df.iterrows():
                    cluster_id = int(row["Cluster_id"])
                    dip = float(row["Dip"])
                    dipdir = float(row["Dipdir"])

                    # 绘制平面弧线
                    plot_plane_great_circle(
                        ax, dip, dipdir,
                        color=colors[idx], alpha=0.6,
                        label=f'Cluster {cluster_id}'
                    )

                    # 标记聚类中心点（可选，用小点表示）
                    r_center = stereographic_r_from_dip(dip)
                    theta_center = np.radians(dipdir)
                    ax.scatter(
                        theta_center, r_center,
                        s=30, alpha=1.0,
                        color=colors[idx],
                        edgecolors='k', linewidth=0.8,
                        marker='o',
                        zorder=5
                    )

                    # 添加聚类产状标注
                    ax.annotate(
                        f'{dip:.1f}°/{dipdir:.1f}°',
                        xy=(theta_center, r_center),
                        xytext=(theta_center + 0.15, r_center + 0.05),
                        textcoords='data',
                        fontsize=9,
                        weight='bold',
                        color=colors[idx],
                        arrowprops=dict(
                            arrowstyle='->',
                            color=colors[idx],
                            lw=1.0,
                            alpha=0.7
                        ),
                        zorder=6
                    )

            # 2) 绘制整体坡面弧线
            try:
                # 估计整体坡面
                import open3d as o3d
                from scipy.spatial import ConvexHull

                # 读取点云
                pcd = o3d.io.read_point_cloud(point_path)
                points = np.asarray(pcd.points)

                if len(points) >= 3:
                    # 使用 PCA 估计主要坡面
                    centroid = np.mean(points, axis=0)
                    points_centered = points - centroid

                    # 计算协方差矩阵
                    cov_matrix = np.cov(points_centered.T)

                    # 计算特征值和特征向量
                    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

                    # 最小特征值对应的特征向量为坡面法向量
                    slope_normal = eigenvectors[:, np.argmin(eigenvalues)]

                    # 统一到上半球 (nz > 0)
                    if slope_normal[2] < 0:
                        slope_normal = -slope_normal

                    # 计算倾角/倾向
                    vertical = np.array([0, 0, 1])
                    dip_rad = np.arccos(np.clip(np.dot(slope_normal, vertical), -1.0, 1.0))
                    dip_deg = np.degrees(dip_rad)

                    # 倾向
                    horizontal_proj = np.array([slope_normal[0], slope_normal[1], 0])
                    if np.linalg.norm(horizontal_proj) > 1e-6:
                        horizontal_proj = horizontal_proj / np.linalg.norm(horizontal_proj)
                        dipdir_deg = np.degrees(np.arctan2(horizontal_proj[0], horizontal_proj[1]))
                        dipdir_deg = (dipdir_deg + 360) % 360
                    else:
                        dipdir_deg = 0

                    # 绘制坡面弧线（用粗线表示）
                    plot_plane_great_circle(
                        ax, dip_deg, dipdir_deg,
                        color='red', alpha=0.9,
                        label='Overall Slope'
                    )

                    # 标记坡面极点
                    r_slope = stereographic_r_from_dip(dip_deg)
                    theta_slope = np.radians(dipdir_deg)
                    ax.scatter(
                        theta_slope, r_slope,
                        s=80, alpha=1.0,
                        color='red',
                        edgecolors='darkred', linewidth=1.5,
                        marker='*',
                        zorder=7
                    )

                    # 添加坡面产状标注
                    ax.annotate(
                        f'Slope: {dip_deg:.1f}°/{dipdir_deg:.1f}°',
                        xy=(theta_slope, r_slope),
                        xytext=(theta_slope + 0.3, r_slope + 0.1),
                        textcoords='data',
                        fontsize=10,
                        weight='bold',
                        color='red',
                        arrowprops=dict(
                            arrowstyle='->',
                            color='red',
                            lw=1.5
                        ),
                        zorder=8
                    )

            except Exception as e:
                self.logger.warning(f"估计整体坡面失败: {e}")
                # 如果不成功，不绘制坡面

            # 设置 stereonet 网格和标签
            ax.set_thetagrids(
                np.arange(0, 360, 30),
                labels=['N', '30°', '60°', 'E', '120°', '150°',
                        'S', '210°', '240°', 'W', '300°', '330°'],
                fontsize=10
            )

            # 设置半径网格和标签
            r_ticks = [0.25, 0.5, 0.75, 1.0]
            r_labels = ['15°', '30°', '45°', '60°']
            ax.set_rgrids(
                r_ticks, labels=r_labels,
                angle=135, fontsize=10
            )

            # 设置标题
            ax.set_title(
                f'Relationship between Dominant Discontinuities and Overall Slope\n({base_name})',
                fontsize=13, weight='bold', pad=20
            )

            # 添加图例
            ax.legend(
                loc='upper left',
                bbox_to_anchor=(1.05, 1.0),
                fontsize=10,
                frameon=True,
                fancybox=True,
                shadow=True,
                title='Discontinuity Clusters'
            )

            # 添加网格线
            ax.grid(True, alpha=0.3, linestyle='-')

            # 调整布局
            fig.tight_layout()

            # 保存图像
            self._SaveFigure(
                fig=fig,
                fig_dpi=self.dpi,
                out_dirs=[out_dir],
                base_name=base_name,
                plot_short_name=plot_short_name,
                output_formats=output_formats,
                logger=self.logger
            )

            if show:
                plt.show()
            else:
                plt.close(fig)
