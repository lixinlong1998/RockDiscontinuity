import os
from typing import Dict, List, Optional, Sequence, Tuple, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .logging_utils import LoggerManager, Timer

PathTuple = Tuple[str, str]  # (out_dir, point_path)


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
            # 后续可以在此继续注册其它单一分析图件
        }

        self.compare_plot_meta: Dict[str, Dict[str, Callable]] = {
            "dipdir_rose": {
                "func": self._PlotDipDirectionRoseCompare,
                "short_name": "dipdir_rose_cmp",
            },
            # 后续可以在此继续注册其它对比分析图件
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
                fig.savefig(save_path, dpi=300)
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
            对单一结果(out_dir, point_path)绘制倾向(dip direction)玫瑰图,
            使用该结果目录下的 <basename>_discontinuitys.csv 中的 Dipdir 列。

            输出文件:
                对每个输出格式 fmt(如 ".png", ".svg"), 生成:
                    <out_dir>/<base_name>_<plot_short_name><fmt>
                例如:
                    Rock_GLS4_part1_localize_0.05m_dipdir_rose.png

        实现思路:
            1) 构造结构面级 CSV 路径:
                   disc_csv_path = out_dir / "<base_name>_discontinuitys.csv";
            2) 使用 pandas 读取 CSV, 提取 "Dipdir" 列并规范到 [0, 360) 区间;
            3) 以固定分箱宽度(如 10°)构造角度直方图, 计算每个扇区内的计数;
            4) 在极坐标系中绘制柱状玫瑰图:
                 - 0° 在北(N), 角度顺时针增加;
                 - 每 30° 一个角度刻度;
            5) 调用 _SaveFigure 将图像保存为多种格式。
        输入:
            out_dir: str
                当前结果目录。
            point_path: str
                当前点云路径(仅用于日志)。
            base_name: str
                点云 basename(不含扩展名)。
            plot_short_name: str
                图件简称, 如 "dipdir_rose"。
            output_formats: Sequence[str]
                输出格式列表(".png", ".svg" 等)。
            show: bool
                是否调用 plt.show() 显示图像。
        输出:
            无显式返回。
        """
        disc_csv_path = self._BuildDiscCsvPath(out_dir, base_name)
        if not os.path.isfile(disc_csv_path):
            self.logger.warning(
                f"_PlotDipDirectionRoseSingle: 未找到结构面 CSV, 跳过: {disc_csv_path}"
            )
            return

        with Timer(
                f"_PlotDipDirectionRoseSingle(base={base_name}, csv={os.path.basename(disc_csv_path)})",
                self.logger,
        ):
            try:
                df = pd.read_csv(disc_csv_path)
            except Exception as e:
                self.logger.warning(
                    f"_PlotDipDirectionRoseSingle: 读取 CSV 失败, 跳过: {disc_csv_path}, 错误: {e}"
                )
                return

            if "Dipdir" not in df.columns:
                self.logger.warning(
                    f"_PlotDipDirectionRoseSingle: 'Dipdir' 列不存在, 跳过: {disc_csv_path}"
                )
                return

            dipdir_deg = df["Dipdir"].to_numpy(dtype=float)
            dipdir_deg = np.mod(dipdir_deg, 360.0)

            if dipdir_deg.size == 0:
                self.logger.warning(
                    f"_PlotDipDirectionRoseSingle: Dipdir 数量为 0, 跳过: {disc_csv_path}"
                )
                return

            bin_size_deg = 10.0
            edges_deg = np.arange(0.0, 360.0 + bin_size_deg, bin_size_deg)
            width_rad = np.deg2rad(bin_size_deg)

            hist, _ = np.histogram(dipdir_deg, bins=edges_deg)
            centers_deg = edges_deg[:-1] + bin_size_deg * 0.5
            angles_rad = np.deg2rad(centers_deg)

            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, polar=True)

            ax.bar(
                angles_rad,
                hist,
                width=width_rad,
                bottom=0.0,
                alpha=0.7,
                edgecolor="k",
                label=base_name,
                align="center",
            )

            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.set_thetagrids(np.arange(0, 360, 30))

            ax.set_title(f"Dip Direction Rose ({base_name})", fontsize=14)
            ax.legend(loc="upper right", bbox_to_anchor=(1.1, 1.1))

            fig.tight_layout()

            self._SaveFigure(
                fig=fig,
                out_dirs=[out_dir],
                base_name=base_name,
                plot_short_name=plot_short_name,
                output_formats=output_formats,
                logger=self.logger,
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
