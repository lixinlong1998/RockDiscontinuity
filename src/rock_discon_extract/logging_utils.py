import logging
import time
import os
from datetime import datetime
from typing import Optional


class LoggerManager:
    """
    功能简介:
        统一管理整个项目中的日志配置, 提供简单的日志记录接口.

    实现思路:
        使用 Python 内置 logging 库, 在初始化时配置日志格式与输出级别.
        提供 GetLogger 接口按模块名获取 logger.
    """

    _initialized: bool = False
    _file_handler: Optional[logging.Handler] = None  # 新增: 记录当前文件 handler

    @classmethod
    def CreatLogFile(cls, result_path: str) -> None:
        """
        创建log文件
        """
        logs_dir = os.path.join(result_path, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        LoggerManager.AddFileHandler(os.path.join(logs_dir, f"{start_time}_log.txt"), level=logging.INFO)

    @classmethod
    def Initialize(cls, level: int = logging.INFO) -> None:
        """
        功能简介:
            初始化全局 logging 配置(仅执行一次), 设置日志级别与基础输出格式.

        实现思路:
            - 调用 logging.basicConfig 配置 root logger 的基本行为:
                * level 决定最低输出级别;
                * format 统一控制日志输出格式。
            - 使用 _initialized 变量保证整个进程中只初始化一次。
        """
        if not cls._initialized:
            logging.basicConfig(
                level=level,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            cls._initialized = True

    @classmethod
    def AddFileHandler(cls, log_file_path: str, level: int = logging.INFO) -> None:
        """
        功能简介:
            为全局 root logger 追加一个文件日志 handler, 将日志写入指定文件。

        实现思路:
            1) 若尚未调用 Initialize, 先以给定 level 初始化 logging;
            2) 若之前已经添加过文件 handler, 先从 root logger 移除并关闭,
               避免重复写入或文件句柄泄露;
            3) 使用 logging.FileHandler 创建新的文件 handler:
               - mode="w" 表示每次运行覆盖旧文件(由调用方通过路径控制多次运行的文件名);
               - encoding="utf-8" 保证中文日志正常写入;
            4) 使用与控制台相同的 formatter, 保持日志格式统一;
            5) 将该 handler 添加到 root logger 上, 并记录在 _file_handler 中。

        输入:
            log_file_path: str
                日志文件的完整路径。
            level: int
                文件日志的输出级别, 默认 logging.INFO。

        输出:
            无显式返回。
        """
        # 确保 logging 至少初始化一次
        if not cls._initialized:
            cls.Initialize(level=level)

        root_logger = logging.getLogger()

        # 若之前已经存在文件 handler, 先移除并关闭
        if cls._file_handler is not None:
            root_logger.removeHandler(cls._file_handler)
            try:
                cls._file_handler.close()
            except Exception:
                pass
            cls._file_handler = None

        # 创建新的文件 handler
        file_handler = logging.FileHandler(log_file_path, mode="w", encoding="utf-8")
        file_handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)

        root_logger.addHandler(file_handler)
        cls._file_handler = file_handler

    @classmethod
    def GetLogger(cls, name: str) -> logging.Logger:
        """
        功能简介:
            按模块名/类名获取 logger 对象。

        实现思路:
            - 若尚未初始化 logging, 先调用 Initialize() 使用默认配置;
            - 然后使用 logging.getLogger(name) 返回对应的 logger。

        输入:
            name: str
                logger 名称, 一般为模块名或类名。

        输出:
            logger: logging.Logger
                对应名称的 logger 实例。
        """
        if not cls._initialized:
            cls.Initialize()
        return logging.getLogger(name)


class Timer:
    """
    功能简介:
        简单的计时工具类, 用于统计某个代码块或函数的运行时间.

    实现思路:
        使用上下文管理器, 在进入时记录起始时间, 退出时计算耗时并日志输出.
    """

    def __init__(self, name: str, logger: Optional[logging.Logger] = None):
        self.name = name
        self.logger = logger
        self.start_time: float = 0.0
        self.duration: float = 0.0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.duration = time.time() - self.start_time
        if self.logger:
            self.logger.info(f"[Timer] {self.name} took {self.duration:.3f} s")
