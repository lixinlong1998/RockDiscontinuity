import logging
import time
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

    @classmethod
    def Initialize(cls, level: int = logging.INFO) -> None:
        if not cls._initialized:
            logging.basicConfig(
                level=level,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            cls._initialized = True

    @classmethod
    def GetLogger(cls, name: str) -> logging.Logger:
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
