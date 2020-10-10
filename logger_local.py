import os
import logging
import time


# from param import log_path

class logger_self:
    __species = None
    __first_init = True

    def __new__(cls, *args, **kwargs):
        if cls.__species == None:
            cls.__species = object.__new__(cls)
        return cls.__species

    def __init__(self, set_level='info', file_path='', use_console=True):
        """
            set_level： 设置日志的打印级别，默认为DEBUG
            name： 日志中将会打印的name，默认为运行程序的name
            log_name： 日志文件的名字，默认为当前时间（年-月-日.basic_log）
            log_path： 日志文件夹的路径，默认为logger.py同级目录中的log文件夹
            use_console： 是否在控制台打印，默认为True

        """
        if self.__first_init:
            # self.name = name
            self.__class__.__first_init = False

        # 创建一个logger
        self.logger = logging.getLogger(__name__)

        # DesktopPath = get_desktop()
        # 设置日志等级
        if set_level.lower() == "critical":
            self.logger.setLevel(logging.CRITICAL)
        elif set_level.lower() == "error":
            self.logger.setLevel(logging.ERROR)
        elif set_level.lower() == "warning":
            self.logger.setLevel(logging.WARNING)
        elif set_level.lower() == "info":
            self.logger.setLevel(logging.INFO)
        elif set_level.lower() == "debug":
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.NOTSET)
        now = time.strftime('%Y_%m_%d_%H', time.localtime(time.time()))
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        logpathname = str(now) + '_backtest.log'
        logname = file_path + logpathname
        if not self.logger.handlers:
            log_handler = logging.FileHandler(logname)
            log_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(pathname)s[:%(lineno)d] -  %(name)s - %(levelname)s: %(message)s'))
            self.logger.addHandler(log_handler)
            if use_console:
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(logging.Formatter(
                    '%(asctime)s - %(pathname)s[:%(lineno)d] -  %(name)s - %(levelname)s: %(message)s'))
                self.logger.addHandler(console_handler)

    def addHandler(self, hdlr):
        self.logger.addHandler(hdlr)

    def removeHandler(self, hdlr):
        self.logger.removeHandler(hdlr)

    def InItlogger(self):
        return self.logger
