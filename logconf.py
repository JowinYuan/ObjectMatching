import logging
import logging.handlers

# 获取 root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

#清除已有 handler，防止重复输出或不同配置冲突。
for handler in list(root_logger.handlers):
    root_logger.removeHandler(handler)

# 设置日志格式
logfmt_str = "%(asctime)s %(levelname)-8s pid:%(process)d %(message)s"
formatter = logging.Formatter(logfmt_str)

# 控制台日志输出 handler
streamHandler = logging.StreamHandler()
streamHandler.setFormatter(formatter)
streamHandler.setLevel(logging.DEBUG)

# 添加 handler 到 logger
root_logger.addHandler(streamHandler)