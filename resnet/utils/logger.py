import logging
import torch
from torch.utils.tensorboard import SummaryWriter

def setup_logger(name=__name__):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 控制台处理器
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger

class TensorBoardLogger:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)
    
    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)