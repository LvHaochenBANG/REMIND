"""Shared filesystem, logging and retrieval-similarity helpers."""
import os
import numpy as np


def get_workspace():
    """
    get the workspace path, i.e., the root directory of the project
    """
    cur_path = os.path.abspath(__file__)
    file = os.path.dirname(cur_path)
    file = os.path.dirname(file)
    return file


def dir_check(path):
    """
    check weather the dir of the given path exists, if not, then create it
    """
    import os
    dir = path if os.path.isdir(path) else os.path.split(path)[0]
    if not os.path.exists(dir): os.makedirs(dir)
    return path


class Logger(object):
    def __init__(self):
        import sys
        self.terminal = sys.stdout  #stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None: mode ='w'
        self.file = open(file, mode)

    def write(self, message, is_terminal=True, is_file=True):
        if '\r' in message: is_file=False

        if is_terminal:
            self.terminal.write(message)
            self.terminal.flush()

        if is_file:
            self.file.write(message)
            self.file.flush()


def exp_softmax(neg_mae, tau=1):
    """
    计算 exp(-MAE / tau) ，可用于根据MAE计算权重，相似 run_retrieval.py 的权重机制.
    
    参数:
        mae: array-like, MAE 值，可以为标量或 numpy 数组
        tau: 标量，温度参数（平滑系数）
    返回:
        权重: numpy 数组，和 mae 形状一致
    """
    return np.exp(neg_mae / tau)
