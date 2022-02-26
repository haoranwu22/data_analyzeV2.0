# 端到端机器学习---深度学习
# 损失函数
# 梯度法
import numpy as np

"""
# 单个数据的均方误差
def mean_squared_error(y, t):
    return 0.5*np.sum((y-t)**2)
"""


# mini_batch班MSE
def mean_squared_error(y, t):
    batch_size = y.shape[0]
    return (np.sum((y-t)**2)/batch_size)


"""
# 单个数据交叉熵误差
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t*np.log(y + delta))  # 防止出现负无限大
"""


# mini-batch交叉熵误差
def cross_entropy_error(y, t):
    # 输入若为一维数组变为二维数组
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)
    batch_size = y.shape[0]
    delta = 1e-7
    return -np.sum(t*np.log(y + delta)) / batch_size  # one-hot表示
#    return -np.sum(np.log(y[np.arange(batch_size), t] + delta))
#    / batch_size  # 非 one-hot表示
