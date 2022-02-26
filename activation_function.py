# 与非门感知机
# 简单神经网络
# 常见激活函数
import numpy as np
# import matplotlib.pylab as plt


def AND(x1, x2):  # 与
    x = np.array([x1, x2])  # 输入
    w = np.array([0.5, 0.5])  # 权重
    b = -0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def NAND(x1, x2):  # 与非
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def OR(x1, x2):  # 或
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.3
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def XOR(x1, x2):  # 异或
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)  # 第一层
    y = AND(s1, s2)  # 第二层
    return y


def sigmoid(x):  # sigmoid激活函数（二分类问题常用）
    return 1 / (1 + np.exp(-x))


def step_function(x):  # 阶跃函数
    y = x > 0
    return y.astype(np.int)  # 转换numpy数组的类型


def relu(x):  # relu函数
    return np.maximum(0, x)


def identity_function(x):  # 恒等函数（回归问题常用）
    return x


# def softmax(a):  # softmax函数（多分类问题常用）
#     c = np.max(a)
#     exp_a = np.exp(a - c)
#     sum_exp_a = np.sum(exp_a)
#     y = exp_a/sum_exp_a
#     return y
def softmax(x):
    # mini-batch专用
    if x.ndim == 2:
        # 求转置的原因：https://blog.csdn.net/qq_18433441/article/details/56834207
        x = x.T
        x = x - np.max(x, axis=0)  # 列方向
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T
    # 一维数组形式的输出数据专用
    x = x - np.max(x)  # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)
