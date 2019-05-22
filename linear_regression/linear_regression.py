# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt


def load_npy_data(file_name):
    """
    加载numpy文件
    :param file_name: 文件名
    :return:
    """
    return np.load(file_name)


def load_txt_and_csv(file_name, delimiter, dtype):
    """
    加载txt和csv格式文件
    :param file_name: 文件名
    :param delimiter: 分隔符
    :param dtype: 数据类型
    :return:
    """
    return np.loadtxt(file_name, delimiter=delimiter, dtype=dtype)


def compute_cost(X, y, theta):
    """
    计算代价函数
    :param X: 样本矩阵
    :param y: 标签向量
    :param theta: 线性回归的系数向量
    :return: 代价函数值，即偏差平方和的均值
    """

    m = len(y)
    # 偏差平方和的均值
    J = (np.transpose(X * theta - y) * (X * theta - y)) / (2 * m)
    return J


def feature_normaliza(X):
    """
    归一化，每一列的数减去该列的平均值，再除以该列的标准差
    :param X:
    :return:
    """
    # 将X转为numpy数组，以便进行矩阵运算
    X_norm = np.array(X)

    # 求每一列的平均值，0代表列，1代表行
    mu = np.mean(X_norm, 0)
    # 计算每一列标准差
    sigma = np.std(X_norm, 0)
    for i in range(X.shape[1]):
        # 遍历进行归一化操作
        X_norm[:, 1] = (X_norm[:, 1] - mu[i]) / sigma[i]
    return X_norm, mu, sigma


def plot_2d(X):
    """
    画二维散点图
    :param X:
    :return:
    """
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()

