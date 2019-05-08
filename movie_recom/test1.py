# -*- coding=utf-8 -*-
import random

# 每次实验选取不同的k(0≤k≤m-1)和相同的随机数种子seed，
# 进行M次实验就可以得到m个不同的训练集和测试集，
# 然后分别进行实验，用m次实验的平均值作为最后的评测指标。
# 这样做主要是防止某次实验的结果是过拟合的结果(over fitting)，
# 但如果数据集够大，模型够简单，为了快速通过离线实验初步地选择算法，
# 也可以只进行一次实验。


def split_data(data, m, k, seed):
    """
    产生训练集与测试集
    :param data: 数据集
    :param m: 划分数据集的份数
    :param k: 第k份数据作为测试集
    :param seed: 随机种子
    :return: 训练集，测试集
    """

    test = []
    train = []
    random.seed(seed)
    for user, item in data:
        if random.randint(0, m) == k:
            test.append((user, item))
        else:
            train.append((user, item))
    return train, test
