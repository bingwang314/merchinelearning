# -*- coding=utf-8 -*-
import math
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


def recall(train, test, n):
    """
    计算召回率
    :param train: 训练集
    :param test: 测试集
    :param n: 推荐的物品数量
    :return: 召回率
    """

    hit = 0                                             # 命中数，即推荐的正好是喜欢的
    all = 0                                             # 对所有用户推荐的全部商品数量
    for user in train.keys():
        tu = test[user]                                 # user在测试集上喜欢的物品集合
        rank = get_recommendation(user, n)              # 推荐给user的商品集合
        for item, pui in rank:
            if item in tu:
                hit += 1
        all += len(tu)                                  # 所有用户喜欢的商品总数量
    return hit / (all * 1.0)


def precision(train, test, n):
    """
    计算准确率
    :param train: 训练集
    :param test: 测试集
    :param n: 推荐的商品数
    :return: 准确率
    """

    hit = 0                                             # 命中数，即推荐的正好是喜欢的
    all = 0                                             # 对所有用户推荐的全部商品数量
    for user in train.keys():
        tu = test[user]                                 # user在测试集上喜欢的物品集合
        rank = get_recommendation(user, n)              # 推荐给user的商品集合
        for item, pui in rank:
            if item in tu:
                hit += 1
        all += n                                        # 给所有用户推荐的商品总数量
    return hit / (all * 1.0)


def coverage(train, test, n):
    """
    计算覆盖率
    :param train: 训练集
    :param test: 测试集
    :param n: 推荐的商品数
    :return: 覆盖率
    """

    recommend_items = set()                             # 推荐的所有的商品集合
    all_items = set()                                   # 所有的商品集合
    for user in train.keys():
        for item in train[user].keys():
            all_items.add(item)                         # 训练集中的用户喜欢的商品
        rank = get_recommendation(user, n)              # 为该用户推荐的商品集
        for item, pui in rank:
            recommend_items.add(item)                   # 对每一个用户推荐的商品
    return len(recommend_items) / (len(all_items) * 1.0)


def popularity(train, test, n):
    """
    计算新颖度
    用推荐列表中物品的平均流行度度量推荐结果的新颖度。
    如果推荐出的物品都很热门，说明推荐的新颖度较低，否则说明推荐结果比较新颖。
    :param train: 训练集
    :param test: 测试集
    :param n: 推荐商品数
    :return: 新颖度
    """
    item_popularity = dict()
    # 计算所有训练集中每个商品的流行度，即每个商品被多少人喜欢
    for user, items in train.items():
        for item in items.keys():
            if item not in item_popularity:
                item_popularity[item] = 0
            item_popularity[item] += 1
    ret = 0
    num = 0
    for user in train.keys():
        rank = get_recommendation(user, n)
        for item, pui in rank:
            ret += math.log(1 + item_popularity[item])
            num += 1
    ret /= num * 1.0
    return ret



def get_recommendation(user, n):
    """
    推荐核心算法
    :param user: 用户id
    :param n: 推荐的商品数
    :return: 推荐的商品集合
    """

    return []
