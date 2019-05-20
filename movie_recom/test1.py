# -*- coding=utf-8 -*-
import math
import random

# 每次实验选取不同的k(0≤k≤m-1)和相同的随机数种子seed，
# 进行M次实验就可以得到m个不同的训练集和测试集，
# 然后分别进行实验，用m次实验的平均值作为最后的评测指标。
# 这样做主要是防止某次实验的结果是过拟合的结果(over fitting)，
# 但如果数据集够大，模型够简单，为了快速通过离线实验初步地选择算法，
# 也可以只进行一次实验。
from _operator import itemgetter


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


def recall(train, test, N):
    """
    计算召回率
    :param train: 训练集
    :param test: 测试集
    :param N: 推荐的物品数量
    :return: 召回率
    """

    hit = 0  # 命中数，即推荐的正好是喜欢的
    all = 0  # 对所有用户推荐的全部商品数量
    for user in train.keys():
        tu = test[user]  # user在测试集上喜欢的物品集合
        rank = get_recommendation(user, N)  # 推荐给user的商品集合
        for item, pui in rank:
            if item in tu:
                hit += 1
        all += len(tu)  # 所有用户喜欢的商品总数量
    return hit / (all * 1.0)


def precision(train, test, N):
    """
    计算准确率
    :param train: 训练集
    :param test: 测试集
    :param N: 推荐的商品数
    :return: 准确率
    """

    hit = 0  # 命中数，即推荐的正好是喜欢的
    all = 0  # 对所有用户推荐的全部商品数量
    for user in train.keys():
        tu = test[user]  # user在测试集上喜欢的物品集合
        rank = get_recommendation(user, N)  # 推荐给user的商品集合
        for item, pui in rank:
            if item in tu:
                hit += 1
        all += N  # 给所有用户推荐的商品总数量
    return hit / (all * 1.0)


def coverage(train, test, N):
    """
    计算覆盖率
    :param train: 训练集
    :param test: 测试集
    :param N: 推荐的商品数
    :return: 覆盖率
    """

    recommend_items = set()  # 推荐的所有的商品集合
    all_items = set()  # 所有的商品集合
    for user in train.keys():
        for item in train[user].keys():
            all_items.add(item)  # 训练集中的用户喜欢的商品
        rank = get_recommendation(user, N)  # 为该用户推荐的商品集
        for item, pui in rank:
            recommend_items.add(item)  # 对每一个用户推荐的商品
    return len(recommend_items) / (len(all_items) * 1.0)


def popularity(train, test, N):
    """N
    计算新颖度
    用推荐列表中物品的平均流行度度量推荐结果的新颖度。
    如果推荐出的物品都很热门，说明推荐的新颖度较低，否则说明推荐结果比较新颖。
    :param train: 训练集
    :param test: 测试集
    :param N: 推荐商品数
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
    n = 0
    # 累加计算所有用户所有推荐商品的热度和
    for user in train.keys():
        rank = get_recommendation(user, N)
        for item, pui in rank:
            ret += math.log(1 + item_popularity[item])
            n += 1
    ret /= n * 1.0
    return ret


def user_similarity(train):
    """
    计算用户相似度
    :param train: 训练集
    :return:
    """

    item_users = dict()  # 物品到用户的倒排表，保存对每个物品产生过行为的用户列表
    for u, items in train.items():  # 对每个用户和对应的物品集合循环
        for i in items.keys():  # 每个物品的key
            if i not in item_users:  # 判断该物品对应的用户集合是否初始化
                item_users[i] = set()  # 初始化物品对应的用户集合
            item_users[i].add(u)  # 为倒排表中该物品添加用户

    C = dict()
    N = dict()
    for i, users in item_users.items():  # 计算每两个用户之间是共同感兴趣的物品的个数
        for u in users:
            N[u] += 1  # 每个用户出现的次数
            for v in users:
                if u == v:
                    continue
                C[u][v] += 1  # 两个用户之间共同感兴趣的物品数+1

    W = dict()
    for u, related_users in C.items():
        for v, cuv in related_users.items():
            W[u][v] = cuv / math.sqrt(N[u] * N[v])  # 每两个用户共同感兴趣物品的数量，除以各自感兴趣物品数量的乘机开方
    return W


def recommend(user, train, W, K):
    """
    基于用户相似度的推荐算法
    :param user: 为该用户推荐
    :param train: 训练集
    :param W:
    :param K:
    :return:
    """
    rank = dict()
    interacted_items = train[user]
    for v, wuv in sorted(W[user].items, key=itemgetter(1), reverse=True)[0: K]:
        for i, rvi in train[v].items:
            if i in interacted_items:
                # we should filter items user interacted before
                continue
            rank[i] += wuv * rvi
    return rank


def get_recommendation(user, n):
    """
    推荐核心算法
    :param user: 用户id
    :param n: 推荐的商品数
    :return: 推荐的商品集合
    """

    return []
