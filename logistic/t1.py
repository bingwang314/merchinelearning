from math import exp
from numpy import mat, shape, ones


# 加载数据
def loadDataSet(path):
    dataMat = []
    labelMat = []
    fr = open(path)
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

# 计算响应值
def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))


def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).T
    m, n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.T * error
    return weights

if __name__ == '__main__':
    print("初始值：" + str(10))
    print("结果为：" + str(sigmoid(10)))