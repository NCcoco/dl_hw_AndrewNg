import numpy as np
import numpy.random as random
import sigmoid

n = 100
m = 100
X_train = np.array(shape=(m, n))
Y_train = np.array(shape=(m, 1))

# 随机权重
W = random.rand(1, n)
W *= 100
b = random.randint(100)


def iteration(X_train, Y_train, W, b, alpha):
    """
        向量式迭代参数
    Args:
        X_train ([type]): 训练集
        Y_train ([type]): 标签集或结果集
        W ([type]): 权重向量
        b ([type]): 偏导项
        alpha ([type]): 学习率
    Returns:
        [type]: 
    """
    # 计算输出
    # 这里的+b将会产生一个广播，这是python里面的一个语法糖 
    Z = np.dot(X_train, W.T) + b
    A = sigmoid(Z)
    # 计算损失
    J = - [Y_train * np.log(A) + (1 - Y_train) * np.log(1 - A)]
    # 计算均方误差 (感觉用不到)
    MSE = (1 / m) * np.sum(J**2)
    # 对z求导 为什么是这个【A - Y_train】？
    # A是当前模型的输出值，然后Y是真实值，这样可以算出差
    # 然后呢？这个差为什么是dz，（dz是对z求导）
    # 这里省略了一步， 那一步是计算均方误差，我们定义均方误差方程为：J = - [y * log(a) + (1 - y) * log(1 - a)]
    #       那么，它的导数是什么呢？ 因为是对z求导，我们知道 z = Xw 的点积 + b
    #       而a是sigmoid(z) 的结果。因此 dJ/dz = dJ/da * da/dz 链式法则
    #       dJ/da = -[y/a + (1-y)/(1-a)]
    #       da/dz = a(1-a)
    #       dJ/dz = a - y
    # 至此，推导完成，因此 A-Y_train就是dz
    dz = A - Y_train
    # dJ/dw = dJ/dz * dz/dw (微分代数)
    # w全导数
    dw = (1 / m) * np.dot(X_train, dz)
    # b全导数
    db = (1 / m) * np.sum(dz)
    # 修正参数
    W = W - alpha * dw
    b = b - alpha * db
    return W, b, alpha
