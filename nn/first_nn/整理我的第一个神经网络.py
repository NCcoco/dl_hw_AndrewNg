import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model

from planar_utils import load_extra_datasets, plot_decision_boundary, sigmoid
from testCases_v2 import *


## 写一个生成2维数据的函数
def load_planar_dataset():
    np.random.seed(1)
    # 定义数据集的行数
    m = 400
    N = int(m / 2)
    # 定义特征维度
    D = 2
    X = np.zeros(shape=(m, D))
    Y = np.zeros(shape=(m, 1), dtype='uint8')
    a = 4

    for j in range(2):
        ix = range(N * j, N * (j + 1))
        t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2
        r = a * np.sin(4 * t) + np.random.randn(N) * 0.2
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[ix] = j

    X = X.T
    Y = Y.T
    return X, Y


# 加载数据
X, Y = load_planar_dataset()
# 可视化数据 (X的0行作为x轴，1行作y轴，c是颜色下标，cmap是颜色值列表。)
# plt.scatter(X[0, :], X[1, :], c=Y[0, :], s=40, cmap=plt.cm.Spectral)
# plt.show()

clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X.T, Y.T)

plot_decision_boundary(lambda x: clf.predict(x), X, Y[0, :])
# plt.title("linear_model decision boundary")
# plt.show()

LR_predictions = clf.predict(X.T)
print('线性回归正确率: %d ' %
      float((np.dot(Y, LR_predictions) + np.dot(1 - Y, 1 - LR_predictions)) /
            float(Y.size) * 100) + '% ' +
      "(percentage of correctly labelled datapoints)")


# 定义输入层，隐层，输出层的神经元个数
# 输入层的神经元必须与输入数据的维度一致。隐层神经元个数可以自由制定，但是必须合适，输出神经元必须与标签类别数量一致，（除非是2分类，只要一个输出单元）
def layer_sizes(X, Y):
    """
    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)
    
    Returns:
    n_x -- the size of the input layer
    n_h -- the size of the hidden layer
    n_y -- the size of the output layer
    """
    ### START CODE HERE ### (≈ 3 lines of code)
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]
    ### END CODE HERE ###
    return (n_x, n_h, n_y)


(n_x, n_h, n_y) = layer_sizes(X, Y)
print(n_x, n_h, n_y)
# 使用吴老师提供的数据生成函数
X_assess, Y_assess = layer_sizes_test_case()
(n_x, n_h, n_y) = layer_sizes(X_assess, Y_assess)
print(n_x, n_h, n_y)


def initialize_parameters(n_x, n_h, n_y):
    """
    随机初始化参数（不包含学习率超参数）

    Args:
        n_x ([type]): [description]
        n_h ([type]): [description]
        n_y ([type]): [description]
    """

    np.random.seed(2)
    # 因为sigmod或tanh函数在小值范围内梯度较大，为了快速迭代，必须使用较小值进行训练
    W1 = np.random.randn(n_h, n_x) * 0.01
    W2 = np.random.randn(n_y, n_h) * 0.01
    # np.dot(W1, X) 是一个4*m矩阵 那么b的维度为 4*1 ，其余m-1列的值与第一列一致，在做加法时直接利用python的广播效果
    #       4*2 2*m
    b1 = np.random.randn(n_h, 1)
    # np.dot(W2, A1) 是一个1*m矩阵 那么b的维度为 1 ，其余m-1列的值与第一列一致，在做加法时直接利用python的广播效果
    #       1*4  4*m
    b2 = np.random.randn(n_y, 1)

    parameter = {"W1": W1, "W2": W2, "b1": b1, "b2": b2}

    return parameter


# 测试一下生成的效果
parameter = initialize_parameters(n_x, n_h, n_y)
# print(parameter)
print(parameter["W1"].shape)
print(parameter["W2"].shape)
print(parameter["b1"].shape)
print(parameter["b2"].shape)


## 有了权重，就可以开始前向传播了
def forward_propagation(X, parameter):
    """
    进行前向传播计算

    Args:
        X ([type]): 训练集
    """
    W1 = parameter["W1"]
    W2 = parameter["W2"]
    b1 = parameter["b1"]
    b2 = parameter["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}

    return A2, cache

X_assess, parameters = forward_propagation_test_case()
A2, cache = forward_propagation(X_assess, parameters)
# 看看目前能算出什么
print(cache)

# 计算损失：
