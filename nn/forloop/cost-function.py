import numpy as np
import sigmoid

"""
定义损失函数
"""
def costfun(x, y):
    a = sigmoid(x)
    j = -( y * np.log(a) + (1 - y) * np.log(1 - a))
    return j

"""
损失函数的导函数
"""
def costfun_derivative(x, y):
    return sigmoid(x) - y
