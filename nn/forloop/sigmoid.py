import numpy as np
import math
import matplotlib.pyplot as plt

def sigmoid(z):
    """
    激活函数 sigmoid
    """
    return 1 / (1 + math.exp(-z))


def sigmoid_derivative(z):
    """
    激活函数的导函数
    """
    k = sigmoid(z)
    return k * (1 - k)

# 绘制激活函数图像
x = np.linspace(-6, 6, 200)
y = []
for i in x:
    y.append(sigmoid(i))
plt.plot(x, y, c="red")
plt.grid(True)
plt.show()
