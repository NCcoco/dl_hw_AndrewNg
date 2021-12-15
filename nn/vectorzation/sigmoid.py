import numpy as np


def sigmoid(Z):
    # 向量版sigmoid函数
    return 1 / (1 + np.exp(-Z))
