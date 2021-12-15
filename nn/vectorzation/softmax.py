import numpy as np

def softmax(x):
    # softmax函数
    
    # 计算所有行的自然指数之和
    x_exp = np.exp(x)
    x_exp_sum = np.sum(x_exp, axis=1)
    # shape矫正
    x_exp_sum = x_exp_sum.reshape(len(x), 1)
    # 归一 (利用python广播原理)
    s = x_exp / x_exp_sum
    return s

x = np.random.randint(0, 10, size=(10,10))
print(softmax(x))
