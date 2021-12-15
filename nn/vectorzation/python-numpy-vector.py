import numpy as np
import numpy.random as random

# 生成10个符合高斯分布的数组
a = random.randn(10)
print(a)
print(a.shape)  # (10,)
print(a.T)  # 这里的问题就是，a与a.T是一样的，这不符合数学格式，虽然我们让它进行点积运算结果是正常的

# 一个好的编程建议是，在生成数据时，或者转换数据时，就让我们的数据符合向量或矩阵的格式
# 如果要生成向量则：
a = random.randn(10, 1)
print(a)
print(a.shape)
print(a.T)
# 还有一种声明向量的方式
assert (a.shape == (10, 1))
# 或者转换shape
a = a.reshape(1,10)
print(a)
