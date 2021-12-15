import numpy as np
import time
"""
对比向量化的运算和for循环运算的耗时
"""

a = np.random.rand(1000000)
b = np.random.rand(1000000)

tic = time.time()
c = np.dot(a, b)
toc = time.time()

print(c)
k1 = (toc - tic) * 1000
print("向量版耗时" + str(k1) + "ms")

c = 0
# for
tic = time.time()
for i in range(0, 1000000):
    c += a[i] * b[i]

toc = time.time()
print(c)
k2 = (toc - tic) * 1000
print("for循环版耗时" + str(k2) + "ms")

print("向量版是循环版的" + str(k2 / k1) + "倍")
