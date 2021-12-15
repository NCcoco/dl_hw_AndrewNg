import numpy as np
import numpy.random as random
# 一个典型的错误的样例，
a = random.randint(1, 100, size=(10, 1))
b = random.randint(1, 100, size=(1, 10))
# 期待它能报错，但是该示例会出现意想不到的结果
print(a + b)