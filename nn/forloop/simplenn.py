import numpy as np
import pandas as pd
import sigmoid
import random

dataset = np.array([[1,2],[3,4],[5,6]])
Y = np.array([0,1,1])
# print(dataset[1:1+1,1])

# 随机生成W
w1 = random.randint(10, 100)
w2 = random.randint(10, 100)
b = random.randint(10, 100)


