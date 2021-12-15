# 感谢世界提供的公共资源
import pandas as pd
import numpy as np

# 加载泰坦尼克数据作为分析（向逝者的不幸表示哀悼）
train_df = pd.read_csv('titanic/train.csv')
test_df = pd.read_csv('titanic/test.csv')
"""
找出缺失值
"""
# print(train_df.head(10))
k = train_df.isnull().sum()
print(k)
# 算出每一行空值的比例
m = len(train_df)
print(k / m)
# 保留两位小数
print(np.round(k / m, 2))
