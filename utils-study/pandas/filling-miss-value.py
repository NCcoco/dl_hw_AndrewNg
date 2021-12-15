# 感谢世界提供的公共资源
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# 加载泰坦尼克数据作为分析（向逝者的不幸表示哀悼）
train_df = pd.read_csv('titanic/train.csv')
test_df = pd.read_csv('titanic/test.csv')

"""
填充缺失值
"""
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
train_df[["Age"]] = imputer.fit_transform(train_df[["Age"]])
k = train_df.isnull().sum()
# 算出每一行空值的比例
m = len(train_df)
# 保留4位小数
print(np.round(k / m, 4))

"""
填充缺失值的统计
"""
print(imputer.statistics_)
"""
使用常数填充/ 使用最频繁的值填充
"""
# imputer2 = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='C')
imputer2 = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
train_df[["Embarked"]] = imputer2.fit_transform(train_df[["Embarked"]])
k = train_df.isnull().sum()
# 保留4位小数
print(np.round(k / m, 3))
print(imputer2.statistics_)
