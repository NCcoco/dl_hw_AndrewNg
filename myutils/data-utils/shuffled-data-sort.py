import numpy as np

# 随机打乱数据排序

def suffled_data_sort_2D_by_column(X):
    """
        按列随机排序整个数据集
    Args:
        X ([type]): [description]

    Returns:
        [type]: [description]
    """
    # 获取所有的列数量
    count = X.shape[1]
    s = list(np.random.permutation(count))
    # 按s对列进行排序
    suffled_X = X[:, s]
    return suffled_X
    
def suffled_data_sort_2D_by_row(X):
    """
        按行随机排序整个数据集
    Args:
        X ([type]): [description]

    Returns:
        [type]: [description]
    """
    # 获取所有的行数量
    count = X.shape[0]
    s = list(np.random.permutation(count))
    # 按s对列进行排序
    suffled_X = X[s, :]
    return suffled_X

