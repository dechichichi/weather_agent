import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def min_max_normalization(data, columns):
    """
    使用最小 - 最大缩放对指定列进行标准化
    :param data: 输入数据，pandas DataFrame 格式
    :param columns: 需要标准化的列名列表
    :return: 标准化后的数据，pandas DataFrame 格式
    """
    scaler = MinMaxScaler()
    data[columns] = scaler.fit_transform(data[columns])
    return data

def z_score_normalization(data, columns):
    """
    使用 z - 分数标准化对指定列进行标准化
    :param data: 输入数据，pandas DataFrame 格式
    :param columns: 需要标准化的列名列表
    :return: 标准化后的数据，pandas DataFrame 格式
    """
    scaler = StandardScaler()
    data[columns] = scaler.fit_transform(data[columns])
    return data