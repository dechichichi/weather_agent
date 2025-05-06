import pandas as pd

def extract_features(data):
    """
    从原始数据中提取特征
    :param data: 原始数据，pandas DataFrame 格式
    :return: 提取特征后的数据，pandas DataFrame 格式
    """
    # 提取季节特征
    data['season'] = pd.to_datetime(data['date']).dt.month % 12 // 3 + 1
    
    # 提取温度范围特征
    data['temperature_range'] = data['max_temperature'] - data['min_temperature']
    
    return data