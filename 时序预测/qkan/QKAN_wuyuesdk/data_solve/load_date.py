import pandas as pd
import torch
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from io import StringIO


# ========== 数据预处理 ==========
def prepare_ts_for_model(data, date_format='%m/%d/%y %H:%M', freq='5T',
                         window_size=3, forecast_steps=1, test_ratio=0.2):
    """生成B×N×L格式的数据"""
    # 读取数据
    df = pd.read_csv(
        data,
        parse_dates=['LocalTime'],
        date_parser=lambda x: pd.to_datetime(x, format=date_format)
    ).set_index('LocalTime').asfreq(freq).fillna(0.0)

    # 生成监督学习数据 (B, N, L)
    X, y = [], []
    for i in range(len(df) - window_size - forecast_steps + 1):
        # 将窗口数据调整为 (N, L) 格式
        seq = df.iloc[i:i + window_size, 0].values.reshape(1, -1)  # (N=1, L)
        label = df.iloc[i + window_size:i + window_size + forecast_steps, 0].values
        X.append(seq)
        y.append(label)

    # 转换为DataFrame并划分数据集
    X = pd.DataFrame([x.flatten() for x in X])  # 展平用于存储
    y = pd.DataFrame(y)
    test_size = int(len(X) * test_ratio)
    return (
        X.iloc[:-test_size], X.iloc[-test_size:],
        y.iloc[:-test_size], y.iloc[-test_size:]
    )


# ========== PyTorch Dataset ==========
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        # 转换为 (B, N, L) 格式
        self.X = torch.tensor(X.values, dtype=torch.float32).unsqueeze(1)  # (B, N=1, L)
        self.y = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ========== DataLoader 创建 ==========
def create_loaders(X_train, X_test, y_train, y_test, batch_size=32):
    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


# ========== 端到端流程 ==========
def full_pipeline(data, window_size=3, forecast_steps=1, test_ratio=0.2, batch_size=32):
    X_train, X_test, y_train, y_test = prepare_ts_for_model(
        data, window_size=window_size, forecast_steps=forecast_steps, test_ratio=test_ratio
    )
    return create_loaders(X_train, X_test, y_train, y_test, batch_size)

def read_power_data(file_path, parse_dates=True, index_col='LocalTime', date_format="%m/%d/%y %H:%M"):
    """
    读取电力数据文件并返回DataFrame

    参数:
        file_path (str): 文件路径
        parse_dates (bool): 是否将时间列解析为日期类型，默认为True
        index_col (str): 设置为索引的列名，默认为'LocalTime'
        date_format (str): 日期解析格式

    返回:
        pd.DataFrame: 包含时间和功率数据的DataFrame
    """
    # 读取数据（假设文件名为 "文件1.csv"）
    df = pd.read_csv(file_path)
    data = df.iloc[:, 1:].values  # 忽略时间戳列，形状 [T, 140]

    # 参数定义
    L = 60*24//5  # 每个样本的时间步长（需自定义）
    T = data.shape[0]  # 总时间步数
    N = data.shape[1]
    B = T // L  # 计算批次数量

    # 截断数据以对齐 B*L
    data_truncated = data[:B * L, :]

    # 重塑为 (B, N, L)
    data_reshaped = data_truncated.reshape(B, N, L)


    return data_reshaped


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def prepare_training_data(csv_path,
                          time_steps=24,
                          pred_steps=1,
                          test_size=0.2,
                          random_state=42):
    """
    完整数据预处理流程

    参数:
    csv_path : str           数据文件路径
    time_steps : int         输入序列长度 (L)
    pred_steps : int         预测步长 (单步/多步预测)
    test_size : float        测试集比例
    random_state : int       随机种子

    返回:
    (X_train, y_train), (X_test, y_test)  # 形状自动对齐深度学习框架
    """
    # 读取并解析数据
    df = pd.read_csv(csv_path, parse_dates=['LocalTime'])
    power_series = df['Power(MW)'].values.astype(np.float32)

    # 数据归一化 (扩展为可选不同归一化方式)
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_data = scaler.fit_transform(power_series.reshape(-1, 1)).flatten()

    # 生成时序样本（特征-标签对）
    X, y = [], []
    for i in range(len(normalized_data) - time_steps - pred_steps + 1):
        X.append(normalized_data[i:i + time_steps])
        y.append(normalized_data[i + time_steps:i + time_steps + pred_steps])

    X = np.array(X)
    y = np.array(y)

    # 转换为3D张量 [samples, features, timesteps]
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))  # N=1 单特征
    y = np.reshape(y, (y.shape[0], pred_steps))  # 多步预测保持二维

    # 时序安全划分（避免打乱时序）
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    return (X_train, y_train), (X_test, y_test), scaler



def load_data(data_path):
    data = pd.read_csv(data_path)
    X = data.iloc[:, :-1].values  # 特征
    y = data.iloc[:, -1].values   # 目标变量
    # 标准化处理
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return torch.FloatTensor(X), torch.FloatTensor(y)
# 示例用法
if __name__ == "__main__":


    data_file = '../dataset/2019.xlsx'

    train_loader, test_loader = full_pipeline(
        data_file, window_size=288, batch_size=356
    )

    for X_batch, y_batch in train_loader:
        print(f"输入维度: {X_batch.shape} (B, N, L)")
        print(f"标签维度: {y_batch.shape} (B, forecast_steps)")
        break