import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# def excel_to_sequences(file_path,
#                        target_col='实际发电功率(mw)',
#                        date_col='时间',
#                        sheet_name='sheet1',
#                        freq='15T',
#                        window_size=24,
#                        forecast_steps=1,
#                        test_ratio=0.2,
#                        exclude_target_from_features=True):
#     """从Excel生成时间序列样本（修复特征泄漏问题）"""
#     # 读取并预处理数据
#     df = pd.read_excel(
#         file_path,
#         sheet_name=sheet_name,
#         parse_dates=[date_col],
#         date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S')
#     ).set_index(date_col).sort_index()
#
#     # 确保时间频率连续
#     df = df.asfreq(freq).fillna(0.0)
#
#     # 分离特征和目标列
#     feature_cols = [col for col in df.columns if col != target_col]
#     target_idx = df.columns.get_loc(target_col)
#
#     # 准备特征数据和目标数据
#     feature_data = df[feature_cols].values
#     target_data = df[[target_col]].values  # 保持2D形状
#
#     # 生成滑动窗口数据
#     X, y = [], []
#     for i in range(len(df) - window_size - forecast_steps + 1):
#         # 特征输入 (仅特征列)
#         X.append(feature_data[i:i + window_size, :])
#
#         # 目标输出 (仅目标列)
#         y.append(target_data[i + window_size:i + window_size + forecast_steps, 0])
#
#     # 转换为3D张量 [B, L, N]
#     X = np.array(X)  # (B, L, N)
#     y = np.array(y)  # (B, T)
#
#     # 划分数据集
#     split_idx = int(len(X) * (1 - test_ratio))
#
#     # 返回特征列名称（不包括目标列）
#     return (X[:split_idx], y[:split_idx],
#             X[split_idx:], y[split_idx:],
#             feature_cols)  # 只返回特征列名
def excel_to_sequences(file_path,
                       target_col='实际发电功率(mw)',#small_car_num_3
                       date_col='时间',#date_time
                       sheet_name='sheet1',
                       freq='15T',
                       window_size=24,
                       forecast_steps=1,
                       test_ratio=0.2):
    """从Excel生成时间序列样本"""
    # 读取并预处理数据
    df = pd.read_excel(
        file_path,
        sheet_name=sheet_name,
        parse_dates=[date_col],
        date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S')  #%m/%d/%y %H:%M, %Y-%m-%d %H:%M:%S  日期格式切换
    ).set_index(date_col).sort_index().fillna(0.0)#.asfreq(freq)

    # 分离特征和目标列
    feature_cols = [col for col in df.columns if col != target_col]
    target_idx = df.columns.get_loc(target_col)

    # 生成滑动窗口数据
    data = df.values
    X, y = [], []
    for i in range(len(data) - window_size - forecast_steps + 1):
        # 多维特征输入 (N features)
        X.append(data[i:i + window_size, :])
        # 单目标输出
        y.append(data[i + window_size:i + window_size + forecast_steps, target_idx])

    # 转换为3D张量 [B, N, L]
    X = np.transpose(np.array(X), (0, 2, 1))  # (B, N, L)
    y = np.array(y)  # (B, T)

    # 划分数据集
    split_idx = int(len(X) * (1 - test_ratio))
    return (X[:split_idx], y[:split_idx],
            X[split_idx:], y[split_idx:],
            df.columns.tolist())


class TimeSeriesDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


def create_excel_loaders(file_path, batch_size=32, **kwargs):
    """创建DataLoader的端到端流程"""
    X_train, y_train, X_test, y_test, cols = excel_to_sequences(file_path, **kwargs)

    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
        cols
    )


# 使用示例
if __name__ == "__main__":
    train_loader, test_loader, columns = create_excel_loaders(
        file_path="../dataset/2019.xlsx",
        #target_col='Power7(MW)',  # small_car_num_3
        date_col='时间',  # date_time
        sheet_name='sheet1',
        window_size=24,
        forecast_steps=3,
        batch_size=256
    )

    # 验证数据维度
    sample_X, sample_y = next(iter(train_loader))
    print(f"输入维度: {sample_X.shape} (batch, features, window)")
    print(f"标签维度: {sample_y.shape} (batch, pred_steps)")
    print(f"特征列表: {columns}")