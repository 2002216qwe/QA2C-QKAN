import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from model.Mul_QKAN_V2 import QKANMultiTrainer
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

file_path = "./dataset/minidata.xlsx"
df = pd.read_excel(file_path)
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

def prepare_data(df, target_column):
    """准备时间序列预测数据"""
    if target_column not in df.columns:
        raise ValueError(f"列名 '{target_column}' 不存在于DataFrame中")

    # 分离目标列并创建时间索引
    df['时间'] = pd.to_datetime(df['时间'])
    df.set_index('时间', inplace=True)
    target_series = df[target_column].copy().sort_index()
    other_cols = df.drop(columns=[target_column]).copy().sort_index()

    # 检测并处理缺失值
    missing_mask = target_series.isnull()
    if missing_mask.any():
        print(f"警告: 目标列 '{target_column}' 包含 {missing_mask.sum()} 个缺失值，将使用插值处理")
        target_series = target_series.interpolate(method='time')

    return target_series, other_cols

def prepare_lstm_data(data, n_steps=24):
    """为LSTM模型准备数据"""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))

    X, y = [], []
    for i in range(n_steps, len(scaled_data)):
        X.append(scaled_data[i - n_steps:i])
        y.append(scaled_data[i])

    return np.array(X), np.array(y), scaler

# 主流程
window_size = 50
pre_step = 1

target_series, other_cols = prepare_data(df, "实际发电功率(mw)")

# 使用window_size作为时间窗口
X, Y, scaler = prepare_lstm_data(target_series, n_steps=window_size)
X = np.squeeze(X, axis=-1)
Y = np.squeeze(Y, axis=-1)

# 时间序列安全划分
test_ratio = 0.2
split_index = int(len(X) * (1 - test_ratio))

X_train, X_test = X[:split_index], X[split_index:]
Y_train, Y_test = Y[:split_index], Y[split_index:]

layer_sizes = [window_size,16,32,16, pre_step]
degrees = [3] * (len(layer_sizes) - 1)  # 每层的切比雪夫阶数
trainer = QKANMultiTrainer(
    layer_sizes=layer_sizes,
    degrees=degrees,
    maxiter=50,
    learning_rate=0.1,
    perturbation=0.00001
)

x = X_train.tolist()
y = Y_train.tolist()

# 执行训练
print("开始训练多层QKAN...")
optimal_params = trainer.train(x, y)
print(f"训练完成，最终损失: {trainer.loss_history[-1]:.4f}")

# 可视化训练过程
plt.figure(figsize=(10, 6))
plt.plot(trainer.loss_history)
plt.title('多层QKAN训练损失变化')
plt.xlabel('迭代次数')
plt.ylabel('均方误差')
plt.grid(True)
plt.savefig('qkan_multi_layer_training_loss.png')
print("损失曲线已保存至 qkan_multi_layer_training_loss.png")

# 预测并反归一化
output = trainer.predict(X_test)
Y_test_actual = scaler.inverse_transform(Y_test.reshape(-1, 1)).flatten()
output_array = np.array(output)

# 然后进行 reshape 操作
output_actual = scaler.inverse_transform(output_array.reshape(-1, 1)).flatten()

# 可视化预测结果（时序对比图）
plt.figure(figsize=(15, 6))
plt.plot(range(len(Y_test_actual)), Y_test_actual, label='真实值', color='blue')
plt.plot(range(len(output_actual)), output_actual, label='预测值', color='orange')
plt.title('预测值与真实值时序对比')
plt.xlabel('时间步')
plt.ylabel('发电功率 (MW)')
plt.legend()
plt.grid(True)
plt.savefig('qkan_multi_layer_forecast_comparison.png')
print("时序对比图已保存至 qkan_multi_layer_forecast_comparison.png")

# 计算评估指标（使用反归一化后的值）
mae = mean_absolute_error(Y_test_actual, output_actual)
rmse = np.sqrt(mean_squared_error(Y_test_actual, output_actual))

print(f"MAE: {mae:.4f} MW")
print(f"RMSE: {rmse:.4f} MW")