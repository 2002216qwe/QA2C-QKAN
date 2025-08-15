import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

# 设置matplotlib支持中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


# 从Excel文件读取数据
def load_data_from_excel(file_path, column_name='实际发电功率(mw)'):
    """从Excel文件读取数据并返回时间序列"""
    try:
        df = pd.read_excel(file_path)
        # 确保数据按时间顺序排列
        if 'date' in df.columns: =
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')

        # 提取目标列数据
        if column_name in df.columns:
            data = df[column_name].values
            print(f"成功加载数据，共{len(data)}个数据点")
            return data
        else:
            print(f"错误: 数据中不存在列 '{column_name}'")
            return None
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None


# 准备时间序列数据
def create_sequences(data, seq_length):
    """将时间序列数据转换为监督学习格式"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


# 数据预处理
def preprocess_data(X, y, split_ratio=0.8):
    """数据预处理和划分训练测试集"""
    # 归一化C:\Users\shan\Desktop\lstm\lstm\Autofomer.py
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = np.concatenate([X.reshape(-1), y])
    data_scaled = scaler.fit_transform(data.reshape(-1, 1))

    # 重塑X和y
    X_scaled = data_scaled[:len(X) * X.shape[1]].reshape(X.shape[0], X.shape[1], 1)
    y_scaled = data_scaled[len(X) * X.shape[1]:]

    # 划分训练集和测试集
    split_idx = int(len(X) * split_ratio)
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]

    return X_train, X_test, y_train, y_test, scaler


# 构建LSTM模型
def build_lstm_model(input_shape, dropout=0.5):
    """构建LSTM模型"""
    model = Sequential()
    model.add(Input(shape=input_shape))  # 添加Input层
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
# 将预测结果导出到Excel
def export_to_excel(y_test_actual, y_pred_actual, file_path='LSTM_prediction_results.xlsx'):
    """将实际值和预测值导出到Excel文件"""
    try:
        # 创建DataFrame
        results_df = pd.DataFrame({
            '实际值': y_test_actual.flatten(),
            '预测值': y_pred_actual.flatten()
        })

        # 添加误差列
        results_df['误差'] = results_df['实际值'] - results_df['预测值']
        results_df['绝对误差'] = np.abs(results_df['误差'])
        results_df['相对误差(%)'] = (results_df['绝对误差'] / results_df['实际值'] * 100).fillna(0)

        # 计算统计信息
        statistics = pd.DataFrame({
            '统计量': ['平均实际值', '平均预测值', '均方根误差', '平均绝对误差', '平均相对误差(%)'],
            '值': [
                results_df['实际值'].mean(),
                results_df['预测值'].mean(),
                np.sqrt(mean_squared_error(results_df['实际值'], results_df['预测值'])),
                results_df['绝对误差'].mean(),
                results_df['相对误差(%)'].mean()
            ]
        })

        # 写入Excel文件
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            # 写入预测结果
            results_df.to_excel(writer, sheet_name='预测结果', index=False)

            # 写入统计信息
            statistics.to_excel(writer, sheet_name='统计信息', index=False)

            # 获取工作簿和工作表对象以进行格式设置
            workbook = writer.book
            prediction_sheet = writer.sheets['预测结果']
            stats_sheet = writer.sheets['统计信息']

            # 为预测结果表设置列宽
            for i, col in enumerate(results_df.columns):
                column_width = max(len(str(x)) for x in results_df[col])
                column_width = max(column_width, len(col)) + 2
                column_letter = chr(65 + i)  # A, B, C, ...
                prediction_sheet.column_dimensions[column_letter].width = column_width

            # 为统计信息表设置列宽
            for i, col in enumerate(statistics.columns):
                column_width = max(len(str(x)) for x in statistics[col])
                column_width = max(column_width, len(col)) + 2
                column_letter = chr(65 + i)
                stats_sheet.column_dimensions[column_letter].width = column_width

        print(f"预测结果已成功导出到 {file_path}")
        return True
    except Exception as e:
        print(f"导出Excel文件时出错: {e}")
        return False

# 主函数
def main():
    # 文件路径
    file_path = r'C:\Users\shan\Desktop\lstm\lstm\minidata.xlsx'

    # 序列长度
    seq_length = 50

    # 从Excel加载数据
    data = load_data_from_excel(file_path)

    if data is not None:
        # 创建序列
        X, y = create_sequences(data, seq_length)

        # 数据预处理
        X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)

        # 构建模型
        model = build_lstm_model((X_train.shape[1], X_train.shape[2]))

        # 训练模型
        history = model.fit(X_train, y_train, batch_size=32, epochs=50,
                            validation_data=(X_test, y_test))

        # 预测
        y_pred = model.predict(X_test)

        # 反归一化
        y_test_actual = scaler.inverse_transform(y_test)
        y_pred_actual = scaler.inverse_transform(y_pred)

        # 计算RMSE
        rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
        print(f'测试集RMSE: {rmse:.4f}')
        export_to_excel(y_test_actual, y_pred_actual)
        # 绘制预测结果
        plt.figure(figsize=(12, 6))
        plt.plot(y_test_actual, label='实际值')
        plt.plot(y_pred_actual, label='预测值')
        plt.title('LSTM时间序列预测')
        plt.xlabel('时间步')
        plt.ylabel('值')
        plt.legend()
        plt.grid(True)
        plt.savefig('LSTM_prediction.png')
        plt.show()
        # 绘制训练历史
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'], label='训练损失')
        plt.plot(history.history['val_loss'], label='验证损失')
        plt.title('LSTM模型训练历史')
        plt.xlabel('迭代次数')
        plt.ylabel('损失')
        plt.legend()
        plt.grid(True)
        plt.savefig('LSTM_training_history.png')
        plt.show()
        print("====",history.history['loss'])
        print("=====",history.history['val_loss'])
if __name__ == "__main__":
    main()