import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, Multiply
from tensorflow.keras.layers import Conv1D, AveragePooling1D, Activation
from tensorflow.keras.layers import Lambda, Permute, Reshape, Flatten
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# 设置matplotlib支持中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


# 从Excel文件读取数据
def load_data_from_excel(file_path, column_name='实际发电功率(mw)'):
    """从Excel文件读取数据并返回时间序列"""
    try:
        df = pd.read_excel(file_path)
        # 确保数据按时间顺序排列
        if 'date' in df.columns:
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
    # 归一化
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


# Autoformer核心组件
class AutoCorrelation(tf.keras.layers.Layer):
    """自相关机制实现"""

    def __init__(self, d_model, num_heads, **kwargs):
        super(AutoCorrelation, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

    def build(self, input_shape):
        self.wq = Dense(self.d_model)
        self.wk = Dense(self.d_model)
        self.wv = Dense(self.d_model)
        self.dense = Dense(self.d_model)

    def call(self, q, k, v):
        batch_size = tf.shape(q)[0]

        # 线性变换
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        # 分割多头
        q = tf.reshape(q, [batch_size, -1, self.num_heads, self.depth])  # (batch_size, seq_len_q, num_heads, depth)
        q = tf.transpose(q, [0, 2, 1, 3])  # (batch_size, num_heads, seq_len_q, depth)

        k = tf.reshape(k, [batch_size, -1, self.num_heads, self.depth])  # (batch_size, seq_len_k, num_heads, depth)
        k = tf.transpose(k, [0, 2, 1, 3])  # (batch_size, num_heads, seq_len_k, depth)

        v = tf.reshape(v, [batch_size, -1, self.num_heads, self.depth])  # (batch_size, seq_len_v, num_heads, depth)
        v = tf.transpose(v, [0, 2, 1, 3])  # (batch_size, num_heads, seq_len_v, depth)

        # FFT计算自相关
        q_fft = tf.signal.rfft(q, fft_length=[tf.shape(q)[2] * 2])
        k_fft = tf.signal.rfft(k, fft_length=[tf.shape(k)[2] * 2])

        # 计算互相关（频域相乘）
        s = q_fft * tf.math.conj(k_fft)
        corr = tf.signal.irfft(s, fft_length=[tf.shape(s)[2] * 2])
        corr = corr[..., :tf.shape(q)[2]]

        # 计算注意力权重
        scale = 1.0 / tf.math.sqrt(tf.cast(self.depth, tf.float32))
        attn = tf.nn.softmax(scale * corr, axis=-1)

        # 应用注意力权重
        output = tf.matmul(attn, v)
        output = tf.transpose(output, [0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        output = tf.reshape(output, [batch_size, -1, self.d_model])  # (batch_size, seq_len_q, d_model)

        # 最终线性变换
        output = self.dense(output)
        return output


# 序列分解模块
class SeriesDecomposition(tf.keras.layers.Layer):
    """趋势和季节性分解模块"""

    def __init__(self, kernel_size=25, **kwargs):
        super(SeriesDecomposition, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2

    def build(self, input_shape):
        self.pool = AveragePooling1D(
            pool_size=self.kernel_size,
            strides=1,
            padding='same'
        )

    def call(self, x):
        # 使用移动平均提取趋势分量
        trend = self.pool(x)

        # 剩余部分为季节性分量
        seasonal = x - trend
        return seasonal, trend


# Autoformer编码器块
class AutoformerEncoderBlock(tf.keras.layers.Layer):
    """Autoformer编码器块"""

    def __init__(self, d_model, num_heads, kernel_size=25, dropout=0.1, **kwargs):
        super(AutoformerEncoderBlock, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.dropout_rate = dropout

        # 自相关机制
        self.auto_correlation = AutoCorrelation(d_model, num_heads)

        # 序列分解
        self.decomposition1 = SeriesDecomposition(kernel_size)
        self.decomposition2 = SeriesDecomposition(kernel_size)

        # 前馈网络
        self.dense1 = Dense(4 * d_model, activation='relu')
        self.dense2 = Dense(d_model)

        # 正则化和dropout
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.norm3 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

    def call(self, x):
        # 序列分解
        seasonal_init, trend_init = self.decomposition1(x)

        # 自相关机制
        attn_output = self.auto_correlation(seasonal_init, seasonal_init, seasonal_init)
        attn_output = self.dropout1(attn_output)
        out1 = self.norm1(attn_output + seasonal_init)

        # 前馈网络
        ffn_output = self.dense1(out1)
        ffn_output = self.dense2(ffn_output)
        ffn_output = self.dropout2(ffn_output)
        seasonal_out = self.norm2(out1 + ffn_output)

        # 残差连接和最终分解
        seasonal_out = seasonal_out + seasonal_init
        seasonal_final, trend_out = self.decomposition2(seasonal_out)

        # 整合趋势信息
        trend_out = trend_out + trend_init
        x_out = seasonal_final + trend_out

        return self.norm3(x_out)
def export_to_excel(y_test_actual, y_pred_actual, file_path='Autoformer_results.xlsx'):
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

# 构建Autoformer模型
def build_autoformer_model(input_shape, d_model=64, num_heads=4, num_blocks=2, dropout=0.1):
    """构建Autoformer模型"""
    inputs = Input(shape=input_shape)

    # 初始嵌入层
    x = Dense(d_model)(inputs)

    # Autoformer编码器块
    for _ in range(num_blocks):
        x = AutoformerEncoderBlock(d_model, num_heads, dropout=dropout)(x)

    # 最终输出层
    x = Flatten()(x)
    x = Dropout(dropout)(x)
    outputs = Dense(1)(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='mean_squared_error')

    return model


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
        model = build_autoformer_model(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            d_model=64,
            num_heads=4,
            num_blocks=2,
            dropout=0.1
        )

        # 模型结构摘要
        model.summary()

        # 训练模型
        history = model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=100,
            validation_data=(X_test, y_test),
            verbose=1
        )

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
        plt.title('Autoformer时间序列预测')
        plt.xlabel('时间步')
        plt.ylabel('值')
        plt.legend()
        plt.grid(True)
        plt.savefig('autoformer_prediction.png')
        plt.show()

        # 绘制训练历史
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'], label='训练损失')
        plt.plot(history.history['val_loss'], label='验证损失')
        plt.title('Autoformer模型训练历史')
        plt.xlabel('迭代次数')
        plt.ylabel('损失')
        plt.legend()
        plt.grid(True)
        plt.savefig('autoformer_training_history.png')
        plt.show()
        print("====",history.history['loss'])
        print("=====",history.history['val_loss'])



if __name__ == "__main__":
    main()