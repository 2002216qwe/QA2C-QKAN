import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import math

# 设置随机种子确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


# 定义位置编码类
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # 将位置编码添加到输入中
        return self.pe[:, :x.size(1)]


# 定义Fourier Enhanced Attention模块
class FourierEnhancedAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(FourierEnhancedAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # 线性投影
        q = self.q_linear(q).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

        # 计算注意力得分
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 应用softmax获取注意力权重
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # 应用注意力权重
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 输出投影
        output = self.out_proj(context)
        return output, attn


# 定义FEDformer模型
class FEDformer(nn.Module):
    def __init__(self, input_dim, output_dim, seq_len, pred_len, d_model=512, n_heads=8, n_layers=3, dropout=0.1):
        super(FEDformer, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.input_dim = input_dim
        self.output_dim = output_dim

        # 输入嵌入
        self.value_embedding = nn.Linear(input_dim, d_model)
        self.position_embedding = PositionalEmbedding(d_model)

        # 编码器层
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=4 * d_model,
                dropout=dropout,
                activation='gelu'
            )
            for _ in range(n_layers)
        ])

        # Fourier增强注意力层
        self.fourier_attn = FourierEnhancedAttention(d_model, n_heads, dropout)

        # 解码器层
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=4 * d_model,
                dropout=dropout,
                activation='gelu'
            )
            for _ in range(n_layers)
        ])

        # 输出层
        self.projection = nn.Linear(d_model, output_dim)

    def forward(self, x_enc):
        # 嵌入和位置编码
        x = self.value_embedding(x_enc)
        x = x + self.position_embedding(x)

        # 编码器
        for layer in self.encoder_layers:
            x = layer(x)

        # Fourier增强注意力
        attn_output, _ = self.fourier_attn(x, x, x)

        # 解码器（这里简化处理）
        dec_output = attn_output

        # 预测
        output = self.projection(dec_output)

        # 只返回预测长度的部分
        return output[:, -self.pred_len:, :]


# 自定义数据集类
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# 从Excel加载数据
def load_data_from_excel(file_path):
    try:
        # 读取Excel文件
        df = pd.read_excel(file_path)
        print(f"数据加载成功，形状: {df.shape}")
        return df
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None


# 创建序列
def create_sequences(data, seq_length, pred_length=1):
    # 假设最后一列为目标变量
    if isinstance(data, pd.DataFrame):
        data = data.values

    X, y = [], []
    for i in range(len(data) - seq_length - pred_length + 1):
        X.append(data[i:i + seq_length, :])  # 所有特征
        y.append(data[i + seq_length:i + seq_length + pred_length, -1:])  # 只取目标变量（最后一列）

    return np.array(X), np.array(y)


# 数据预处理（优化归一化部分）
def preprocess_data(X, y, test_size=0.2):
    # X形状: (n_samples, seq_len, n_features)
    # y形状: (n_samples, pred_len, 1)  假设目标变量是1维的

    n_samples, seq_len, n_features = X.shape
    pred_len = y.shape[1]

    # 1. 将X和y展平，合并为完整数据集用于拟合scaler
    # X展平: (n_samples * seq_len, n_features)
    X_flat = X.reshape(-1, n_features)
    # y展平: (n_samples * pred_len, 1)，需要扩展到n_features维度以便合并
    y_flat = np.zeros((y.size, n_features))
    y_flat[:, -1] = y.reshape(-1)  # 目标变量放在最后一列

    # 合并所有数据
    all_data = np.vstack([X_flat, y_flat])

    # 2. 拟合scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    all_data_scaled = scaler.fit_transform(all_data)

    # 3. 拆分回X和y的缩放版本
    # X_scaled: (n_samples, seq_len, n_features)
    X_scaled = all_data_scaled[:X_flat.shape[0]].reshape(n_samples, seq_len, n_features)
    # y_scaled: (n_samples, pred_len, 1)  只取目标变量列（最后一列）
    y_scaled = all_data_scaled[X_flat.shape[0]:, -1].reshape(n_samples, pred_len, 1)

    # 4. 划分训练集和测试集
    split_idx = int(n_samples * (1 - test_size))
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]

    # 5. 转换为PyTorch张量
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.FloatTensor(y_train)
    y_test = torch.FloatTensor(y_test)

    return X_train, X_test, y_train, y_test, scaler


# 训练模型
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device):
    model.to(device)
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)
        history['train_loss'].append(train_loss)

        # 验证阶段
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                val_loss += loss.item() * X_batch.size(0)

        val_loss /= len(val_loader.dataset)
        history['val_loss'].append(val_loss)

        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_fedformer_model.pth')

    return history


# 评估模型（优化反归一化部分）
def evaluate_model(model, test_loader, scaler, device):
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)

            # 收集预测值和真实值（缩放后的）
            y_true.append(y_batch.cpu().numpy())
            y_pred.append(outputs.cpu().numpy())

    # 合并所有批次
    y_true = np.vstack(y_true)  # 形状: (n_test_samples, pred_len, 1)
    y_pred = np.vstack(y_pred)  # 形状: (n_test_samples, pred_len, 1)

    # 1. 反归一化
    n_test_samples, pred_len, _ = y_true.shape
    n_features = scaler.n_features_in_

    # 构建用于反归一化的数组（只关注最后一列目标变量）
    # 真实值反归一化准备
    y_true_flat = y_true.reshape(-1)  # (n_test_samples * pred_len,)
    y_true_scaled = np.zeros((len(y_true_flat), n_features))
    y_true_scaled[:, -1] = y_true_flat  # 目标变量放在最后一列

    # 预测值反归一化准备
    y_pred_flat = y_pred.reshape(-1)
    y_pred_scaled = np.zeros((len(y_pred_flat), n_features))
    y_pred_scaled[:, -1] = y_pred_flat

    # 执行反归一化
    y_true_actual = scaler.inverse_transform(y_true_scaled)[:, -1].reshape(n_test_samples, pred_len, 1)
    y_pred_actual = scaler.inverse_transform(y_pred_scaled)[:, -1].reshape(n_test_samples, pred_len, 1)

    # 2. 计算RMSE
    rmse = np.sqrt(mean_squared_error(
        y_true_actual.reshape(-1),
        y_pred_actual.reshape(-1)
    ))
    print(f'测试集RMSE: {rmse:.6f}')

    return y_true_actual, y_pred_actual, rmse


def export_to_excel(y_test_actual, y_pred_actual, file_path='FEDfomer_results.xlsx'):
    """将实际值和预测值导出到Excel文件"""
    try:
        # 展平数据
        y_test_flat = y_test_actual.flatten()
        y_pred_flat = y_pred_actual.flatten()

        # 创建DataFrame
        results_df = pd.DataFrame({
            '实际值': y_test_flat,
            '预测值': y_pred_flat
        })

        # 添加误差列
        results_df['误差'] = results_df['实际值'] - results_df['预测值']
        results_df['绝对误差'] = np.abs(results_df['误差'])
        # 避免除以零
        results_df['相对误差(%)'] = np.where(
            results_df['实际值'] != 0,
            (results_df['绝对误差'] / results_df['实际值'] * 100),
            0
        )

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
            results_df.to_excel(writer, sheet_name='预测结果', index=False)
            statistics.to_excel(writer, sheet_name='统计信息', index=False)

            # 调整列宽
            prediction_sheet = writer.sheets['预测结果']
            stats_sheet = writer.sheets['统计信息']

            for i, col in enumerate(results_df.columns):
                prediction_sheet.column_dimensions[chr(65 + i)].width = max(10, len(col) + 2)

            for i, col in enumerate(statistics.columns):
                stats_sheet.column_dimensions[chr(65 + i)].width = max(15, len(col) + 2)

        print(f"预测结果已成功导出到 {file_path}")
        return True
    except Exception as e:
        print(f"导出Excel文件时出错: {e}")
        return False


# 主函数
def main():
    # 文件路径（建议改为相对路径）
    file_path = r'C:\Users\shan\Desktop\lstm\lstm\minidata.xlsx'

    # 序列长度和预测长度
    seq_length = 50
    pred_length = 1  # 预测未来1个时间步

    # 从Excel加载数据
    data = load_data_from_excel(file_path)

    if data is not None:
        # 创建序列
        X, y = create_sequences(data, seq_length, pred_length)
        print(f"序列创建完成: X形状 {X.shape}, y形状 {y.shape}")

        # 数据预处理（包含优化的归一化）
        X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y, test_size=0.2)
        print(f"预处理完成: 训练集X {X_train.shape}, 测试集X {X_test.shape}")

        # 创建数据加载器
        batch_size = 32
        train_dataset = TimeSeriesDataset(X_train, y_train)
        test_dataset = TimeSeriesDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        # 构建模型
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")

        input_dim = X_train.shape[2]
        output_dim = y_train.shape[2]

        model = FEDformer(
            input_dim=input_dim,
            output_dim=output_dim,
            seq_len=seq_length,
            pred_len=pred_length,
            d_model=128,
            n_heads=8,
            n_layers=2,
            dropout=0.1
        )

        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 训练模型
        epochs = 50
        print(f"开始训练，共{epochs}轮...")
        history = train_model(model, train_loader, test_loader, criterion, optimizer, epochs, device)

        # 加载最佳模型
        model.load_state_dict(torch.load('best_fedformer_model.pth'))

        # 评估模型
        y_true, y_pred, rmse = evaluate_model(model, test_loader, scaler, device)

        # 导出结果到Excel
        export_to_excel(y_true, y_pred)

        # 绘制预测结果
        plt.figure(figsize=(12, 6))
        plt.plot(y_true.reshape(-1), label='实际值', alpha=0.7)
        plt.plot(y_pred.reshape(-1), label='预测值', alpha=0.7)
        plt.title('FEDformer时间序列预测')
        plt.xlabel('时间步')
        plt.ylabel('值')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig('FEDformer_prediction.png', dpi=300)
        plt.show()

        # 绘制训练历史
        plt.figure(figsize=(12, 6))
        plt.plot(history['train_loss'], label='训练损失', alpha=0.7)
        plt.plot(history['val_loss'], label='验证损失', alpha=0.7)
        plt.title('FEDformer模型训练历史')
        plt.xlabel('迭代次数')
        plt.ylabel('MSE损失')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig('FEDformer_training_history.png', dpi=300)
        plt.show()
        print("====",history['train_loss'])
        print("=====",history['val_loss'])

if __name__ == "__main__":
    main()