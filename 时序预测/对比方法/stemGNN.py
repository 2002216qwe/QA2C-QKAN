import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import time

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
    X_scaled = data_scaled[:len(X) * X.shape[1]].reshape(X.shape[0], X.shape[1])
    y_scaled = data_scaled[len(X) * X.shape[1]:].reshape(-1)

    # 划分训练集和测试集
    split_idx = int(len(X) * split_ratio)
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]

    # 转换为PyTorch张量
    X_train = torch.FloatTensor(X_train).unsqueeze(2)  # [样本数, 序列长度, 特征数]
    X_test = torch.FloatTensor(X_test).unsqueeze(2)
    y_train = torch.FloatTensor(y_train)
    y_test = torch.FloatTensor(y_test)

    return X_train, X_test, y_train, y_test, scaler


# 优化后的图注意力层
class GraphAttentionLayer(nn.Module):
    """高效图注意力层"""

    def __init__(self, in_features, out_features, dropout=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Linear(in_features, out_features)
        self.a = nn.Linear(2 * out_features, 1)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, h, adj):
        Wh = self.W(h)  # [N, out_features]

        # 使用矩阵运算替代循环，提高计算效率
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(self.a(a_input).squeeze(2))

        # 使用稀疏矩阵表示邻接矩阵（如果适用）
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = torch.softmax(attention, dim=1)
        attention = self.dropout(attention)
        h_prime = torch.matmul(attention, Wh)

        return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # 使用向量化操作替代循环
        N = Wh.size()[0]  # 节点数

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)

        return all_combinations_matrix.view(N, N, 2 * self.out_features)


# 优化后的StemGNN模型 - 内存优化版本
class StemGNN(nn.Module):
    def __init__(self, seq_length, hidden_dim=64, dropout=0.2):
        super(StemGNN, self).__init__()

        # 特征提取层 - 使用更高效的卷积结构
        self.feature_extraction = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),  # 添加BatchNorm加速收敛
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        # 动态计算num_heads，确保能整除seq_length
        max_heads = 8
        num_heads = 1
        for i in range(min(max_heads, seq_length), 0, -1):
            if seq_length % i == 0:
                num_heads = i
                break

        print(f"使用{num_heads}个注意力头，序列长度为{seq_length}")

        # 图构建 - 使用自注意力机制
        self.self_attention = nn.MultiheadAttention(embed_dim=seq_length, num_heads=num_heads, dropout=dropout)

        # 图神经网络层 - 减少层数以提高效率
        self.gat1 = GraphAttentionLayer(hidden_dim, hidden_dim, dropout)

        # 时间注意力层
        self.time_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softmax(dim=2)
        )

        # 全局平均池化
        self.global_pooling = nn.AdaptiveAvgPool1d(1)

        # 预测层 - 简化结构
        self.prediction = nn.Sequential(
            nn.Linear(hidden_dim, 32),  # 减少神经元数量
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: [batch_size, seq_length, 1]
        batch_size, seq_length, _ = x.size()

        # 特征提取
        x = x.transpose(1, 2)  # [batch_size, 1, seq_length]
        x = self.feature_extraction(x)  # [batch_size, hidden_dim, seq_length]
        x = self.dropout(x)

        # 图构建 - 使用自注意力机制生成邻接矩阵
        adj = self._build_adjacency_matrix(x)

        # 图神经网络 - 减少计算量
        batch_size, hidden_dim, seq_length = x.size()
        x = x.transpose(1, 2).reshape(batch_size * seq_length, hidden_dim)  # [batch_size*seq_length, hidden_dim]

        h = self.gat1(x, adj)
        h = self.dropout(h)

        # 重塑回时序格式
        h = h.view(batch_size, seq_length, hidden_dim)

        # 时间注意力
        h_time = h
        time_weights = self.time_attention(h_time)
        h = h * time_weights

        # 转置为 [batch_size, hidden_dim, seq_length]
        h = h.transpose(1, 2)

        # 使用全局平均池化简化维度
        h = self.global_pooling(h)  # [batch_size, hidden_dim, 1]
        h = h.view(batch_size, -1)  # [batch_size, hidden_dim]

        # 预测
        output = self.prediction(h)

        return output

    def _build_adjacency_matrix(self, x):
        # 优化邻接矩阵构建过程，减少内存占用
        batch_size, hidden_dim, seq_length = x.size()

        # 使用小批量处理构建邻接矩阵，避免一次性生成大矩阵
        max_batch_size = 100  # 可调整的小批量大小
        adj_batches = []

        x = x.transpose(1, 2)  # [batch_size, seq_length, hidden_dim]
        x = x.reshape(batch_size * seq_length, hidden_dim)

        # 分批处理
        for i in range(0, batch_size, max_batch_size):
            end_i = min(i + max_batch_size, batch_size)
            batch_x = x[i * seq_length:end_i * seq_length]

            # 简化的自注意力机制构建邻接矩阵
            attention = torch.matmul(batch_x, batch_x.transpose(0, 1))
            adj_batch = torch.softmax(attention, dim=1)
            adj_batches.append(adj_batch)

        # 合并邻接矩阵
        if len(adj_batches) > 1:
            adj = torch.cat([torch.cat(adj_row, dim=1) for adj_row in adj_batches], dim=0)
        else:
            adj = adj_batches[0]

        return adj


# 优化的训练函数 - 添加内存优化和进度跟踪
def train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs=50):
    model.train()
    history = {'train_loss': [], 'val_loss': []}
    total_start_time = time.time()

    # 检查是否有可用的GPU
    use_cuda = device.type == 'cuda'

    for epoch in range(epochs):
        epoch_start_time = time.time()
        train_loss = 0.0
        model.train()

        # 添加进度条跟踪
        total_batches = len(train_loader)
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            # 优化内存使用
            if use_cuda:
                # 使用non_blocking=True异步传输数据到GPU
                X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
            else:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()

            # 梯度裁剪，防止内存溢出
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)

            # 打印进度
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx + 1}/{total_batches}, Loss: {loss.item():.6f}")

            # 释放不再需要的张量
            del X_batch, y_batch, outputs, loss
            if use_cuda:
                torch.cuda.empty_cache()

        # 验证
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                if use_cuda:
                    X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
                else:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)

                # 释放不再需要的张量
                del X_batch, y_batch, outputs, loss
                if use_cuda:
                    torch.cuda.empty_cache()

        # 记录损失
        train_loss /= len(train_loader.dataset)
        val_loss /= len(test_loader.dataset)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        # 计算并打印时间
        epoch_time = time.time() - epoch_start_time
        print(
            f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Time: {epoch_time:.2f}s')

    total_time = time.time() - total_start_time
    print(f"训练完成！总耗时: {total_time:.2f}s")
    print(f"平均每个epoch耗时: {total_time / epochs:.2f}s")

    return history


# 预测函数
def predict(model, X_test, device):
    model.eval()
    with torch.no_grad():
        start_time = time.time()  # 在这里初始化start_time
        # 分批预测，避免内存溢出
        batch_size = 32
        y_pred = []
        for i in range(0, len(X_test), batch_size):
            batch = X_test[i:i + batch_size].to(device)
            outputs = model(batch).squeeze().cpu().numpy()
            y_pred.append(outputs)
            del batch, outputs
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        y_pred = np.concatenate(y_pred)

        end_time = time.time()
        print(f"预测耗时: {end_time - start_time:.2f}s")
    return y_pred
# 将预测结果导出到Excel
def export_to_excel(y_test_actual, y_pred_actual, file_path='StemGNN_results.xlsx'):
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

        # 设备配置 - 优先使用GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")

        # 创建数据加载器 - 优化内存使用
        batch_size = 16  # 减小batch_size以降低内存压力
        num_workers = 0 if device.type == 'cuda' else 2  # GPU训练时不使用额外进程
        pin_memory = False  # 关闭内存锁定以减少内存使用

        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=pin_memory)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                 num_workers=num_workers, pin_memory=pin_memory)

        # 构建模型
        model = StemGNN(seq_length).to(device)

        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 训练模型
        print("开始训练StemGNN模型...")
        history = train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs=50)

        # 预测
        y_pred = predict(model, X_test, device)

        # 反归一化
        y_test_actual = scaler.inverse_transform(y_test.numpy().reshape(-1, 1))
        y_pred_actual = scaler.inverse_transform(y_pred.reshape(-1, 1))

        # 计算RMSE
        rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
        print(f'测试集RMSE: {rmse:.4f}')
        export_to_excel(y_test_actual, y_pred_actual)
        # 绘制预测结果
        plt.figure(figsize=(12, 6))
        plt.plot(y_test_actual, label='实际值')
        plt.plot(y_pred_actual, label='预测值')
        plt.title('StemGNN时间序列预测')
        plt.xlabel('时间步')
        plt.ylabel('值')
        plt.legend()
        plt.grid(True)
        plt.savefig('StemGNN_prediction.png')
        plt.show()
        # 绘制训练历史
        plt.figure(figsize=(12, 6))
        plt.plot(history['train_loss'], label='训练损失')
        plt.plot(history['val_loss'], label='验证损失')
        plt.title('StemGNN模型训练历史')
        plt.xlabel('迭代次数')
        plt.ylabel('损失')
        plt.legend()
        plt.grid(True)
        plt.savefig('StemGNN_training_history.png')
        plt.show()
        print("====",history['train_loss'])
        print("=====",history['val_loss'])
if __name__ == "__main__":
    main()