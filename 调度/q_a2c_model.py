import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import numpy as np
from config import QuantumConfig
from typing import Tuple, Union
from sb3_contrib import RecurrentPPO  # 修复导入
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import TensorDict


class QuantumCritic(nn.Module):
    """
    量子Critic网络实现（基于文档2的Nav-Q架构）
    核心功能：使用量子电路计算状态价值函数
    """

    def __init__(self, n_qubits: int, n_layers: int, ansatz: int, quantum_device: object):
        """
        初始化量子Critic
        :param n_qubits: 量子比特数（默认4）
        :param n_layers: 量子层数（默认2）
        :param ansatz: 量子电路类型（1=硬件高效, 2=CRY门）
        :param quantum_device: PennyLane量子设备
        """
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.ansatz = ansatz
        self.device = quantum_device

        # 每个量子比特处理3个输入特征
        self.input_dim = 3 * n_qubits

        # 创建量子电路
        self.qnode = self._create_quantum_circuit()

        self.adapter = nn.Linear(64, 3 * n_qubits)  # 64→12 (当n_qubits=4时)

        # 量子参数初始化
        if ansatz == 1:
            self.q_weights = nn.Parameter(torch.randn(n_layers, 2 * n_qubits))
        else:
            self.q_weights = nn.Parameter(torch.randn(n_layers, n_qubits))

        # 经典输出层
        self.fc = nn.Linear(n_qubits, 1)

    def _create_quantum_circuit(self):
        """创建量子电路（基于文档2中的QIDEP策略）"""
        dev = self.device

        @qml.qnode(dev, interface="torch")
        def circuit(inputs: torch.Tensor, weights: torch.Tensor):
            """
            量子电路前向传播
            :param inputs: 输入特征 (3 * n_qubits)
            :param weights: 可训练量子参数
            """
            # 输入特征预处理（文档7的QIDEP策略）
            inputs = inputs.squeeze(0)

            # 数据重上传（Data Reuploading）
            for l in range(self.n_layers):
                # 编码层（每个量子比特3个特征）
                for q in range(self.n_qubits):
                    start_idx = q * 3
                    qml.RX(inputs[start_idx], wires=q)
                    qml.RY(inputs[start_idx + 1], wires=q)
                    qml.RZ(inputs[start_idx + 2], wires=q)

                # 变分层（根据ansatz类型）
                if self.ansatz == 1:
                    # 硬件高效ansatz（文档2 Fig.5）
                    for q in range(self.n_qubits):
                        qml.RY(weights[l, q], wires=q)
                        qml.RZ(weights[l, q + self.n_qubits], wires=q)

                    # 纠缠层
                    for q in range(self.n_qubits - 1):
                        qml.CZ(wires=[q, q + 1])

                elif self.ansatz == 2:
                    # CRY门ansatz（文档2 Section 4.3）
                    for q in range(self.n_qubits):
                        qml.CRY(weights[l, q], wires=[q, (q + 1) % self.n_qubits])

            # 测量（Pauli-Z期望值）
            return [qml.expval(qml.PauliZ(q)) for q in range(self.n_qubits)]

        return circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        :param x: 输入特征 (batch_size, latent_dim)
        :return: 状态价值估计
        """
        # 输入预处理（文档7的arctan归一化）
        x = self.adapter(x)  # [batch_size, 64] → [batch_size, 12]
        x = torch.atan(x)

        # 确保输入维度匹配
        if x.shape[1] > self.input_dim:
            padding = torch.zeros(x.shape[0], self.input_dim - x.shape[1])
            x = torch.cat((x, padding), dim=1)

        # 量子电路执行
        quantum_out = self.qnode(x, self.q_weights)

        # 转换为tensor
        if isinstance(quantum_out, list):
            quantum_out = torch.tensor(quantum_out, dtype=torch.float32)

        # 经典输出层
        return self.fc(quantum_out)


class Q_A2C(RecurrentPPO):
    def __init__(self, policy, env, policy_kwargs, **kwargs):
        quantum_kwargs = policy_kwargs.pop("quantum_kwargs", {})
        self.quantum_critic = QuantumCritic(
            n_qubits=quantum_kwargs["n_qubits"],
            n_layers=quantum_kwargs["n_layers"],
            ansatz=quantum_kwargs["ansatz"],
            quantum_device=quantum_kwargs["quantum_device"]
        )

        # 初始化父类（直接使用正确的policy_kwargs）
        super().__init__(policy, env, policy_kwargs=policy_kwargs,** kwargs)

    def evaluate_actions(self,
                         obs: torch.Tensor,
                         lstm_states: torch.Tensor,
                         mask: torch.Tensor,
                         actions: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        评估动作（覆盖父类方法）
        :param obs: 观察值
        :param lstm_states: LSTM状态
        :param mask: 动作掩码
        :param actions: 选择的动作
        :return: (价值估计, 对数概率, 熵, LSTM状态)
        """
        # 获取LSTM特征
        features, _ = self.policy.get_latent(obs, lstm_states, mask)

        # 量子Critic计算价值函数
        values = self.quantum_critic(features)

        # 策略分布
        distribution = self.policy.get_distribution(features)

        # 计算对数概率和熵
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        return values, log_prob, entropy, _

    def save(self, path: str):
        """保存模型（量子参数和经典参数）"""
        super().save(path)

        # 额外保存量子参数
        torch.save({
            "quantum_critic_state": self.quantum_critic.state_dict(),
            "q_weights": self.quantum_critic.q_weights
        }, path + "_quantum.pth")

    def load(self, path: str, env=None):
        """加载模型（量子参数和经典参数）"""
        super().load(path, env)

        # 加载量子参数
        quantum_data = torch.load(path + "_quantum.pth")
        self.quantum_critic.load_state_dict(quantum_data["quantum_critic_state"])
        self.quantum_critic.q_weights.data.copy_(quantum_data["q_weights"])