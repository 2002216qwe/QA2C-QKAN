import yaml
import torch
import numpy as np
from VPP_environment import VPPEnv, VPP_Scenario_config
from elvis.config import ScenarioConfig
import os
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from sb3_contrib import RecurrentPPO
from q_a2c_model import Q_A2C  # 量子A2C模型
import pennylane as qml
from config import QuantumConfig  # 量子配置参数
import matplotlib.pyplot as plt
from tqdm import tqdm
import contextlib
import warnings
from config import config
# 忽略特定警告
warnings.filterwarnings("ignore", category=UserWarning, module="pennylane")

# 检查CUDA设备
print("Torch-Cuda available device:", torch.cuda.is_available())

# 确保确定性行为
torch.backends.cudnn.deterministic = True
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
np.random.seed(0)


# ========== 量子初始化设置 ==========
def setup_quantum_device():
    """配置量子设备"""

    noise_model = None
    if QuantumConfig.depolarising_error:
        noise_model = qml.transforms.insert(qml.DepolarizingChannel,
                                            QuantumConfig.noise_level)
    if QuantumConfig.gate_control_noise:
        # 新增门控制噪声支持
        noise_model = qml.transforms.insert(qml.AmplitudeDamping,
                                            QuantumConfig.noise_level)

    if noise_model:
        return qml.device("default.mixed",
                          wires=QuantumConfig.n_qubits,
                          noise_model=noise_model)
    else:
        return qml.device("default.qubit",
                          wires=QuantumConfig.n_qubits,
                          shots=QuantumConfig.shots)


quantum_device = setup_quantum_device()


# ========== 量子模型创建函数 ==========
def create_quantum_model(env, config):
    # ===== 1. 量子参数配置 =====
    quantum_kwargs = {
        "n_layers": QuantumConfig.n_layers,
        "n_qubits": QuantumConfig.n_qubits,
        "ansatz": QuantumConfig.ansatz_type,
        "depolarising_error": QuantumConfig.depolarising_error,
        "gate_control_noise": QuantumConfig.gate_control_noise,
        "quantum_device": quantum_device
    }

    # ===== 2. 策略网络架构 =====
    policy_kwargs = {
        "enable_critic_lstm": True,
        "net_arch": [
            {"pi": config["net_arch"]["pi"], "vf": []}  # 值函数网络保持为空
        ],
        "quantum_kwargs": quantum_kwargs
    }

    # ===== 3. 显式传递参数列表 =====
    explicit_params = [
        "policy_type", "n_steps", "batch_size", "n_epochs",
        "gamma", "gae_lambda", "clip_range", "ent_coef",
        "vf_coef", "max_grad_norm", "learning_rate",
        "ortho_init", "activation_fn", "optimizer_class",
        "normalize_advantage", "net_arch"  # 添加net_arch到排除列表
    ]

    # ===== 4. 剩余参数过滤 =====
    # 关键修改：排除net_arch
    remaining_config = {
        k: v for k, v in config.items()
        if k not in explicit_params
    }

    # ===== 5. 量子模型创建 =====
    return Q_A2C(
        policy=config["policy_type"],
        env=env,
        learning_rate=config["learning_rate"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_range=config["clip_range"],
        ent_coef=config["ent_coef"],
        vf_coef=config["vf_coef"],
        max_grad_norm=config["max_grad_norm"],
        policy_kwargs=policy_kwargs,
        tensorboard_log="wandb/tensorboard_log/",
        verbose=1,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )  # 移除了 **remaining_config

    # ========== 主训练流程 ==========
if __name__ == "__main__":
    # 确保代码在直接运行时执行
    current_folder = ''
    VPP_data_input_path = current_folder + 'data/data_training/environment_table/' + 'Environment_data_2019.csv'
    elvis_input_folder = current_folder + 'data/config_builder/'

    # 加载场景配置
    case = 'wohnblock_household_simulation_adaptive_25.yaml'
    with open(elvis_input_folder + case, 'r') as file:
        yaml_str = yaml.full_load(file)

    elvis_config_file = ScenarioConfig.from_yaml(yaml_str)
    VPP_config_file = VPP_Scenario_config(yaml_str)

    # 环境初始化
    print("VPP_config_file keys:", VPP_config_file.keys())  # ✨ 验证键是否存在
    env = VPPEnv(VPP_data_input_path, elvis_config_file, VPP_config_file)
    X_env = Monitor(env)
    X_env = DummyVecEnv([lambda: X_env])

    # 模型配置
    config = {
        "policy_type": "MultiInputLstmPolicy",
        "n_steps": 8760,
        "batch_size": 8760,
        "n_epochs": 15,
        "total_timesteps": 8760,
        "learning_rate": 0.001033322426832622,
        "gamma": 0.9513482961981914,
        "gae_lambda": 0.92,
        "clip_range": 0.2,
        "ent_coef": 2.088889416246867e-9,
        "vf_coef": 0.1329116829939515,
        "max_grad_norm": 5,
        "net_arch": dict(pi=[128, 128], vf=[128, 128]),
        "ortho_init": True,
        "activation_fn": torch.nn.modules.activation.Tanh,
        "optimizer_class": torch.optim.Adam,
        "normalize_advantage": True,
    }

    # 添加量子配置
    config.update({
        "quantum_params": {
            "n_qubits": QuantumConfig.n_qubits,
            "n_layers": QuantumConfig.n_layers,
            "ansatz": QuantumConfig.ansatz_type,
            "depolarising_error": QuantumConfig.depolarising_error,
            "gate_control_noise": QuantumConfig.gate_control_noise,
            "shots": QuantumConfig.shots
        }
    })

    # 初始化Wandb
    os.environ['WANDB_NOTEBOOK_NAME'] = "Agent_trainer_notebooks/Quantum_RecurrentPPO_VPP_agent_trainer.ipynb"
    run = wandb.init(
        project="Quantum_RL_VPP",
        config=config,
        sync_tensorboard=True,
        monitor_gym=False,
        save_code=False
    )

    # 创建量子模型
    model = create_quantum_model(X_env, config)

    # 量子梯度特殊处理
    if QuantumConfig.use_parameter_shift:
        qml.enable_tape()

    # 训练过程
    print("Starting quantum-enhanced training...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.learn(
            total_timesteps=config["total_timesteps"],
            tb_log_name=f'QuantumPPO_{run.id}',
            callback=WandbCallback(gradient_save_freq=10000, verbose=0),
            progress_bar=True
        )

    # 量子电路可视化
    if QuantumConfig.visualize_circuit:
        print("Visualizing quantum circuit...")


        @qml.qnode(quantum_device)
        def circuit():
            qml.Hadamard(wires=0)
            for i in range(QuantumConfig.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(0))


        # 兼容所有版本的绘图方法
        try:
            # 尝试直接绘制电路
            circuit_text = qml.draw(circuit)()
            print("Quantum Circuit:\n", circuit_text)

            # 保存为文本文件并上传到 wandb
            with open("quantum_circuit.txt", "w") as f:
                f.write(circuit_text)
            wandb.log({"Quantum_Circuit": wandb.File("quantum_circuit.txt")})

            print("Saved quantum circuit as text")
        except Exception as e:
            print(f"Failed to visualize quantum circuit: {e}")
            wandb.log({"Quantum_Circuit_Error": str(e)})

    # 保存模型
    model.save(f"quantum_ppo_vpp_{run.id}")
    print(f"Model saved as quantum_ppo_vpp_{run.id}")

    # 评估模型
    print("Evaluating quantum model...")
    mean_reward, std_reward = evaluate_policy(model, X_env, n_eval_episodes=5)
    print(f"Quantum Model - Mean reward: {mean_reward}, Std: {std_reward}")

    # 对比经典模型
    if QuantumConfig.compare_classical:
        try:
            print("Loading classical model for comparison...")
            classical_model = RecurrentPPO.load("classical_model.zip", env=X_env)
            classical_mean, classical_std = evaluate_policy(classical_model, X_env, n_eval_episodes=5)
            print(f"Classical Model - Mean reward: {classical_mean}, Std: {classical_std}")
            wandb.log({
                "Quantum_vs_Classical": wandb.Table(
                    columns=["Model", "Mean Reward", "Std Reward"],
                    data=[
                        ["Quantum", mean_reward, std_reward],
                        ["Classical", classical_mean, classical_std]
                    ]
                )
            })
        except FileNotFoundError:
            print("Classical model not found, skipping comparison")
            wandb.log({"Quantum_vs_Classical": "Classical model not available"})

    # 结束Wandb
    run.finish()
    print("Training complete!")