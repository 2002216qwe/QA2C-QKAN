#电力系统调度优化算法

##项目概述

本项目提出一套基于量子计算的新型电力系统调度问题的解决方案，包含量子KAN网络、环形联邦学习模型、量子A2C算法等。

##环境依赖
qiskit==0.45.1

qiskit-aer==0.3.0

```
name: rl_vpp

channels:
  - defaults
  - conda-forge

dependencies:
  - python=3.10
  - pip=23.3.2
  - numpy=1.26
  - pandas=2.1
  - matplotlib=3.8
  - pip:
      - setuptools==65.5.0
      - torch==1.13.1
      - gym==0.21.0
      - stable-baselines3==1.8.0
      - sb3-contrib==1.8.0
      - pyyaml==6.0
      - wandb>=0.15,<0.17
      - plotly>=5.3,<6.3
      - python-dateutil>=2.8,<3

#可视化与数据处理

matplotlib==3.7.2

pandas==2.0.3

sympy==1.12.0
```

安装方法：

```
pip install -r requirements.txt

```

##代码结构

|文件名称                    |主要功能                    |                
|--------------------------|---------------------------|
| `data_utils.py`            |调度代码工作模块          |
| `Fed.py`                     |调度训练主入口              |
| `trainer.py`                 |调度代码绘图和训练       |
| `q_a2c_model.py`          |QA2C模型训练            |
| `Agent_trainer_notebooks`   |QA2C的对比实验    |
| `VPP_simulator.py`            |VPP仿真系统的入口   |
| `VPP_environment.py`       |基于 VPP环境类VPPEnv |
| `q_a2c_environment.py`    |适配QA2C的VPP环境    |
| `q_a2c_VPP_agent_trainer.py`  |QA2C调度训练入口 |
| `a2c_model.py`                  |A2C模型                 |              
| `Mul_QKAN_v3.py`             |QKAN模型         |
| `BlockEncoder.py`              |实现QKAN块编码    |
| `Verification_gate_structure.py`  |辅助参数优化   |
  
##核心算法说明

##1.电力系统仿真

*  实现代码：`VPP_simulator.py`、`VPP_environment.py`、`q_a2c_environment.py`

*  核心功能：`VPP_simulator.py`是VPP仿真系统的入口，`VPP_environment.py`实现基于 VPP环境类VPPEnv，`q_a2c_environment.py`适配QA2C的VPP环境

###2.QKAN预测模型

*  实现文件：`Mul_QKAN_v3.py` 、`BlockEncoder.py`、`Verification_gate_structure.py` 

*  核心功能：`Mul_QKAN_v3.py`实现光伏出力预测，`BlockEncoder.py`实现块编码操作，辅助`Mul_QKAN_v3.py`，`Verification_gate_structure.py`辅助参数优化

###3.环形联邦

*  实现文件：`data_utils.py`、`Fed.py`、`trainer.py` 

*  核心功能： `Fed.py`调度训练主入口，`data_utils.py`调度代码工作模块，`trainer.py` 调度代码绘图和训练

###4.量子A2C调度

*  实现代码：`q_a2c_model.py`、`q_a2c_environment.py`、`q_a2c_VPP_agent_trainer.py`

*  核心功能：`q_a2c_VPP_agent_trainer.py`是QA2C调度训练入口，`q_a2c_environment.py`适配QA2C的VPP环境，`q_a2c_model.py`QA2C模型训练  

## 注意事项


1.  量子算法部分需配置适当的模拟器或量子硬件访问权限

2.  注意环境依赖与兼容性问题

3.  迭代次数和采样数可根据精度需求调整（建议先从小规模测试）

4. 等等


***

通过量子技术的应用，本项目为新型电力系统调度问题提供了高效的解决方案，尤其适用于高动态性、多约束的现代电力系统优化场景。

