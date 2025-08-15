from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt
from qiskit import QuantumCircuit , QuantumRegister , transpile
from qiskit.circuit import Gate , Parameter , ParameterExpression , ParameterVector , Instruction , Barrier , Measure
import re

from qiskit.circuit.library import HGate , CXGate , ZGate , XGate , MCPhaseGate , RYGate , QFT , UGate , UnitaryGate
from qiskit.quantum_info import Operator
from collections import OrderedDict

from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Unroller
from scipy.linalg import sqrtm
from qiskit.transpiler.passes.synthesis import UnitarySynthesis
class SymbolicUnitaryGate(Gate):
    def __init__(self , U , label=None):
        self.num_qubits = int(np.log2(U.shape[0]))

        self._params = self._extract_parameters(U)
        super().__init__("sym_unitary" , self.num_qubits , self._params , label=label)
        self.U = U
        self._is_symbolic = True
        self._create_placeholder_definition()
    def _is_numeric_matrix(self , matrix):
        """检查矩阵是否由纯数值构成（不含符号参数）"""
        return not any(isinstance(x , ParameterExpression) for x in matrix.flatten())
    def _create_placeholder_definition(self):
        """创建占位定义：由恒等门组成的简单电路"""
        placeholder_circ = QuantumCircuit(self.num_qubits)
        # 每个量子比特上添加恒等门（无实际操作，仅作为占位）
        for q in range(self.num_qubits):
            placeholder_circ.id(q)
            placeholder_circ.x(q)
        self.definition = placeholder_circ  # 绑定占位定义

    def _extract_parameters(self , matrix):
        params = set()

        for elem in matrix.flat:
            if isinstance(elem , ParameterExpression):
                params |= elem.parameters
        return sorted(params , key=lambda p: p.name)

    def _define(self):
        from qiskit.quantum_info import Operator
        from qiskit import transpile

        try:
            operator = Operator(self.U)
            # instruction = operator.to_instruction()
            unitary_gate = UnitaryGate(operator)
            # 创建临时电路进行分解
            temp_circ = QuantumCircuit(self.num_qubits)
            temp_circ.append(unitary_gate , range(self.num_qubits))

            # 关键修改：将UnitaryGate分解为基础门（U3和CX）
            basis_gates = ['u' , 'cx']  # 目标基础门集
            decomposed_circ = transpile(
                temp_circ ,
                basis_gates=basis_gates ,
                optimization_level=2  # 启用优化减少门数
            )

            # 将分解后的电路设为门定义
            self.definition = decomposed_circ
            # print("自定义门构建并分解成功")
        except Exception as e:
            # 异常处理（例如非幺正矩阵）
            pass
            return


    def bind_parameters(self , parameters):
        """动态绑定参数到矩阵"""
        bound_matrix = np.zeros_like(self.U)
        for idx in np.ndindex(self.U.shape):
            elem = self.U[idx]
            if isinstance(elem , ParameterExpression):
                # 替换为绑定后的值
                bound_matrix[idx] = elem.bind(parameters)
            else:
                bound_matrix[idx] = elem
        return bound_matrix

    def inverse(self):
        U_dag = self.U.conj().T
        inv_gate = SymbolicUnitaryGate(U_dag , label=f"{self.label}†")
        inv_gate._params = self._params  # 继承参数集合
        return inv_gate


class CompliantBlockEncoder:
    def __init__(self ,A=None):
        self.A = None
        self.n = None
        self.dim = None
        self.is_parametric = None
        self.alpha_param = None
        self.spectral_norm = None
        self.alpha = None


    def set_matrix(self , A,param_name:str):
        """设置矩阵并初始化相关属性"""
        self.A = np.array(A , dtype=object)
        self.n = int(np.log2(len(A)))
        self.dim = 2 ** self.n

        # 检测参数化状态
        self.is_parametric = any(
            isinstance(x , ParameterExpression)
            for x in self.A.flat
        )
        self.alpha_param = Parameter(param_name) if self.is_parametric else None

        # 非参数矩阵直接计算谱范数
        if not self.is_parametric:
            self.spectral_norm = np.linalg.norm(A , 2)
            self.alpha = self.spectral_norm * 1.0001
        else:
            self.alpha = 1  # 占位值，绑定后更新

    def construct_block_encoding(self ,layer_idx=None,config=None, a=1):
        # 动态选择缩放方式
        scale = self.alpha_param if self.is_parametric else self.alpha
        scaled_A = self.A / scale  # 符号化操作
        # 计算共轭转置
        scaled_A_dag = scaled_A.conj().T  # (A/α)†

        # 计算公式中的中间矩阵
        term1 = scaled_A_dag @ scaled_A  # (A/α)†(A/α)
        term2 = scaled_A @ scaled_A_dag  # (A/α)(A/α)†
        # 构造块编码矩阵
        full_dim = self.dim * 2 ** a
        U = np.zeros((full_dim , full_dim) , dtype=object)

        # 块结构赋值（保留常数部分）
        U[:self.dim , :self.dim] = scaled_A
        U[:self.dim , self.dim:] = np.eye(self.dim)-term1 # 常数单位矩阵
        U[self.dim: , :self.dim] = np.eye(self.dim)-term2  # 常数单位矩阵
        U[self.dim: , self.dim:] = -scaled_A.conj().T

        # 创建量子电路
        # if layer_idx == config and config!=None:
        #     self.n = self.n+1
        qr_main = QuantumRegister(self.n , 'main')
        qr_aux = QuantumRegister(a , 'aux')
        self.circuit = QuantumCircuit(qr_aux , qr_main)
        # 添加参数化门（自动注册参数）
        self.param_gate = SymbolicUnitaryGate(U , label="BlockEnc")
        self.circuit.append(self.param_gate , [*qr_aux , *qr_main])
        # for gate , qubits , clbits in circuit.data:
        #     # 通过门的标识（如名称、参数等）确认是否为目标门
        #     if gate.name == self.param_gate.name:  # 假设name可唯一标识
        if layer_idx == config and config != None:
            self.n = self.n + 1
            qr_main = QuantumRegister(self.n , 'main')
            qr_aux = QuantumRegister(a , 'aux')
            self.circuit = QuantumCircuit(qr_aux , qr_main)
            self.circuit.append(self.param_gate,qr_main)
            return self.circuit , (qr_main , qr_aux)


        return self.circuit , (qr_main , qr_aux)

    def is_unitary(self,mat , tol=1e-5):
        # 计算共轭转置
        mat_dag = np.conj(mat.T)
        # 检查 U†·U 是否等于单位矩阵
        identity = np.eye(mat.shape[0])
        return np.allclose(mat_dag @ mat , identity , atol=tol) and np.allclose(mat @ mat_dag , identity , atol=tol)

    def bind_expression_matrix(self , matrix , full_param_dict):
        # 验证所有表达式所需参数
        self.validate_parameters(matrix , full_param_dict)

        bound_matrix = np.zeros_like(matrix , dtype=complex)
        dim = self.dim

        # 处理左上块（不需要sqrtm）
        for i in range(dim):
            for j in range(dim):
                bound_matrix[i , j] = self._evaluate_element(matrix[i , j] , full_param_dict)

        # 处理右下块（不需要sqrtm）
        for i in range(dim , matrix.shape[0]):
            for j in range(dim , matrix.shape[1]):
                bound_matrix[i , j] = self._evaluate_element(matrix[i , j] , full_param_dict)

        # 处理右上块（需要sqrtm）
        block_top_right = np.zeros((dim , matrix.shape[1] - dim) , dtype=object)
        for i in range(dim):
            for j in range(matrix.shape[1] - dim):
                elem = matrix[i , j + dim]
                block_top_right[i , j] = elem
                bound_matrix[i , j + dim] = self._evaluate_element(elem , full_param_dict)

        # 将绑定的右上块转换为数值矩阵进行sqrtm
        num_block_top_right = np.zeros_like(block_top_right , dtype=complex)
        for i in range(block_top_right.shape[0]):
            for j in range(block_top_right.shape[1]):
                num_block_top_right[i , j] = bound_matrix[i , j + dim]

        # 计算矩阵平方根（只对方块有效）
        if num_block_top_right.shape[0] == num_block_top_right.shape[1]:
            sqrt_top_right = sqrtm(num_block_top_right)
            bound_matrix[:dim , dim:] = sqrt_top_right

        # 处理左下块（需要sqrtm）
        block_bottom_left = np.zeros((matrix.shape[0] - dim , dim) , dtype=object)
        for i in range(matrix.shape[0] - dim):
            for j in range(dim):
                elem = matrix[i + dim , j]
                block_bottom_left[i , j] = elem
                bound_matrix[i + dim , j] = self._evaluate_element(elem , full_param_dict)

        # 将绑定的左下块转换为数值矩阵进行sqrtm
        num_block_bottom_left = np.zeros_like(block_bottom_left , dtype=complex)
        for i in range(block_bottom_left.shape[0]):
            for j in range(block_bottom_left.shape[1]):
                num_block_bottom_left[i , j] = bound_matrix[i + dim , j]

        # 计算矩阵平方根（只对方块有效）
        if num_block_bottom_left.shape[0] == num_block_bottom_left.shape[1]:
            sqrt_bottom_left = sqrtm(num_block_bottom_left)
            bound_matrix[dim: , :dim] = sqrt_bottom_left

        return bound_matrix

    def _evaluate_element(self,element , param_dict):
        # 非表达式类型直接返回
        if not isinstance(element , ParameterExpression):
            try:
                return complex(element)  # 确保转为复数类型
            except TypeError:
                return element

        # 存储当前表达式的所有依赖参数
        expr_params = element.parameters

        # 创建当前表达式专用的参数字典（仅包含必要的参数）
        expr_param_dict = {}

        for param in expr_params:
            if param in param_dict:
                param_value = param_dict[param]

                # 如果值本身是ParameterExpression，递归处理
                if isinstance(param_value , ParameterExpression):
                    expr_param_dict[param] = self._evaluate_element(param_value , param_dict)
                else:
                    # 数值类型直接使用
                    expr_param_dict[param] = param_value

        # 绑定表达式并计算数值

        bound_expr = element.bind(expr_param_dict)
        return bound_expr


    def validate_parameters(self,matrix , param_dict):
        required_params = set()
        for row in matrix:
            for elem in row:
                if isinstance(elem , ParameterExpression):
                    required_params |= elem.parameters
        missing = required_params - set(param_dict.keys())
        if missing:
            raise ValueError(f"缺少参数: {missing}")
    def update_alpha(self , bound_A,param_map,param_cir):
        """绑定参数后更新实际缩放因子"""
        self.spectral_norm = np.linalg.norm(bound_A , 2)
        self.alpha = self.spectral_norm * 1.0001
        dict0 = {self.alpha_param:self.alpha}
        all_dict = param_map | dict0
        bound_matrix = self.bind_expression_matrix(self.param_gate.U,all_dict)
        # print(bound_matrix)
        # print(f"检查是否为幺正矩阵：{self.is_unitary(bound_matrix)}")
        # print(self.param_gate.name)

        a = self.find_symbolic_gates(param_cir)
        self.update_symbolic_gate(a,bound_matrix)

        #self.param_gate.U = bound_matrix
        # self.param_gate._params = []
        # self.param_gate.definition = None
        #print("门当前参数：" , self.param_gate.params)

        return dict0



    def find_symbolic_gates(self,circuit):
        """
        查找电路中所有的SymbolicUnitaryGate实例，包括被封装和受控的情况
        """
        target_gates = []

        def _recursive_search(current_circuit , parent_instr=None):
            for instr , qubits , clbits in current_circuit.data:
                # 1. 尝试穿透可能的封装层，包括控制门的封装
                current_gate = instr
                # 增加穿透层数，确保能穿过control()创建的封装
                for _ in range(10):  # 增加到10层以应对更复杂的封装
                    if hasattr(current_gate , "operation"):
                        current_gate = current_gate.operation
                    elif hasattr(current_gate , "base_gate"):  # 处理受控门的情况
                        current_gate = current_gate.base_gate
                    else:
                        break

                # 2. 识别目标门：类型+名称双重判断
                is_target = (
                        isinstance(current_gate , SymbolicUnitaryGate) or  # 类型匹配
                        (hasattr(current_gate , "name") and current_gate.name == "sym_unitary")  # 名称匹配
                )

                if is_target:
                    target_gates.append({
                        "gate": current_gate ,
                        "instr": instr ,
                        "qubits": qubits ,
                        "parent_circuit": current_circuit
                    })

                # 3. 递归搜索子电路，包括控制门可能包含的子电路
                if hasattr(instr , "definition") and isinstance(instr.definition , QuantumCircuit):
                    try:
                        _recursive_search(instr.definition , parent_instr=instr)
                    except Exception as e:
                        print(f"Error searching in {instr}: {e}")

                # 4. 处理可能包含子电路的其他属性
                if hasattr(current_gate , "definition") and isinstance(current_gate.definition , QuantumCircuit):
                    try:
                        _recursive_search(current_gate.definition , parent_instr=instr)
                    except Exception as e:
                        print(f"Error searching in {current_gate}: {e}")

        # 从顶层电路开始搜索
        _recursive_search(circuit)
        return target_gates

    def update_symbolic_gate(self,gates , new_U):
        """
        更新SymbolicUnitaryGate的U矩阵，并重置定义，确保_define基于新U生成
        """
        for gate in gates:
            # 2. 更新门的U矩阵
            gate["gate"].U = new_U  # 直接修改U属性
            # 3. 重置旧定义（关键：迫使下次_define使用新U）
            gate["gate"].definition = None  # 清除缓存的旧定义







if __name__ == "__main__":
    # 测试代码
    from qiskit import Aer
    A = np.eye(4)  # 单位矩阵（简单测试）
    encoder = CompliantBlockEncoder()
    encoder.set_matrix(A)
    circ , _ = encoder.construct_block_encoding()

    # 检查电路是否仅含 u/cx 门
    print(circ.count_ops())  # 应输出：{'u': X, 'cx': Y} 或空字典（单位矩阵无操作）

    # 模拟真实编译流程
    from qiskit import transpile

    simulator = Aer.get_backend('statevector_simulator')  # 模拟 127-qubit 设备
    transpiled_circ = transpile(circ , simulator)  # 此处不应报错
    print(transpiled_circ)