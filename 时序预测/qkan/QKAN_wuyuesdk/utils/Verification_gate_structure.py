from matplotlib import pyplot as plt
from qiskit import QuantumCircuit , Aer , transpile

from utils.BlockEncoder import SymbolicUnitaryGate


def find_custom_gates( circuit , custom_gate_types=None , custom_gate_names=None ,
                      draw_format='text' , draw_options=None):
    """
    查找电路中所有的自定义门实例，并以线路图形式展示门的结构

    参数:
        circuit: 要搜索的量子电路
        custom_gate_types: 自定义门类型的列表，默认为包含SymbolicUnitaryGate
        custom_gate_names: 自定义门名称的列表，默认为包含"sym_unitary"
        draw_format: 线路图绘制格式，可选'text'、'mpl'、'latex'等
        draw_options: 绘制选项字典，将传递给draw()方法
    """
    # 设置默认的自定义门类型和名称
    if custom_gate_types is None:
        custom_gate_types = [SymbolicUnitaryGate]
    if custom_gate_names is None:
        custom_gate_names = ["sym_unitary"]
    if draw_options is None:
        draw_options = {}

    target_gates = []
    gate_count = 0  # 用于计数找到的门，方便标识

    def _draw_gate_circuit(gate , gate_id):
        """绘制门的内部线路图"""
        if hasattr(gate , "definition") and isinstance(gate.definition , QuantumCircuit):
            print(f"\n----- 门结构线路图 (ID: {gate_id}) -----")
            # 使用Qiskit的draw方法绘制线路图
            print(gate.definition.draw(output=draw_format , **draw_options))

            #gate.definition.draw('mpl' , filename=f"needed_istru.png")
            if draw_format == 'mpl':
                plt.title(f"自定义门结构: {type(gate).__name__} (ID: {gate_id})")
                plt.show()
        else:
            print("\n该门没有内部定义的线路结构")

    def _print_gate_info(gate , gate_id):
        """打印门的基本信息"""
        gate_type = type(gate).__name__
        gate_name = getattr(gate , "name" , "unknown")

        print(f"\n===== 找到自定义门 (ID: {gate_id}) =====")
        print(f"类型: {gate_type}")
        print(f"名称: {gate_name}")

        # 打印门的参数（如果有）
        if hasattr(gate , "params") and gate.params:
            print(f"参数: {gate.params}")

        # 打印门的 qubits 数量
        if hasattr(gate , "num_qubits"):
            print(f"量子比特数: {gate.num_qubits}")

    def _recursive_search(current_circuit , parent_instr=None , depth=0):
        nonlocal gate_count
        depth_indent = "  " * depth

        for instr , qubits , clbits in current_circuit.data:
            # 1. 穿透封装层
            current_gate = instr
            for _ in range(10):  # 穿透封装层
                if hasattr(current_gate , "operation"):
                    current_gate = current_gate.operation
                elif hasattr(current_gate , "base_gate"):  # 处理受控门
                    current_gate = current_gate.base_gate
                else:
                    break

            # 2. 识别目标门
            is_target = (
                    any(isinstance(current_gate , gate_type) for gate_type in custom_gate_types) or
                    (hasattr(current_gate , "name") and current_gate.name in custom_gate_names)
            )

            if is_target:
                gate_count += 1
                _print_gate_info(current_gate , gate_count)
                _draw_gate_circuit(current_gate , gate_count)

                target_gates.append({
                    "gate": current_gate ,
                    "instr": instr ,
                    "qubits": qubits ,
                    "parent_circuit": current_circuit ,
                    "depth": depth ,
                    "id": gate_count
                })

            # 3. 递归搜索子电路
            if hasattr(instr , "definition") and isinstance(instr.definition , QuantumCircuit):
                try:
                    _recursive_search(instr.definition , parent_instr=instr , depth=depth + 1)
                except Exception as e:
                    print(f"{depth_indent}搜索子电路时出错: {e}")

            # 4. 搜索门定义中的子电路
            if hasattr(current_gate , "definition") and isinstance(current_gate.definition , QuantumCircuit):
                try:
                    _recursive_search(current_gate.definition , parent_instr=instr , depth=depth + 1)
                except Exception as e:
                    print(f"{depth_indent}搜索门定义时出错: {e}")

    # 开始搜索
    print("开始搜索自定义门...")
    _recursive_search(circuit)
    print(f"\n搜索完成，共找到 {gate_count} 个自定义门")

    return target_gates