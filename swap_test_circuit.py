from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

def create_swap_test_circuit(num_qubits):
    """
    Create a SWAP test circuit for 2*num_qubits + 1 qubits.
    
    Args:
    num_qubits (int): Number of qubits in each state to be compared.
    
    Returns:
    QuantumCircuit: The SWAP test circuit.
    """
    total_qubits = 2 * num_qubits + 1
    
    qr = QuantumRegister(total_qubits)
    cr = ClassicalRegister(1)
    
    qc = QuantumCircuit(qr, cr)
    qc.h(total_qubits - 1)
    
    for i in range(num_qubits):
        qc.cswap(total_qubits - 1, i, i + num_qubits)
    
    qc.h(total_qubits - 1)
    qc.measure(total_qubits - 1, 0)
    
    return qc