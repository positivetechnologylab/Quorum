import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.circuit.library import Initialize
from qiskit.quantum_info import Statevector

def prepare_for_embedding(data_point: np.ndarray) -> np.ndarray:
    """
    Prepare a single data point for amplitude embedding.
    
    Args:
    data_point (np.ndarray): Data point
    
    Returns:
    np.ndarray: Prepared state vector
    """
    probabilities = data_point ** 2
    
    sum_prob = np.sum(probabilities)
    
    # add a "trash state" probability for the remaining probability mass
    trash_state_prob = max(0, 1 - sum_prob)
    probabilities = np.append(probabilities, trash_state_prob)
    
    # normalize to ensure sum of probabilities is 1
    sum_prob = np.sum(probabilities)
    if sum_prob > 0:
        probabilities = probabilities / sum_prob
    
    amplitudes = np.sqrt(probabilities)
    
    return amplitudes

def create_amplitude_encoding_circuit(data_point: np.ndarray, num_qubits: int) -> QuantumCircuit:
    """
    Create an amplitude encoding circuit for a single data point.
    The circuit performs two identical amplitude encodings and leaves an ancilla qubit.
    
    Args:
    data_point (np.ndarray): Single data point
    num_qubits (int): Number of qubits for each encoding (total qubits will be 2*num_qubits + 1)
    
    Returns:
    QuantumCircuit: Amplitude encoding circuit
    """
    total_qubits = 2 * num_qubits + 1  # Two encodings plus one ancilla qubit
    qc = QuantumCircuit(total_qubits)
    
    prepared_state = prepare_for_embedding(data_point)
    state_vector = Statevector(prepared_state)
    init_gate = Initialize(state_vector)
    
    qc.append(init_gate, range(num_qubits))
    
    qc.append(init_gate, range(num_qubits, 2*num_qubits))
    
    return qc

def create_amplitude_encoding_circuits(data: pd.DataFrame, num_qubits: int) -> list:
    """
    Create a list of amplitude encoding circuits for all data points.
    
    Args:
    data (pd.DataFrame): Input data
    num_qubits (int): Number of qubits for each encoding
    
    Returns:
    list: List of amplitude encoding circuits
    """
    circuits = []
    for _, row in data.iterrows():
        circuit = create_amplitude_encoding_circuit(row.values, num_qubits)
        circuits.append(circuit)
    
    print(f"Created {len(circuits)} amplitude encoding circuits, each with {2*num_qubits + 1} qubits")
    return circuits