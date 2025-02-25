import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

def create_ansatz(num_qubits, compression_level, param_prefix='θ'):
    """
    Create a parameterized quantum circuit for the autoencoder ansatz (encoder part).

    Args:
    num_qubits (int): Total number of qubits in the original space.
    compression_level (int): Number of qubits to compress to (1 <= compression_level <= num_qubits).
    param_prefix (str): Prefix for parameter names to avoid conflicts.

    Returns:
    QuantumCircuit: The parameterized ansatz circuit (encoder).
    ParameterVector: The parameters for the RY gates.
    """
    # Total number of qubits including the second set and ancilla
    total_qubits = num_qubits * 2 + 1

    qc = QuantumCircuit(total_qubits)

    num_parameters = sum(num_qubits - i for i in range(num_qubits - compression_level + 1))
    params = ParameterVector(param_prefix, num_parameters)

    param_index = 0
    for layer in range(num_qubits - compression_level + 1):
        for qubit in range(num_qubits - layer):
            qc.ry(params[param_index], qubit)
            param_index += 1

        if layer < num_qubits - compression_level:
            for qubit in range(num_qubits - layer - 1):
                qc.cx(qubit, qubit + 1)

    return qc, params

def create_reset_circuit(num_qubits, compression_level):
    """
    Create a circuit that resets qubits compression_level to num_qubits-1.

    Args:
    num_qubits (int): Total number of qubits in the original space.
    compression_level (int): Number of qubits to compress to.

    Returns:
    QuantumCircuit: Circuit with reset operations.
    """
    total_qubits = num_qubits * 2 + 1
    reset_circuit = QuantumCircuit(total_qubits)
    for qubit in range(compression_level, num_qubits):
        reset_circuit.reset(qubit)
    return reset_circuit

def create_encoder_decoder_circuit(num_qubits, compression_level, decoder_option):
    """
    Create a complete encoder-decoder circuit based on the specified option and compression level.

    Args:
    num_qubits (int): Total number of qubits in the original space.
    compression_level (int): Number of qubits to compress to (1 <= compression_level <= num_qubits).
    decoder_option (int): 1 for Qiskit's .inverse(), 2 for manual decoder.

    Returns:
    QuantumCircuit: The complete encoder-decoder circuit with parameterized gates.
    ParameterVector: The parameters for the encoder RY gates.
    ParameterVector: The parameters for the decoder RY gates (None for option 1).
    """
    if compression_level < 1 or compression_level > num_qubits:
        raise ValueError(f"Compression level must be between 1 and {num_qubits}")

    encoder, encoder_params = create_ansatz(num_qubits, compression_level, param_prefix='θ_enc')
    reset_circuit = create_reset_circuit(num_qubits, compression_level)

    if decoder_option == 1:
        decoder = encoder.inverse()
        complete_circuit = encoder.compose(reset_circuit).compose(decoder)
        return complete_circuit, encoder_params, None

    elif decoder_option == 2:
        decoder, decoder_params = create_ansatz(num_qubits, compression_level, param_prefix='θ_dec')
        decoder = decoder.reverse_ops()

        complete_circuit = encoder.compose(reset_circuit).compose(decoder)
        return complete_circuit, encoder_params, decoder_params

    else:
        raise ValueError("Invalid decoder option. Choose 1 for Qiskit's .inverse() or 2 for manual decoder.")

def update_circuit_parameters(circuit, encoder_params, decoder_params, new_angles):
    """
    Update the circuit with new angles for encoder and decoder.

    Args:
    circuit (QuantumCircuit): The parameterized encoder-decoder circuit.
    encoder_params (ParameterVector): The encoder parameters.
    decoder_params (ParameterVector): The decoder parameters (None for decoder option 1).
    new_angles (list): New angles for both encoder and decoder.

    Returns:
    QuantumCircuit: The updated circuit with new angles.
    """
    param_dict = {}
    
    num_encoder_params = len(encoder_params)
    param_dict.update(dict(zip(encoder_params, new_angles[:num_encoder_params])))
    
    if decoder_params is not None:
        param_dict.update(dict(zip(decoder_params, new_angles[num_encoder_params:])))
    
    bound_circuit = circuit.assign_parameters(param_dict)
    
    return bound_circuit