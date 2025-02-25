import argparse
import numpy as np
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
import threading
from Preprocessing.goldstein_uchida_preprocess import preprocess_goldstein_uchida
from Preprocessing.ccpp_preprocess import preprocess_ccpp
from data_bucketing import perform_bucketing
from feature_selection import select_features
from Embedding.range_amplitude_enc import create_amplitude_encoding_circuit
# from Ansatzes.ry_cx_ansatz import create_encoder_decoder_circuit, update_circuit_parameters
from Ansatzes.rx_rz_ansatz import create_encoder_decoder_circuit, update_circuit_parameters
# from Ansatzes.ry_rz_ansatz import create_encoder_decoder_circuit, update_circuit_parameters
from swap_test_circuit import create_swap_test_circuit
from qiskit_aer import AerSimulator

from qiskit_aer.noise import (NoiseModel, QuantumError, ReadoutError,
                             pauli_error, depolarizing_error, thermal_relaxation_error)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Data Preprocessing and Feature Selection")
    parser.add_argument("num_qubits", type=int, help="Number of qubits to use")
    parser.add_argument("decoder_option", type=int, choices=[1, 2], help="Decoder option: 1 for Qiskit's .inverse(), 2 for manual decoder")
    parser.add_argument("--num_threads", type=int, default=4, help="Number of threads to use")
    return parser.parse_args()


def create_realistic_noise_model(num_qubits):
    """
    Create a noise model matching IBM's Brisbane quantum computer specifications.
    
    Args:
    num_qubits (int): Number of qubits in the system
    
    Returns:
    NoiseModel: A Qiskit noise model matching Brisbane's error rates
    """
    noise_model = NoiseModel()
    
    # Brisbane specifications
    T1 = 230.42e3  # 230.42  in nanoseconds
    T2 = 143.41e3  # 143.41 in nanoseconds
    
    time_1q = 60 
    time_2q = 660
    time_readout = 1300
    
    # Error rates
    p_sx = 2.274e-4  
    p_cx = 2.903e-3  
    p_readout = 1.38e-2  
    
    # Add single-qubit gate errors
    for qubit in range(num_qubits):
        thermal_error_1q = thermal_relaxation_error(
            T1, T2, time_1q)
        
        depol_error_1q = depolarizing_error(p_sx, 1)
        
        gate_error_1q = thermal_error_1q.compose(depol_error_1q)
        
        noise_model.add_quantum_error(gate_error_1q, ["sx"], [qubit])
        
        readout_error = ReadoutError([[1 - p_readout, p_readout], 
                                    [p_readout, 1 - p_readout]])
        noise_model.add_readout_error(readout_error, [qubit])
        
        meas_thermal_error = thermal_relaxation_error(
            T1, T2, time_readout)
        noise_model.add_quantum_error(meas_thermal_error, ["measure"], [qubit])
    
    # Add two-qubit gate errors
    for q1 in range(num_qubits-1):
        for q2 in range(q1+1, num_qubits):
            thermal_error_q1 = thermal_relaxation_error(
                T1, T2, time_2q)
            thermal_error_q2 = thermal_relaxation_error(
                T1, T2, time_2q)
            thermal_error_2q = thermal_error_q1.expand(thermal_error_q2)
            
            depol_error_2q = depolarizing_error(p_cx, 2)
            
            gate_error_2q = thermal_error_2q.compose(depol_error_2q)
            
            noise_model.add_quantum_error(gate_error_2q, ["cx"], [q1, q2])
    
    return noise_model

#mimics Brisbane noise model
def configure_noisy_simulator(num_qubits):
    """
    Configure the AerSimulator with IBM Brisbane noise settings.
    
    Args:
    num_qubits (int): Number of qubits in the system
    
    Returns:
    AerSimulator: Configured noisy simulator matching Brisbane specifications
    """
    noise_model = create_realistic_noise_model(num_qubits)
    
    basis_gates = ['sx', 'rz', 'cx', 'measure']  # Brisbane's basis gates
    simulator = AerSimulator(
        noise_model=noise_model,
        basis_gates=basis_gates,
        coupling_map=[[i, i+1] for i in range(num_qubits-1)] 
    )
    
    return simulator

def process_iteration(iteration, num_qubits, decoder_option, preprocessed_data, high_risk_indices, swap_test, simulator, target_proportion, anomaly_likelihood_per_bucket, num_iterations, num_bucketruns):
    """
    Process a single iteration of the quantum autoencoder optimization.

    Args:
    iteration (int): The current iteration number.
    num_qubits (int): Number of qubits for a single amplitude encoding instance.
    decoder_option (int): Option for decoder circuit (1 or 2).
    preprocessed_data (pd.DataFrame): The preprocessed input data.
    high_risk_indices (list): Indices of high-risk data points.
    swap_test (QuantumCircuit): The swap test circuit.
    simulator (AerSimulator): The quantum circuit simulator.
    target_proportion (float): The target proportion for optimization.
    anomaly_likelihood_per_bucket (float): The anomaly likelihood per bucket.
    num_iterations (int): Total number of iterations.
    num_bucketruns (int): Number of random angle runs per bucket.

    Returns:
    dict: Results of the iteration, including buckets, selected features, and optimization results.
    """
    # Calculate the compression level based on the iteration number
    compression_levels = num_qubits - 1
    iterations_per_level = num_iterations // compression_levels
    compression_level = (iteration // iterations_per_level) + 1
    compression_level = min(compression_level, num_qubits - 1)

    print(f"\nStarting iteration {iteration + 1} with compression_level {compression_level}")

    # Run the preprocessed data through the bucketing algorithm
    target_probability = anomaly_likelihood_per_bucket
    buckets, bucket_size = perform_bucketing(preprocessed_data, high_risk_indices, target_probability)
    
    print(f"Number of buckets created: {len(buckets)}")
    print(f"Bucket size: {bucket_size}")

    # Run feature selection on the data to select features for amplitude encoding
    selected_data, selected_features = select_features(preprocessed_data, num_qubits, strategy='e')
    
    print(f"Number of features selected: {len(selected_features)}")
    print("Selected features:", selected_features)

    # Create amplitude encoding circuits for each datapoint+feature set
    amplitude_encoding_circuits = {}
    for idx, row in selected_data.iterrows():
        selected_row = row.values
        circuit = create_amplitude_encoding_circuit(selected_row, num_qubits)
        amplitude_encoding_circuits[idx] = circuit

    print(f"Created amplitude encoding circuits for {len(amplitude_encoding_circuits)} datapoints")

    # Create the "encoder-decoder" ansatz
    ansatz, encoder_params, decoder_params = create_encoder_decoder_circuit(num_qubits, compression_level, decoder_option)

    # Run random angle iterations for each bucket
    iteration_results = []
    for bucket_idx, bucket in enumerate(buckets):
        # print(f"\nProcessing bucket {bucket_idx + 1}/{len(buckets)}")
        bucket_circuits = [amplitude_encoding_circuits[idx] for idx in bucket]
        
        final_results = []
        for _ in range(num_bucketruns):
            if decoder_option == 1:
                random_angles = np.random.uniform(0, 2*np.pi, len(encoder_params))
            else:
                random_angles = np.random.uniform(0, 2*np.pi, len(encoder_params) + len(decoder_params))
            
            random_ansatz = update_circuit_parameters(ansatz, encoder_params, decoder_params, random_angles)
            # Run the circuit for each datapoint in the bucket
            for idx in bucket:
                # print('here',bucket_idx, idx)
                full_circuit = amplitude_encoding_circuits[idx].compose(random_ansatz).compose(swap_test)
                result = simulator.run(full_circuit, shots=4096).result()
                proportion_zero = result.get_counts(full_circuit).get('0', 0) / 4096
                final_results.append(proportion_zero)
        
        average_proportion = np.mean(final_results)

        bucket_result = {
            'bucket_idx': bucket_idx,
            'final_results': final_results,
            'average_proportion': average_proportion,
            'encoder_params': encoder_params
        }
        # print("done")
        iteration_results.append(bucket_result)

    return {
        'iteration': iteration,
        'buckets': buckets,
        'selected_features': selected_features,
        'bucket_results': iteration_results,
        'high_risk_indices': high_risk_indices,
        'compression_level': compression_level
    }

def main():
    """
    Main function to run the quantum autoencoder optimization process.

    This function parses command-line arguments, preprocesses the data,
    and runs multiple iterations of the optimization process using
    multiple threads. The results are then saved to a file.

    Args:
    None

    Returns:
    None
    """
    args = parse_arguments()
    num_qubits = args.num_qubits
    decoder_option = args.decoder_option
    num_threads = args.num_threads
    num_iterations = 1000
    num_bucketruns = 1
    target_proportion = 0.50
    anomaly_likelihood_per_bucket = 0.98

    start_time = time.time()

    file_path = './Data/Goldstein_Uchida_datasets/breast-cancer-unsupervised-ad.csv'

    #Preprocess the data
    preprocessed_data, high_risk_indices, _ = preprocess_goldstein_uchida(file_path)
    
    print(f"Initial dataset size: {len(preprocessed_data)}")
    print(f"Total number of anomalies: {len(high_risk_indices)}")
    print("High risk indices:", high_risk_indices)

    swap_test = create_swap_test_circuit(num_qubits)

    simulator = AerSimulator()
    # simulator = configure_noisy_simulator(num_qubits)

    all_results = []
    all_results_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for iteration in range(num_iterations):
            future = executor.submit(
                process_iteration,
                iteration,
                num_qubits,
                decoder_option,
                preprocessed_data,
                high_risk_indices,
                swap_test,
                simulator,
                target_proportion,
                anomaly_likelihood_per_bucket,
                num_iterations,
                num_bucketruns  # Add this new parameter
            )
            futures.append(future)

        for future in futures:
            result = future.result()
            with all_results_lock:
                all_results.append(result)

    print("\nAll iterations completed.")

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Total execution time: {execution_time:.2f} seconds")
    
    with open('results/ensemble_res.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    print("Results saved to ensemble_res.pkl")

if __name__ == "__main__":
    main()