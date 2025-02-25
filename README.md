# Quorum: Zero-Training Unsupervised Anomaly Detection using Quantum Autoencoders

## Overview

Quorum is a novel quantum computing framework designed for unsupervised anomaly detection that requires no training.

## Project Structure

```
Quorum/
│
├── main.py                        # Main execution script
├── data_bucketing.py              # Data bucketing implementation
├── feature_selection.py           # Feature selection algorithms
├── swap_test_circuit.py           # Quantum SWAP test implementation
│
├── Ansatzes/                      # Quantum circuit ansatzes
│   ├── rx_rz_ansatz.py            # Ansatz using RX and RZ gates
│   ├── ry_cx_ansatz.py            # Ansatz using RY and CX gates
│   └── ry_rz_ansatz.py            # Ansatz using RY and RZ gates
│
├── Embedding/                     # Quantum data embedding methods
│   └── range_amplitude_enc.py     # Range-based amplitude encoding
│
├── Preprocessing/                 # Data preprocessing modules
│   ├── goldstein_uchida_preprocess.py  # Preprocessor for Goldstein-Uchida datasets
│   └── ccpp_preprocess.py         # Preprocessor for Combined Cycle Power Plant dataset
│
└── Data/                          # Dataset directory (not included in repository)
    └── Goldstein_Uchida_datasets/
        └── breast-cancer-unsupervised-ad.csv
        └── letter-unsupervised-subset-ad.csv
        └── pen-global-unsupervised-ad.csv
    └── PowerPlant/
        └── processed_ccpp_dataset.csv




```

## Requirements

- Python 3.8+
- NumPy
- pandas
- scikit-learn
- Qiskit
- Qiskit Aer

## Installation

```bash
# Clone the repository
git clone https://github.com/positivetechnologylab/Quorum.git
cd Quorum

# Install dependencies from requirements.txt
pip install -r requirements.txt
```

## Usage

The main entry point of the framework is `main.py`, which can be executed with command-line arguments to specify the number of qubits and decoder option:

```bash
python main.py <num_qubits> <decoder_option> [--num_threads <threads>]
```

### Parameters:

- `num_qubits`: Number of qubits to use for quantum encoding (integer)
- `decoder_option`: Decoder type (1 for Qiskit's .inverse(), 2 for manual decoder)
- `--num_threads`: Optional parameter to specify the number of threads for parallel processing (default: 4)

### Example:

```bash
python main.py 4 1 --num_threads 8
```

This will run the quantum anomaly detection with 4 qubits, using decoder option 1 (Qiskit's inverse), and 8 threads for parallel execution.

## Core Components

### Data Preprocessing

Two preprocessing modules are provided:
- `goldstein_uchida_preprocess.py`: Preprocesses datasets from the Goldstein-Uchida collection
- `ccpp_preprocess.py`: Preprocesses the Combined Cycle Power Plant dataset

Both modules normalize the data and identify anomaly indices.

### Feature Selection

The `feature_selection.py` module offers multiple strategies for feature selection:
- Strategy 'a': Selects top features based on PCA importance
- Strategy 'b': Selects bottom features based on PCA importance
- Strategy 'c': Selects a mix of top and bottom features
- Strategy 'd': Weighted random selection based on feature importance
- Strategy 'e': Uniform random selection of features (used for experimental evaluation)

### Data Bucketing

The `data_bucketing.py` module implements probabilistic bucketing of data points.

### Quantum Embedding

The `range_amplitude_enc.py` module provides amplitude encoding circuit structure to represent classical data as quantum states.

### Quantum Ansatzes

Three different ansatz options are provided:
- `rx_rz_ansatz.py`: Uses RX and RZ rotation gates
- `ry_cx_ansatz.py`: Uses RY rotation gates and CX (CNOT) gates
- `ry_rz_ansatz.py`: Uses RY and RZ rotation gates

### Quantum Circuits

The `swap_test_circuit.py` implements the quantum SWAP test, which measures the similarity between two quantum states.

## Algorithm Overview

1. **Data Preprocessing**: Normalize data and identify anomalies
2. **Feature Selection**: Select features based on the specified strategy
3. **Data Bucketing**: Group data points into buckets with a target probability of containing anomalies
4. **Quantum Encoding**: Encode data into quantum states using amplitude encoding
5. **Autoencoder**: Process the quantum states through a parameterized encoder-decoder circuit
6. **SWAP Test**: Measure the similarity between input and output states
7. **Anomaly Detection**: Identify anomalies based on the similarity measurements

## Output

The results of the execution are saved as a pickle file in the `results/` directory as `ensemble_res.pkl`. This file contains information about:
- Buckets created during execution
- Selected features for each iteration
- Compression levels used
- Results from quantum circuit executions

## Noise Simulation

The framework supports realistic noise simulation mimicking IBM's Brisbane quantum computer specifications. This can be enabled in the main script by uncommenting the appropriate simulator configuration.

## Copyright

Copyright © 2025 Positive Technology Lab. All rights reserved. For permissions, contact ptl@rice.edu.
