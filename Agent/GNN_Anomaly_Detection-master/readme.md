

# README: Real-Time Network Anomaly Detection System

## Overview

This repository provides a real-time Network Intrusion Detection System (NIDS) that detects known and unknown threats using an unsupervised approach. The model operates on network traffic data by dynamically partitioning network flows, representing them as graph structures, and using Graph Neural Networks (GNNs) with dynamic clustering for anomaly detection.

Key features:
- **Zero-shot learning** for detecting unknown threats
- **Graph-based flow analysis** to capture network relationships
- **Real-time dynamic clustering** to identify coordinated attacks

## Installation

### Requirements
- Python 3.7+
- Required Python libraries (installed via `requirements.txt`):
  - PyTorch and Torch Geometric
  - Scikit-learn
  - HDBSCAN
  - NetworkX
  - Tshark for PCAP parsing (optional)

To install dependencies, run:
```bash
pip install -r requirements.txt
```

### Docker DevContainer Setup (Optional)

You can also run this project using a Docker DevContainer, which sets up a consistent development environment with all dependencies pre-installed. To do so:
1. Ensure you have Docker and VS Code (with the DevContainers extension) installed.
2. Open the repository in VS Code and select “Reopen in Container” from the command palette.
3. The DevContainer setup will automatically install all necessary dependencies and prepare the environment.

This option is ideal for streamlined setup and isolated environment management.

## Data Preparation

1. **Network Traffic Capture**:
   - Input network traffic data can be in PCAP or CSV format.
   - **Tshark** (command-line tool) can be used to preprocess PCAP files by extracting flow data if needed.

2. **Preprocessed Data**:
   - Preprocessed flows can be directly processed by the model.
   - Ensure extracted flows contain required features like IP addresses, ports, packet size, and TCP flags.

## Model Pipeline

The pipeline processes the data in several stages:

### 1. Flow Separation
   - **Objective**: Separate network traffic into individual flows based on IP, port, and protocol information.
   - Divide the traffic and extract essential features.
   - Outputs are used to construct the graph representation.

### 2. Graph Representation
   - **Objective**: Represent network flows in a dynamic tripartite graph.
   - The graph has three types of nodes: client IP, client socket (IP+port), and server socket.
   - Edges capture interactions between these nodes, creating a comprehensive view of network communication.

### 3. Graph Embedding
   - **Objective**: Transform graph-structured data into embeddings using Graph Convolutional Networks (GCNs).
   - Run GCN embedding on the tripartite graph to create dense vector representations.
   - This step prepares the data for clustering and anomaly detection.

### 4. Dynamic Clustering
   - **Objective**: Group nodes based on similarity and identify potential anomalies.
   - Uses the **HDBSCAN** clustering algorithm to identify clusters and outliers.
   - Runs in batches defined by the parameter `F` (number of flows per batch).

### 5. Anomaly Detection
   - **Objective**: Flag clusters as anomalous based on size, density, and centroid distance.
   - Clusters are flagged based on deviations from expected patterns, allowing real-time detection of abnormal traffic behaviors.

## Usage

### Running the Pipeline
To start the detection pipeline, run the following command:
```bash
python main.py path/to/pcap_or_csv_file [flow_count]
```
- **`input_file`**: Path to the PCAP or CSV data file.
- **`flow_count`**: Number of flows (`F`) per batch for dynamic graph embedding. Recommended values: `1000`, `2000`, or `4000`.

### Parameter Tuning

Parameters for feature selection, anomaly detection thresholds, clustering, and graph embedding can be configured in `config.py`.

- **Feature Selection**: Specify the network traffic features used in flow analysis.
  ```python
  features = [
      'amount', 'length', 'mean_packet_length', 
      'FIN_count', 'SYN_count', 'RST_count', 
      'PSH_count', 'ACK_count', 'URG_count',
  ]
  ```
- **Anomaly Detection Thresholds**:
  - `anomaly_score_history_size`: Number of past anomaly scores retained for thresholding.
  - `ann_threshold`: Score threshold to identify an anomaly in individual analysis.
  - `ann_history_threshold`: Threshold to detect historical patterns of anomalies.
  - `clustering_threshold`: Cluster density, amount and size range for flagging anomalies.
  - `network_threshold`: Threshold for network-wide anomaly alerting.

- **Graph Embedding Parameters**:
  - `hidden_size`: Size of hidden layers in the GCN.
  - `output_size`: Size of output embeddings for each node after GCN processing.

- **Evaluation Parameters**:
  - `attacker_ip`: IP address used to simulate attack traffic (evaluation purposes only).
  - `victim_ip`: IP address used to simulate target traffic (evaluation purposes only).
  
  > **Note**: IP addresses `attacker_ip` and `victim_ip` are used solely for evaluation.

- **Dataset Configuration**:
  - `dataset_type`: Specify `'flows-csv'`, `'packets-csv'` or `'packets-pcap'` format for input data.

### Evaluation Metrics
Accuracy, Precision, Recall, and F1-score are used to evaluate performance, with false positive rates tracked for robustness.

## Results and Analysis

The model has been validated on CIC-IDS-2017 and CSE-CIC-IDS-2018 datasets, achieving high detection accuracy and low false-positive rates.