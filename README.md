
# Smart AI Agent for Anomaly Detection

👤 **Authors**  
Lior Trachtman & Matan Klein

A smart AI agent that combines Graph Neural Networks (GNNs) and Large Language Models (LLMs) to improve anomaly detection. Through a transparent Plan-and-Execute architecture, each anomaly is evaluated across four reasoning steps: confidence scoring, classification, justification, and actionable recommendation. This integrated pipeline ensures both high detection accuracy and clear, human-readable insights—minimizing false positives while reducing the analyst workload.

---

## 🗂️ Project Structure

```
Smart-AI-Agent/
│
├── README.md                          # Project documentation
├── requirements.txt                  # Core Python dependencies
├── plan-and-execute.ipynb            # Notebook showing LLM reasoning flow
├── src/                              # Main execution and interface logic
│   ├── main.py                       # Orchestrates end-to-end detection
│   ├── agent_graph.py                # LangGraph-based planner/executor
│   ├── web_app.py                    # Flask web interface
│   └── templates/index.html          # Web UI template
│
├── Agent/                            # Anomaly detection engine & datasets
│   ├── GNN_Anomaly_Detection-master/ # GNN detection pipeline
│   │   ├── src/                      # Graph embedding, flow parsing, etc.
│   ├── data/                         # CIC & 2017 datasets (CSV, summaries)
│   └── run_multiple.sh               # Batch anomaly detection script
│
├── run_gnn_anomaly.sh                # Shell wrapper for GNN inference
```

---

## ✨ Features

1. **Input**: `pcap` files ingested and transformed into flow-level representations.
2. **Flow Separation**: Parses traffic flows to structured network sessions.
3. **Graph Representation**: Converts sessions to graphs with contextual embeddings.
4. **GNN Module**: Trains on graph features to identify anomalies.
5. **Dynamic Clustering**: Groups similar anomalies to identify correlated threats.
6. **LLM Module**: `Llama 3.1` (8B params) generates detailed, human-readable summaries.
7. **Planning & Execution**: Follows a LangGraph flow with steps like `planner`, `executor`, `reflector`, `aggregator`.
8. **Reporting**: Outputs structured findings with event timelines, flow metrics, and response recommendations.


## 🚀 Installation

```bash
git clone https://github.com/TrachtmanLior/Smart-AI-Agent.git
cd Smart-AI-Agent
# Create virtual environment (optional)
python -m venv venv && source venv/bin/activate
# Install dependencies
pip install -r requirements.txt
```

---

### 🔧 How To Launch:

The project includes a browser-based UI to explore anomaly clusters visually and interactively.


```bash
export FLASK_APP=web_app.py
export PYTHONPATH=.
flask run
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

### UI Features

- 📁 File selector for preloaded `.txt` anomaly logs
- 📝 Paste cluster logs for live analysis
- 📄 Structured SOC reports with LLM-generated summaries
- 📊 Chart.js visualizations of anomaly metrics
- 💾 Export reports to PDF and JSON
- 🕓 Real-time progress log and elapsed time tracking

---

## Terminal Usage

```bash
python main.py --input traffic.pcap --output report.json
```

### Parameters

| Argument     | Description                          |
|--------------|--------------------------------------|
| `--input`    | Path to input PCAP file              |
| `--output`   | Path to output report file           |
| `--verbose`  | Enable debug information (optional)  |


