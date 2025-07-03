from flask import Flask, render_template, request, jsonify, send_file
import asyncio
import threading
import time
import os
import io
import json
import re
from langchain_ollama import ChatOllama
from agent_graph import build_graph, AgentState
from fpdf import FPDF

app = Flask(__name__, template_folder="templates", static_folder="static")

llm = ChatOllama(model="llama3.1:8b", temperature=0.0)
graph = build_graph(llm)

DATA_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))

progress_log = []
final_output = {}
task_done = False


@app.route("/")
def index():
    txt_files = []
    for root, _, files in os.walk(DATA_FOLDER):
        for file in files:
            if file.endswith(".txt"):
                rel_path = os.path.relpath(os.path.join(root, file), DATA_FOLDER)
                txt_files.append(rel_path)
    return render_template("index.html", file_options=sorted(txt_files))


@app.route("/load-file", methods=["POST"])
def load_file():
    filename = request.json.get("filename")
    if not filename:
        return jsonify({"error": "No filename provided"}), 400
    file_path = os.path.join(DATA_FOLDER, filename)
    try:
        with open(file_path, "r") as f:
            content = f.read()
        return jsonify({"content": content})
    except FileNotFoundError:
        return jsonify({"error": "File not found"}), 404


@app.route("/start-task", methods=["POST"])
def start_task():
    global progress_log, final_output, task_done
    progress_log = []
    final_output = {}
    task_done = False

    input_text = request.form["input_text"]

    def run_agent():
        nonlocal input_text
        state: AgentState = {
            "raw_text": input_text,
            "plan": [],
            "executors": {},
            "reflection": "",
            "final_output": {},
            "llm": llm
        }

        progress_log.append("📌 [Planning] Generating structured SOC reports for each anomaly...")

        anomalies = split_anomalies(input_text)
        full_report = ""

        for cluster_idx, (cluster_id, anomaly_texts) in enumerate(anomalies, 1):
            progress_log.append(f"📦 Processing cluster {cluster_id} ({len(anomaly_texts)} nodes)...")
            
            # 1️⃣ Get a single cluster-level summary with confidence & classification
            cluster_text_block = "\n\n".join(anomaly_texts)
            cluster_summary_prompt = f"""
        You are a cybersecurity analyst from the SOC team "Rans Pupils".

        Analyze the following anomaly cluster AS A WHOLE:

        === Anomaly Cluster ===
        {cluster_text_block}
        =======================

        Write a **cluster-level summary** in Markdown with exactly this format:

        ## 📦 Cluster {cluster_id} Summary

        **Confidence Score**: <single overall score out of 100>  
        **Classification**: "Potential Malicious Activity"

        **Summary**:
        - 3–5 sentences describing overall threat, common patterns, risk level
        - Recommended SOC priority

        Do not mention individual node IDs in the summary.
        """
            cluster_summary_result = llm.invoke(cluster_summary_prompt)
            cluster_summary = cluster_summary_result.content.strip()

            # 2️⃣ Generate all node reports without confidence/classification
            node_reports = []
            node_ids = []
            for idx, anomaly_text in enumerate(anomaly_texts, 1):
                # Detailed per-node report
                prompt = generate_soc_prompt(anomaly_text)
                result = llm.invoke(prompt)
                report = result.content.strip()
                node_reports.append(f"\n\n### 🧾 Report for Node {idx} in Cluster {cluster_id}\n{report}\n")

                # Extract just the Anomaly ID (for the brief list)
                node_id_prompt = f"""
        From the following anomaly text, extract **only** the Anomaly ID (e.g. IP and port) in this format:

        - <Anomaly ID>

        Here is the raw anomaly text:
        {anomaly_text}
        """
                node_id_result = llm.invoke(node_id_prompt)
                node_ids.append(node_id_result.content.strip())

                progress_log.append(f"✅ Finished processing node {idx} from cluster {cluster_id}")

            # 3️⃣ Combine all cluster parts
            cluster_text = f"\n\n{cluster_summary}\n"
            cluster_text += "".join(node_reports)
            cluster_text += "\n\n### 📜 Brief Node Anomalies\n"
            cluster_text += "\n".join(node_ids)

            # 4️⃣ Append to full report
            full_report += cluster_text



        progress_log.append("✅ All nodes processed.")

        dummy_metrics = {
            "avg_packet_length": 523,
            "max_packet_length": 1412,
            "packet_count": 38,
            "source_ips": ["192.168.0.1", "10.0.0.2"],
            "ports": [22, 80, 443],
            "flags": {"PSH": 12, "ACK": 20, "URG": 6},
            "risk_level": "High"
        }

        final_output.update({
            "assign": "SOC Report Ready",
            "classify": "SOC Report Ready",
            "justify": full_report,
            "recommend": "SOC Report Ready",
            "metrics": dummy_metrics
        })

        global task_done
        task_done = True

    threading.Thread(target=run_agent).start()
    return jsonify({"status": "started"})


@app.route("/progress")
def progress():
    return jsonify({
        "log": progress_log,
        "done": task_done,
        "result": final_output if task_done else None
    })


@app.route("/download/json")
def download_json():
    if not final_output:
        return "No data available", 400
    return jsonify(final_output)


@app.route("/download/txt")
def download_txt():
    if not final_output:
        return "No report to download", 400
    buffer = io.StringIO()
    buffer.write(final_output.get("justify", "No report"))
    buffer.seek(0)
    return send_file(io.BytesIO(buffer.read().encode()), mimetype="text/plain", as_attachment=True, download_name="report.txt")


@app.route("/download/pdf")
def download_pdf():
    if not final_output:
        return "No report to download", 400
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in final_output.get("justify", "No report").splitlines():
        pdf.multi_cell(0, 10, line)
    buffer = io.BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    return send_file(buffer, mimetype="application/pdf", as_attachment=True, download_name="report.pdf")


def split_anomalies(raw_text: str) -> list[tuple[str, list[str]]]:
    pattern = re.compile(r"(found(?: \([^)]+\))? anomaly in node: ({.*?}))", re.DOTALL)
    results = {}
    for match in pattern.finditer(raw_text):
        full_text = match.group(1).strip()
        import ast
        try:
            node = ast.literal_eval(match.group(2))
            cluster_id = str(node.get("cluster", "unknown"))
        except Exception:
            cluster_id = "unknown"
        
        if cluster_id not in results:
            results[cluster_id] = []
        results[cluster_id].append(full_text)
    
    # Convert to list of (cluster_id, list of anomaly texts)
    return [(cluster_id, anomalies) for cluster_id, anomalies in results.items()]



def generate_soc_prompt(raw_text: str) -> str:
    return f"""
You are a cybersecurity analyst reporting to the SOC team "Rans Pupils".

Analyze the following anomaly log and generate a detailed technical report **without any confidence score or classification**.

=== Anomaly ===
{raw_text}
===============

Respond in this exact format:

📄 **Rans Pupils Anomaly Report**

🧪 **Impact Summary**:
- Describe affected systems, risks, goals

📊 **Key Metrics**:
- Packet Lengths: avg/max
- Packet Count: total
- Source IPs: [list]
- Affected Ports: [list]
- Flags: e.g. PSH/ACK/URG

🔍 **Supporting Evidence**:
- Unusual timing, repeated flags, etc.

🛡 **Recommendations**:
1. Investigate source IPs
2. Apply firewall/rate-limit
3. Monitor traffic anomalies

Respond with **only the report**.
"""
