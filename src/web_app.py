from flask import Flask, render_template, request, jsonify
import asyncio
import threading
import time
import os
from langchain_ollama import ChatOllama
from agent_graph import build_graph, AgentState

app = Flask(__name__, template_folder="templates", static_folder="static")

llm = ChatOllama(model="llama3.1:8b", temperature=0.0)
graph = build_graph(llm)

# Absolute path to /data folder
DATA_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))

progress_log = []
final_output = {}
task_done = False

@app.route("/")
def index():
    # Get .txt files under /data
    txt_files = []
    for root, dirs, files in os.walk(DATA_FOLDER):
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

        progress_log.append("📌 [Planning] Generating structured SOC report...")
        prompt = generate_soc_prompt(input_text)
        result = asyncio.run(async_invoke(llm, prompt))
        full_report = result.content.strip()
        progress_log.append("✅ Report ready.")
        progress_log.append(full_report)

        # Optional dummy metrics — replace with real analysis logic
        dummy_metrics = {
            "avg_packet_length": 523,
            "max_packet_length": 1412,
            "packet_count": 38,
            "source_ips": ["192.168.0.1", "10.0.0.2"],
            "ports": [22, 80, 443]
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

async def async_invoke(llm, prompt):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, llm.invoke, prompt)

def generate_soc_prompt(raw_text: str) -> str:
    return f"""
You are a cybersecurity analyst reporting to the SOC team "Rans Pupils".

You are analyzing the following anomaly cluster. Treat it as a single cohesive event.

=== Anomaly Cluster ===
{raw_text}
=======================

Generate a structured, concise report in this format:

📄 **Rans Pupils Anomaly Report**

🔹 **Anomaly ID**: Auto-generated or derived from IP  
🔹 **Confidence Score**: <0-100>

🔸 **Classification**: e.g. Denial-of-Service (DoS), Port Scan

🧪 **Impact Summary**:
- Describe affected systems, risks, goals

📊 **Key Metrics**:
- Packet Lengths: avg/max
- Packet Count: total
- Source IPs: [list]
- Affected Ports: [list]
- Flags: e.g. PSH/ACK/URG

🔍 **Supporting Evidence**:
- Unusual timing, packet bursts, repeated flags, etc.

🛡 **Recommendations**:
1. Investigate source IPs
2. Analyze suspicious ports
3. Apply firewall/rate-limit

Respond with **only this structured report**. No extra commentary.
"""
