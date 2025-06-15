from flask import Flask, render_template, request, jsonify, send_file
import asyncio
import threading
import time
import os
import io
import json
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

        progress_log.append("\ud83d\udccc [Planning] Generating structured SOC report...")
        prompt = generate_soc_prompt(input_text)
        result = asyncio.run(async_invoke(llm, prompt))
        full_report = result.content.strip()
        progress_log.append("\u2705 Report ready.")
        progress_log.append(full_report)

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

async def async_invoke(llm, prompt):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, llm.invoke, prompt)

def generate_soc_prompt(raw_text: str) -> str:
    return f"""
You are a cybersecurity analyst reporting to the SOC team \"Rans Pupils\".

You are analyzing the following anomaly cluster. Treat it as a single cohesive event.

=== Anomaly Cluster ===
{raw_text}
=======================

Generate a structured, concise report in this format:

\ud83d\udcc4 **Rans Pupils Anomaly Report**

\ud83d\udd39 **Anomaly ID**: Auto-generated or derived from IP  
\ud83d\udd39 **Confidence Score**: <0-100>

\ud83d\udd38 **Classification**: e.g. Denial-of-Service (DoS), Port Scan

\ud83e\uddea **Impact Summary**:
- Describe affected systems, risks, goals

\ud83d\udcca **Key Metrics**:
- Packet Lengths: avg/max
- Packet Count: total
- Source IPs: [list]
- Affected Ports: [list]
- Flags: e.g. PSH/ACK/URG

\ud83d\udd0d **Supporting Evidence**:
- Unusual timing, packet bursts, repeated flags, etc.

\ud83d\udee1 **Recommendations**:
1. Investigate source IPs
2. Analyze suspicious ports
3. Apply firewall/rate-limit

Respond with **only this structured report**. No extra commentary.
"""
