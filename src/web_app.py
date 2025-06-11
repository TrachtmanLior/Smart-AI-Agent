from flask import Flask, render_template, request, jsonify
import asyncio
import threading
import time
from langchain_ollama import ChatOllama
from agent_graph import build_graph, AgentState, planner

app = Flask(__name__, template_folder="templates", static_folder="static")

# LangChain LLM setup
llm = ChatOllama(model="llama3.1:8b", temperature=0.0)
graph = build_graph(llm)

# Global state for progress updates
progress_log = []
final_output = {}
task_done = False

@app.route("/")
def index():
    return render_template("index.html")

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

        # Start the plan step (full report generation)
        progress_log.append("📌 [Planning] Generating structured SOC report...")

        # Build and run your new prompt directly
        prompt = generate_soc_prompt(input_text)
        result = asyncio.run(async_invoke(llm, prompt))
        full_report = result.content.strip()

        # Add it to progress log and final output
        progress_log.append("✅ Report ready.")
        progress_log.append(full_report)

        final_output.update({
            "assign": "SOC Report Ready",
            "classify": "SOC Report Ready",
            "justify": full_report,     # This is shown in the UI
            "recommend": "SOC Report Ready"
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

# Helper to run synchronous invoke from async context
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


if __name__ == "__main__":
    app.run(debug=True)
