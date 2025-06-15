from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph
from rich.console import Console
import time

console = Console()

class AgentState(TypedDict):
    raw_text: str
    plan: List[str]
    executors: Dict[str, str]
    reflection: str
    final_output: Dict[str, str]
    llm: object


def planner(state: AgentState) -> dict:
    prompt = f"""
    You are a cybersecurity analyst reporting to the SOC team "Rans Pupils".

    You are analyzing the following anomaly cluster. Treat it as a **single cohesive event**, not as independent anomalies:

    === Anomaly Cluster ===
    {state['raw_text']}
    =======================

    You must generate a **structured, concise, non-redundant report** with the following exact format:

    ---

    📄 **Rans Pupils Anomaly Report**

    🔹 **Anomaly ID**: Auto-generated or derived from source IP/cluster
    🔹 **Confidence Score**: <number out of 100>

    🔸 **Classification**: <Short name, e.g. "Denial-of-Service (DoS)", "Port Scan", "Suspicious Exfiltration">

    🧪 **Impact Summary**:
    - Clearly describe the potential threat and affected assets
    - Avoid repeating facts; be concise and unique

    📊 **Key Metrics**:
    - Packet Lengths: <avg/max packet size>
    - Number of Packets: <count>
    - Source IPs: <list>
    - Affected Ports: <list>
    - Flags Detected: <e.g. PSH/ACK/URG>

    🔍 **Supporting Evidence**:
    - One or two insights that show this is anomalous compared to normal traffic
    - Show deviations, unusual timing, or correlation across nodes

    🛡 **Recommendations**:
    1. Investigate <ip>:<port>...
    2. Analyze logs for ...
    3. Apply mitigation: <firewall, rate-limit, etc>

    ---

    🔒 Your output must match this structure exactly. Do **not** repeat IPs or patterns more than once. No extra commentary.

    Respond with **only the filled-in report**, nothing else.
    """


    console.print("[bold]Sending plan prompt to LLM...[/bold]")
    result = state["llm"].invoke(prompt)
    full_response = result.content.strip()
    console.print(f"⠼ Planning & Executing... {time.strftime('%H:%M:%S')} Received plan response: {full_response[:120]}...\n")

    # Extract exact steps
    steps = []
    for i in range(1, 5):
        match = re.search(rf"Step {i}:(.*?)\n(?=Step \d:|$)", full_response + "\nStep 5:", re.DOTALL | re.IGNORECASE)
        if match:
            steps.append(f"Step {i}:{match.group(1).strip()}")

    if len(steps) != 4:
        console.print(f"[red]⚠️ Expected 4 steps but got {len(steps)}.\nRaw response:\n{full_response}[/red]")

    return {"plan": steps}



import asyncio
import re

async def async_invoke(llm, prompt):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, llm.invoke, prompt)

async def executor(state: AgentState) -> dict:
    outputs = {}

    async def run_step(idx, orig_step):
        console.print(f"[bold]Executing {orig_step}[/bold]")
        start = time.time()
        try:
            result = await async_invoke(state["llm"], f"{orig_step}\n\nAnomaly:\n{state['raw_text']}")
            duration = time.time() - start
            console.print(f"[green]Completed[/green] in {duration:.1f} seconds\n")
            return idx, result.content.strip()
        except Exception as e:
            duration = time.time() - start
            console.print(f"[red]Failed[/red] after {duration:.1f} seconds: {e}\n")
            return idx, str(e)

    tasks = [run_step(idx, step) for idx, step in enumerate(state["plan"], 1)]
    results = await asyncio.gather(*tasks)

    # Map fixed step index to known keys
    key_map = {
        1: "assign",
        2: "classify",
        3: "justify",
        4: "recommend"
    }
    for idx, value in results:
        key = key_map.get(idx, f"step_{idx}")
        outputs[key] = value

    return {"executors": outputs}




def reflect_node(state: AgentState) -> dict:
    console.print("[bold]Reflecting on outputs...[/bold]")
    start = time.time()
    result = state["llm"].invoke(
        f"Confidence: {state['executors'].get('assign', '')}\n"
        f"Anomaly Type: {state['executors'].get('classify', '')}\n"
        f"Justification: {state['executors'].get('justify', '')}\n\n"
        "Is anything missing or uncertain? If good, reply 'All good'."
    )
    duration = time.time() - start
    console.print(f"[green]Reflection completed[/green] in {duration:.1f} seconds\n")
    return {"reflection": result.content.strip()}


def aggregator(state: AgentState) -> dict:
    if state["reflection"].lower().startswith("all good"):
        console.print("[bold]No rewrite needed (All good).[/bold]\n")
        return {"final_output": state["executors"]}

    console.print("[bold]Aggregator rewriting justification...[/bold]")
    start = time.time()
    prompt = (
        f"Original Justification:\n{state['executors'].get('justify', '')}\n\n"
        f"Issue Found:\n{state['reflection']}\n\n"
        "Please rewrite the justification with more clarity or detail."
    )
    result = state["llm"].invoke(prompt)
    updated = state["executors"].copy()
    updated["justify"] = result.content.strip()
    duration = time.time() - start
    console.print(f"[green]Aggregation completed[/green] in {duration:.1f} seconds\n")
    return {"final_output": updated}


def build_graph(llm) -> StateGraph:
    builder = StateGraph(AgentState)
    builder.add_node("planner", planner)
    builder.add_node("executor", executor)
    builder.add_node("reflector", reflect_node)
    builder.add_node("aggregator", aggregator)

    builder.set_entry_point("planner")
    builder.add_edge("planner", "executor")
    builder.add_edge("executor", "reflector")
    builder.add_edge("reflector", "aggregator")

    return builder.compile()

def rename_plan_steps(steps):
    renamed = []
    for step in steps:
        step_clean = step.strip()
        lower_step = step_clean.lower()

        if "confidence score" in lower_step:
            new_label = "Confidence Scoring"
        elif "classify" in lower_step or "anomaly type" in lower_step:
            new_label = "Anomaly Classification"
        elif "justify" in lower_step:
            new_label = "Justification"
        elif "recommend" in lower_step or "investigat" in lower_step:
            new_label = "Follow-up Recommendation"
        else:
            new_label = "Additional Analysis"

        new_step = re.sub(r"^\d+\.\s*(Anomaly\s*\d+:)?", f"{new_label}: ", step_clean)
        renamed.append(new_step)

    return renamed

