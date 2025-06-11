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
    prompt = (
        f"You are analyzing the following **network anomaly cluster**. "
        f"Treat all data below as part of ONE event.\n\n"
        f"Anomaly Cluster:\n\n{state['raw_text']}\n\n"
        "Your task is to produce a concise, clear 4-step analysis:\n\n"
        "1. Confidence Score (0–100): Estimate how confident you are this is an anomaly.\n"
        "2. Classification: Name the **type of anomaly** in a single phrase.\n"
        "3. Justification: Briefly justify the classification — avoid repetition or vague phrasing. Mention each unique insight only once.\n"
        "4. Recommendation: Give 1–3 **actionable next steps**, each on a new line. Avoid repeating the justification or general advice.\n\n"
        "**Avoid repeating the same facts multiple times across steps. Be precise and minimal.**\n"
        "Use this exact format:\n"
        "Step 1: Confidence Scoring: <score>\n"
        "Step 2: Anomaly Classification: <type>\n"
        "Step 3: Justification: <concise explanation>\n"
        "Step 4: Follow-up Recommendation: <one-line steps>\n"
    )

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

