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
        f"Given this anomaly:\n\n{state['raw_text']}\n\n"
        "List the steps to:\n"
        "1. Assign a confidence score (0–100).\n"
        "2. Classify the anomaly type.\n"
        "3. Justify the classification.\n"
        "4. Recommend follow-up investigation."
    )

    console.print("[bold]Sending plan prompt to LLM...[/bold]")
    result = state["llm"].invoke(prompt)
    full_response = result.content.strip()
    console.print(f"⠼ Planning & Executing... {time.strftime('%H:%M:%S')}Received plan response: {full_response[:120]}...\n")

    # Extract meaningful steps
    lines = full_response.splitlines()
    steps = [
        line.strip()
        for line in lines
        if line.lower().startswith("step") or line.strip()[:2] in {"1.", "2.", "3.", "4."}
    ]

    console.print(f"Parsed {len(steps)} valid plan steps.\n")
    return {"plan": steps}


import asyncio
import re

async def async_invoke(llm, prompt):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, llm.invoke, prompt)

async def executor(state: AgentState) -> dict:
    renamed_steps = rename_plan_steps(state["plan"])
    outputs = {}

    async def run_step(idx, orig_step, renamed_step):
        key = re.sub(r"[^\w]+", "_", renamed_step.lower().split(":")[0]).strip("_") or f"step_{idx}"
        console.print(f"[bold]Executing {renamed_step}[/bold]")
        start = time.time()
        try:
            result = await async_invoke(state["llm"], f"{orig_step}\n\nDetails:\n{state['raw_text']}")
            duration = time.time() - start
            console.print(f"[green]Completed[/green] in {duration:.1f} seconds\n")
            return key, result.content.strip()
        except Exception as e:
            duration = time.time() - start
            console.print(f"[red]Failed[/red] after {duration:.1f} seconds: {e}\n")
            return key, str(e)

    tasks = [
        run_step(idx, orig_step, renamed_step)
        for idx, (orig_step, renamed_step) in enumerate(zip(state["plan"], renamed_steps), 1)
    ]
    results = await asyncio.gather(*tasks)

    for key, value in results:
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

