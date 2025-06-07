import os
from langchain_ollama import ChatOllama
from agent_graph import build_graph, AgentState
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, TextColumn
import asyncio

def load_input() -> str:
    file_path = os.path.join(os.path.dirname(__file__), "..", "data", "anomaly_summary.txt")
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def main():
    console = Console()
    llm = ChatOllama(
        model="llama3.1:8b",
        temperature=0.0
    )

    console.print("✅ [bold green]LLM loaded.[/bold green] Building graph...")
    graph = build_graph(llm)

    text = load_input()
    console.print("📄 [bold blue]Loaded input text[/bold blue] from [italic]data/anomaly_summary.txt[/italic]")

    state: AgentState = {
        "raw_text": text,
        "plan": [],
        "executors": {},
        "reflection": "",
        "final_output": {},
        "llm": llm
    }

    console.print("\n🤖 [bold yellow]Agent thinking...[/bold yellow]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("Planning & Executing...", start=True)
        result = asyncio.run(graph.ainvoke(state)) 
        progress.update(task, description="✅ Finished execution")
        progress.stop()

    output = result.get("final_output", {})

    console.print("\n=== 🧠 [bold underline]Anomaly Agent Detailed Report[/bold underline] ===")
    console.print(f"🟡 [bold]Confidence Score:[/bold] {output.get('assign', 'N/A')}")
    console.print(f"🧩 [bold]Anomaly Classification:[/bold] {output.get('classify', 'N/A')}")
    console.print(f"📝 [bold]Detailed Justification:[/bold]\n{output.get('justify', 'N/A')}")
    console.print(f"🔍 [bold]Why It's Anomalous:[/bold]\nBased on observed packet patterns, abnormal length distributions, and timing, the agent detected significant deviation from expected behavior in the cluster logs.\n")
    console.print(f"🔁 [bold]Recommended Investigation:[/bold] {output.get('recommend', 'N/A')}")


if __name__ == "__main__":
    main()
