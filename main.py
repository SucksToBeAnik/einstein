import typer
from typing_extensions import Annotated
from agents.intent_classifier import agent
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

app = typer.Typer()
console = Console()


@app.command()
def run_einstein(
    query: Annotated[str, typer.Argument(help="Research query")],
):
    response = agent.invoke({"query": query})
    output = response.get("output") or ""
    if output:
        console.print(Panel(Markdown(output), title="[bold blue]Einstein[/bold blue]", border_style="blue"))


if __name__ == "__main__":
    app()
