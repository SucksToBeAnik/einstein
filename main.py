import typer
from typing_extensions import Annotated
from agents.intent_classifier import agent

app = typer.Typer()


@app.command()
def run_einstein(
    query: Annotated[str, typer.Argument(help="Query to be processed")],
):
    response = agent.invoke({"query": query})

    


if __name__ == "__main__":
    app()
