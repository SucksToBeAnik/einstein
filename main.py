import time
import mlflow
import typer
from typing_extensions import Annotated
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

mlflow.set_tracking_uri("sqlite:///./mlflow.db")
mlflow.langchain.autolog()

from agents.intent_classifier import agent
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

app = typer.Typer()
console = Console()
mlflow.set_experiment("einstein_experiments")

# Cost per million tokens (input, output) — update as pricing changes
COST_PER_MILLION: dict[str, tuple[float, float]] = {
    "openai/gpt-oss-20b":  (0.90, 0.90),
    "llama-3.1-8b-instant": (0.05, 0.08),
    "qwen/qwen3-32b":      (0.29, 0.59),
}


class TokenTracker(BaseCallbackHandler):
    """Accumulates token usage across all LLM calls in a single agent run."""

    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_cost = 0.0

    def on_llm_end(self, response: LLMResult, **kwargs):
        for generations in response.generations:
            for gen in generations:
                msg = getattr(gen, "message", None)
                usage = getattr(msg, "usage_metadata", None)
                if not usage:
                    continue
                inp = usage.get("input_tokens", 0)
                out = usage.get("output_tokens", 0)
                self.input_tokens += inp
                self.output_tokens += out

                model = getattr(msg, "response_metadata", {}).get("model_name", "")
                if model in COST_PER_MILLION:
                    in_cost, out_cost = COST_PER_MILLION[model]
                    self.total_cost += (inp * in_cost + out * out_cost) / 1_000_000


@app.command()
def run_einstein(
    query: Annotated[str, typer.Argument(help="Research query")],
):
    tracker = TokenTracker()

    with mlflow.start_run():
        mlflow.log_param("query", query)
        start = time.time()

        response = agent.invoke({"query": query}, config={"callbacks": [tracker]})
        output = response.get("output") or ""

        mlflow.log_metric("total_time_seconds", round(time.time() - start, 3))
        mlflow.log_metric("input_tokens", tracker.input_tokens)
        mlflow.log_metric("output_tokens", tracker.output_tokens)
        mlflow.log_metric("total_tokens", tracker.input_tokens + tracker.output_tokens)
        mlflow.log_metric("total_cost_usd", round(tracker.total_cost, 6))

        # detected_intent is logged from inside intent_classifier node
        if output:
            console.print(
                Panel(
                    Markdown(output),
                    title="[bold blue]Einstein[/bold blue]",
                    border_style="blue",
                )
            )


if __name__ == "__main__":
    app()
