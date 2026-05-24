import operator
from curses import init_color
from typing import Annotated, Literal, Optional, TypedDict, cast

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send
from pydantic import BaseModel, Field

from models.groq import groq_model
from prompts.system_prompts import DEEP_RESEARCH_SYSTEM_PROMPT


class ResearchState(TypedDict):
    query: str
    analyses: Annotated[list[str], operator.add]
    output: Optional[str]
    reflection_iterations: int


class ClassificationResponse(BaseModel):
    intent: Literal[
        "summarize",
        "explain",
        "sources",
        "compare",
        "deep_research",
        "critique",
        "unknown",
    ]
    confidence: int = Field(ge=0, le=100, description="Confidence score from 0-100")


class CompareSubjects(BaseModel):
    subject_a: str = Field(description="First subject being compared")
    subject_b: str = Field(description="Second subject being compared")


def intent_classifier(
    state: ResearchState,
) -> Command[
    Literal[
        "summarize_agent",
        "explain_agent",
        "sources_agent",
        "compare_agent",
        "deep_research_agent",
        "critique_agent",
        "unknown_agent",
    ]
]:
    system_prompt = """
    You are an expert at classifying user queries into one of the following categories:
    1. summarize: User wants a summary of a topic
    2. explain: User wants an explanation of a topic
    3. sources: User wants sources for a topic
    4. compare: User wants to compare topics
    5. deep_research: User wants deep research on a topic
    6. critique: User wants a critique of a topic
    7. unknown: User wants something else
    Give a score from 0-100 on how confident you are in your classification.
    """
    structured_model = groq_model.use("default").with_structured_output(
        ClassificationResponse
    )
    response = structured_model.invoke(
        [
            ("system", system_prompt),
            ("human", state["query"]),
        ]
    )
    response = cast(ClassificationResponse, response)

    if response.confidence < 50:
        print(f"Low confidence ({response.confidence}%) — routing to unknown.")
        goto = "unknown_agent"
    else:
        intent_map = {
            "summarize": "summarize_agent",
            "explain": "explain_agent",
            "sources": "sources_agent",
            "compare": "compare_agent",
            "deep_research": "deep_research_agent",
            "critique": "critique_agent",
            "unknown": "unknown_agent",
        }
        goto = intent_map.get(response.intent, "unknown_agent")
        print(f"Routing as: {response.intent} (confidence: {response.confidence}%)")

    return Command(goto=goto)


def compare_agent(state: ResearchState) -> Command:
    """Parse two subjects then fan out to analyze_subject in parallel."""
    structured = groq_model.use("default").with_structured_output(CompareSubjects)
    parsed = structured.invoke(
        [
            (
                "system",
                "Extract exactly two things being compared from the query. Return them as subject_a and subject_b.",
            ),
            ("human", state["query"]),
        ]
    )
    parsed = cast(CompareSubjects, parsed)
    print(f"Comparing: '{parsed.subject_a}' vs '{parsed.subject_b}'")
    return Command(
        goto=[
            Send(
                "analyze_subject",
                {"subject": parsed.subject_a, "query": state["query"]},
            ),
            Send(
                "analyze_subject",
                {"subject": parsed.subject_b, "query": state["query"]},
            ),
        ]
    )


def analyze_subject(state: dict) -> dict:
    """Runs in parallel for each subject. Returns one analysis to accumulate."""
    response = groq_model.use("default").invoke(
        [
            (
                "system",
                "You are a research assistant. Provide a thorough, structured analysis of the given subject. "
                "Cover: key characteristics, strengths, weaknesses, and best use cases.",
            ),
            ("human", f"Context: {state['query']}\n\nAnalyze: {state['subject']}"),
        ]
    )
    return {"analyses": [f"## {state['subject']}\n\n{response.content}"]}


def synthesize_compare(state: ResearchState) -> dict:
    """Fan-in: merge both analyses into a final structured comparison."""
    analyses_text = "\n\n---\n\n".join(state["analyses"])
    response = groq_model.use("default").invoke(
        [
            (
                "system",
                "You are a research assistant. Given analyses of two subjects, produce a clear structured comparison. "
                "Cover: key differences, trade-offs, similarities, and when to choose each.",
            ),
            ("human", f"Original query: {state['query']}\n\n{analyses_text}"),
        ]
    )
    return {"output": response.content}


def summarize_agent(state: ResearchState) -> dict:
    return {"output": "[summarize not yet implemented]"}


def explain_agent(state: ResearchState) -> dict:
    return {"output": "[explain not yet implemented]"}


def sources_agent(state: ResearchState) -> dict:
    return {"output": "[sources not yet implemented]"}


def deep_research_agent(state: ResearchState):
    init_messages = [("system", DEEP_RESEARCH_SYSTEM_PROMPT), ("user", state["query"])]
    response = groq_model.use("default").invoke(init_messages)
    return Command(
        update={"output": response.content, "reflection_iterations": 0},
        goto="reflect_deep_research",
    )


def reflect_deep_research(state: ResearchState) -> dict:
    if not state["output"]:
        return Command(update={"reflection_iterations": 0}, goto=END)
    if state["reflection_iterations"] >= 2:
        return Command(update={"reflection_iterations": 0}, goto=END)

    init_critique_messages = [
        (
            "system",
            "Review this response for: gaps in knowledge, inaccuracies, lack of clarity. What should be improved?",
        ),
        ("user", state["output"]),
    ]
    critique_response = groq_model.use("cheap").invoke(init_critique_messages)
    init_revision_messages = [
        (
            "system",
            "You are a helpful assistant that revises responses to improve clarity and accuracy.",
        ),
        ("user", critique_response.content),
    ]
    revision_response = groq_model.use("default").invoke(init_revision_messages)
    return Command(
        update={
            "reflection_iterations": state["reflection_iterations"] + 1,
            "output": revision_response.content,
        },
        goto="reflect_deep_research",
    )


def critique_agent(state: ResearchState) -> dict:
    return {"output": "[critique not yet implemented]"}


def unknown_agent(state: ResearchState) -> dict:
    return {
        "output": "Your query was unclear. Could you rephrase or provide more detail?"
    }


# --- Graph ---

workflow = StateGraph(ResearchState)

workflow.add_node("intent_classifier", intent_classifier)
workflow.add_node("summarize_agent", summarize_agent)
workflow.add_node("explain_agent", explain_agent)
workflow.add_node("sources_agent", sources_agent)
workflow.add_node("compare_agent", compare_agent)
workflow.add_node("analyze_subject", analyze_subject)
workflow.add_node("synthesize_compare", synthesize_compare)
workflow.add_node("deep_research_agent", deep_research_agent)
workflow.add_node("reflect_deep_research", reflect_deep_research)
workflow.add_node("critique_agent", critique_agent)
workflow.add_node("unknown_agent", unknown_agent)

workflow.add_edge(START, "intent_classifier")

# Compare parallel flow: fan-out via Send, fan-in via synthesize_compare
workflow.add_edge("analyze_subject", "synthesize_compare")
workflow.add_edge("synthesize_compare", END)

# All other terminal nodes
for node in (
    "summarize_agent",
    "explain_agent",
    "sources_agent",
    "critique_agent",
    "unknown_agent",
):
    workflow.add_edge(node, END)

agent = workflow.compile()
