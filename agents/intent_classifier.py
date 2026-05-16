from langgraph.types import Command
from typing import TypedDict, Literal
from pydantic import BaseModel, Field
from models.ollama import intent_classifier_model
from langgraph.graph import StateGraph, START, END




class QueryClassification(TypedDict):
    query: str

class ClassificationResponse(BaseModel):
    intent: Literal["summarize", "explain", "sources", "compare", "deep_research", "critique", "unknown"]
    confidence: int = Field(ge=0, le=100, description="Confidence score from 0-100")
    
    

def intent_classifier(state: QueryClassification) -> Command[Literal['summarize_agent', 'explain_agent', 'sources_agent', 'compare_agent', 'deep_research_agent', 'critique_agent', 'unknown_agent']]:
    system_prompt = """
    You are an expert at classifying user queries into one of the following categories:
    1. summarize: User wants a summary of a topic
    2. explain: User wants an explanation of a topic
    3. sources: User wants sources for a topic
    4. compare: User wants to compare topics
    5. deep_research: User wants deep research on a topic
    6. critique: User wants a critique of a topic
    7. unknown: User wants something else
    Finally, give a score from 0-100 on how confident you are in your classification.
    """
    structured_model = intent_classifier_model.with_structured_output(ClassificationResponse)
    response = structured_model.invoke([
        ("system", system_prompt),
        ("human", state["query"]),
    ])
    if response.confidence < 50:
        print(f"Model is not confident ({response.confidence}%) in its classification. Going to unknown agent.")
        goto = "unknown_agent"
    elif response.intent == "summarize":
        print(f"Model is {response.confidence}% confident that the intent is summarize.")
        goto = "summarize_agent"
    elif response.intent == "explain":  
        print(f"Model is {response.confidence}% confident that the intent is explain.")
        goto = "explain_agent"
    elif response.intent == "sources":
        print(f"Model is {response.confidence}% confident that the intent is sources.")
        goto = "sources_agent"
    elif response.intent == "compare":
        print(f"Model is {response.confidence}% confident that the intent is compare.")
        goto = "compare_agent"
    elif response.intent == "deep_research":
        print(f"Model is {response.confidence}% confident that the intent is deep_research.")
        goto = "deep_research_agent"
    elif response.intent == "critique":
        print(f"Model is {response.confidence}% confident that the intent is critique.")
        goto = "critique_agent"

    return Command(
        goto = goto
    )

def summarize_agent(state: QueryClassification):
    return Command(
        goto = END
    )


def explain_agent(state: QueryClassification):
    return Command(
        goto = END
    )

def sources_agent(state: QueryClassification):
    return Command(
        goto = END
    )

def compare_agent(state: QueryClassification):
    return Command(
        goto = END
    )

def deep_research_agent(state: QueryClassification):
    return Command(
        goto = END
    )

def critique_agent(state: QueryClassification):
    return Command(
        goto = END
    )

def unknown_agent(state: QueryClassification):
    return Command(
        goto = END
    )


workflow = StateGraph(QueryClassification)
workflow.add_node("intent_classifier", intent_classifier)
workflow.add_node("summarize_agent", summarize_agent)
workflow.add_node("explain_agent", explain_agent)
workflow.add_node("sources_agent", sources_agent)
workflow.add_node("compare_agent", compare_agent)
workflow.add_node("deep_research_agent", deep_research_agent)
workflow.add_node("critique_agent", critique_agent)
workflow.add_node("unknown_agent", unknown_agent)

workflow.add_edge(START, "intent_classifier")

agent = workflow.compile()