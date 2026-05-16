from langchain_ollama import ChatOllama

ollama_models = ["gemma4:latest", "x/flux2-klein:4b", "qwen3-vl:2b", "nomic-embed-text:latest", "gemma3:1b", "qwen3:8b"]

intent_classifier_model = ChatOllama(model="qwen3:8b")
llm = ChatOllama(model="qwen3:8b")