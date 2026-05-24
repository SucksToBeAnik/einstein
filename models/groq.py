from typing import Literal

from langchain_groq import ChatGroq

from settings import settings

ModelName = Literal["default", "cheap", "coder"]


class GroqModel:
    _models: dict[ModelName, str] = {
        "default": "openai/gpt-oss-20b",
        "cheap": "llama-3.1-8b-instant",
        "coder": "qwen/qwen3-32b",
    }

    def use(self, name: ModelName = "default") -> ChatGroq:
        model_name = self._models[name]

        return ChatGroq(
            model=model_name,
            api_key=settings.groq_api_key,
        )


groq_model = GroqModel()