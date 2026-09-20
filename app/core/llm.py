"""
Base LLM initialization.

Defaults to ChatMistralAI on `open-mistral-nemo` rather than a Premier-tier
Mistral model (mistral-small/medium) or OpenAI, since that's the model
you've had reliable free-tier quota for elsewhere. Swap MISTRAL_MODEL in
.env if you have paid access and want a stronger model.
"""

import os

from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI

load_dotenv()

DEFAULT_MODEL = os.getenv("MISTRAL_MODEL", "open-mistral-nemo")


def get_llm(temperature: float = 0.0) -> ChatMistralAI:
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError(
            "MISTRAL_API_KEY is not set. Copy .env.example to .env and fill it in."
        )
    return ChatMistralAI(
        model=DEFAULT_MODEL,
        temperature=temperature,
        api_key=api_key,
    )
