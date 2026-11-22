"""Settings. The API key is read from the environment / .env file, never hard-coded."""
import os

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = "gpt-4o-mini"


def get_model_name() -> str:
    return os.getenv("OPENAI_MODEL", DEFAULT_MODEL)


def has_api_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY", "").strip())


def build_llm(model: str | None = None, temperature: float = 0.0):
    """Any LangChain chat model works with the chains; swap this function to change provider."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model or get_model_name(),
        temperature=temperature,
        timeout=60,
        max_retries=3,  # network errors and rate limits
    )
