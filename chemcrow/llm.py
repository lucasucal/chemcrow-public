import os
from typing import Any, List, Optional

import langchain
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models.base import BaseChatModel
from langchain.schema import ChatGeneration, ChatResult

try:
    from langchain.schema.messages import (
        AIMessage,
        BaseMessage,
        HumanMessage,
        SystemMessage,
    )
except ImportError:  # pragma: no cover - fallback for older langchain layouts
    from langchain.schema import (
        AIMessage,
        BaseMessage,
        HumanMessage,
        SystemMessage,
    )


class ChatAnthropicMessages(BaseChatModel):
    """Minimal Anthropic chat model built on the Messages API (/v1/messages).

    The ``ChatAnthropic`` shipped with the pinned langchain version targets the
    deprecated ``/v1/complete`` endpoint, which current Claude models reject.
    This wrapper talks to the Messages API directly via the ``anthropic`` SDK so
    the rest of the (old) langchain agent stack keeps working unchanged.
    """

    model: str = "claude-sonnet-4-5-20250929"
    temperature: float = 0.1
    max_tokens: int = 4096
    anthropic_api_key: Optional[str] = None
    streaming: bool = False
    client: Any = None

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        import anthropic

        self.client = anthropic.Anthropic(api_key=self.anthropic_api_key)

    @property
    def _llm_type(self) -> str:
        return "anthropic-messages"

    @staticmethod
    def _convert_messages(messages: List[BaseMessage]):
        """Split langchain messages into an Anthropic (system, messages) pair."""
        system_parts: List[str] = []
        converted: List[dict] = []
        for m in messages:
            content = m.content if isinstance(m.content, str) else str(m.content)
            if isinstance(m, SystemMessage):
                system_parts.append(content)
            elif isinstance(m, AIMessage):
                converted.append({"role": "assistant", "content": content})
            else:  # HumanMessage and anything else map to the user turn
                converted.append({"role": "user", "content": content})

        # Anthropic requires alternating roles; merge consecutive same-role turns.
        merged: List[dict] = []
        for msg in converted:
            if merged and merged[-1]["role"] == msg["role"]:
                merged[-1]["content"] += "\n" + msg["content"]
            else:
                merged.append(dict(msg))

        # Anthropic rejects trailing whitespace on the final message. The ReAct
        # agent ends its prompt with an assistant "Thought:" prefill that often
        # carries a trailing space, so trim the last turn (dropping it if empty).
        if merged:
            merged[-1]["content"] = merged[-1]["content"].rstrip()
            if not merged[-1]["content"]:
                merged.pop()

        # The conversation must start with a user turn and be non-empty.
        if not merged or merged[0]["role"] != "user":
            merged.insert(0, {"role": "user", "content": "."})

        system = "\n".join(p for p in system_parts if p) or None
        return system, merged

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        system, conv = self._convert_messages(messages)

        params: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": conv,
        }
        if system:
            params["system"] = system
        if stop:
            # Anthropic rejects whitespace-only stop sequences.
            cleaned = [s for s in stop if s and s.strip()]
            if cleaned:
                params["stop_sequences"] = cleaned

        response = self.client.messages.create(**params)

        text = "".join(
            block.text
            for block in response.content
            if getattr(block, "type", None) == "text"
        )

        if run_manager and text:
            run_manager.on_llm_new_token(text)

        message = AIMessage(content=text)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def get_num_tokens(self, text: str) -> int:
        """Rough estimate; avoids downloading a GPT-2 tokenizer at runtime."""
        return max(1, len(text) // 4)


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "qwen/qwen3-32b"


def detect_provider(model: str) -> str:
    # OpenRouter slugs look like "meta-llama/llama-3.2-3b-instruct:free"
    if "/" in model or model.startswith("openrouter"):
        return "openrouter"
    if model.startswith("claude"):
        return "anthropic"
    if model.startswith("gpt-") or model.startswith("text-"):
        return "openai"
    raise ValueError(
        f"Cannot detect LLM provider for model '{model}'. "
        "Use an OpenRouter slug (org/model), claude-*, or gpt-* name."
    )


def get_api_key(provider: str, api_key: Optional[str] = None) -> Optional[str]:
    if api_key:
        return api_key
    if provider == "anthropic":
        return os.getenv("ANTHROPIC_API_KEY")
    if provider == "openrouter":
        return os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if provider == "openai":
        return os.getenv("OPENAI_API_KEY")
    return None


def make_llm(
    model: str,
    temp: float,
    api_key: Optional[str] = None,
    streaming: bool = False,
    provider: Optional[str] = None,
) -> BaseLanguageModel:
    if provider is None:
        provider = detect_provider(model)

    key = get_api_key(provider, api_key)
    callbacks = [StreamingStdOutCallbackHandler()] if streaming else []

    if provider == "anthropic":
        return ChatAnthropicMessages(
            model=model,
            temperature=temp,
            streaming=streaming,
            callbacks=callbacks,
            anthropic_api_key=key,
        )

    if provider == "openrouter":
        # OpenRouter is OpenAI-compatible; old langchain ChatOpenAI accepts
        # openai_api_base to point at a custom endpoint.
        return langchain.chat_models.ChatOpenAI(
            temperature=temp,
            model_name=model,
            request_timeout=1000,
            streaming=streaming,
            callbacks=callbacks,
            openai_api_key=key,
            openai_api_base=OPENROUTER_BASE_URL,
        )

    if model.startswith("gpt-3.5-turbo") or model.startswith("gpt-4"):
        return langchain.chat_models.ChatOpenAI(
            temperature=temp,
            model_name=model,
            request_timeout=1000,
            streaming=streaming,
            callbacks=callbacks,
            openai_api_key=key,
        )

    if model.startswith("text-"):
        return langchain.OpenAI(
            temperature=temp,
            model_name=model,
            streaming=streaming,
            callbacks=callbacks,
            openai_api_key=key,
        )

    raise ValueError(f"Invalid model name: {model}")


def make_embeddings(openai_api_key: Optional[str] = None):
    key = openai_api_key or os.getenv("OPENAI_API_KEY")
    if key:
        from langchain.embeddings.openai import OpenAIEmbeddings

        return OpenAIEmbeddings(openai_api_key=key)

    from langchain.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
