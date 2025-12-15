from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
    CompletionResponseAsyncGen,
)
from llama_index.core.llms.callbacks import llm_completion_callback
from typing import Any, Sequence
import os
from utils.llm import call_llm, call_llm_async
from llama_index.core.base.llms.types import ChatMessage, ChatResponse
from llama_index.core.llms import MessageRole

class CustomLLMWrapper(CustomLLM):
    """
    A custom LlamaIndex LLM that wraps the call_llm functions.
    """
    model_name: str = os.getenv('MODEL_NAME', 'default-model')

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=8192,
            num_output=2048,
            is_chat_model=True,
            model_name=self.model_name,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        response = call_llm(prompt)
        return CompletionResponse(text=response or "")

    @llm_completion_callback()
    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseGen:
        raise NotImplementedError("Stream complete is not implemented yet.")

    @llm_completion_callback()
    async def acomplete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        response = await call_llm_async(prompt)
        return CompletionResponse(text=response or "")

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        raise NotImplementedError("Async stream complete is not implemented yet.")

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        prompt = messages[-1].content or ""
        response = call_llm(prompt) or ""
        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=response),
            raw={"model": self.model_name, "message": response},
        )

    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        prompt = messages[-1].content or ""
        response = await call_llm_async(prompt) or ""
        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=response),
            raw={"model": self.model_name, "message": response},
        )

