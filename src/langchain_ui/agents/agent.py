
import logging
from typing import Any, Iterable
from uuid import uuid4
from deepagents import create_deep_agent
from langchain_core.messages import AIMessageChunk
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableGenerator
from langchain_surf import ChatWillma

from langchain_ui.agents.agent_utils import format_data, get_chunkid, get_content
from langchain_ui.agents.default_agent_instruction import default_instructions

logger = logging.getLogger(__name__)


def create_willma_agent(
    api_key: str,
    model: str = "default-text-large",
    temperature: float = 0.1,
    max_tokens: int = 1000,
    timeout: int = 30,
    tools: list[Any] | None = None,
    instructions: str | None = None,
    backend: Any | None = None,
    skills: list[str] | None = None,
) -> Any:
    """Create and return a Willma agent configured for OpenAI-compatible SSE streaming.

    Args:
        api_key: API key for authentication with the Willma service.
        model: Model identifier to use. Defaults to "default-text-large".
        temperature: Sampling temperature for generation. Defaults to 0.1.
        max_tokens: Maximum number of tokens in the response. Defaults to 100000.
        timeout: Request timeout in seconds. Defaults to 30.
        tools: Optional list of LangChain tools to provide to the agent.
        instructions: Optional system prompt instructions. Uses default if None.
        backend: Optional DeepAgents backend for tool execution.
        skills: Optional list of skill identifiers to enable for the agent.

    Returns:
        A LangChain Runnable chain (prompt | agent | stream converter) that
        produces OpenAI-compatible SSE byte chunks.
    """
    if instructions is None:
        instructions = default_instructions

    @RunnableGenerator
    def stream_openai_response(chunks: Iterable[AIMessageChunk]) -> Iterable[bytes]:
        """Convert DeepAgents (or plain AIMessageChunk) output into OpenAI‑compatible SSE.

        DeepAgents may emit either ``AIMessageChunk`` objects or plain ``dict``
        structures.  We gracefully handle both by extracting ``content`` and an
        optional ``id``.  If an ``id`` is missing we generate a UUID so the client
        receives a well‑formed response.
        """
        for chunk in chunks:
            logger.debug("Streaming chunk: %s", chunk)
            chunk_id = get_chunkid(chunk)
            content = get_content(chunk)
            data = format_data(chunk_id, model, str(uuid4()), content)
            logger.debug("Data: %s", data)
            yield bytes(f"data: {data}\n\n", "utf-8")

    prompt = ChatPromptTemplate.from_template("Question: {input}")

    llm = ChatWillma(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        api_key=api_key,
    )

    agent = create_deep_agent(
        model=llm,
        tools=tools,
        system_prompt=instructions,
        backend=backend,
        skills=skills,
    )

    return (prompt | agent | stream_openai_response)
