
from typing import Iterable
from uuid import uuid4

from deepagents import create_deep_agent
from langchain_core.messages import AIMessageChunk
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableGenerator
from langchain_surf import ChatWillma

from langchain_ui.agents.agent_utils import format_data, get_chunkid, get_content
from langchain_ui.agents.default_agent_instruction import default_instructions


def create_willma_agent(
    api_key,
    model="default-text-large",
    temperature=0.1,
    max_tokens=100000,
    timeout=30,
    tools=None,
    instructions=None,
):
    """return a willma agent instance

    Args:
        api_key (_type_): _description_
        model (str, optional): _description_. Defaults to "default-text-large".
        temperature (float, optional): _description_. Defaults to 0.1.
        max_tokens (int, optional): _description_. Defaults to 100000.
        timeout (int, optional): _description_. Defaults to 30.
        tools (_type_, optional): _description_. Defaults to None.
        instructions (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
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
            # Debug output – keep it simple to avoid leaking large data.
            print("\nstreaming chunk:", chunk)
            chunk_id = get_chunkid(chunk)
            content = get_content(chunk)
            data = format_data(chunk_id, model, str(uuid4()), content)
            print("\ndata", data)
            yield bytes(f"data: {data}\n\n", "utf-8")

    def agent_wilma():
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
        )

        return prompt | agent | stream_openai_response

    return agent_wilma()
