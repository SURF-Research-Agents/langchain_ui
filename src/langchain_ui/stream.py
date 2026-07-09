import json
import time
from typing import Iterable
from uuid import uuid4

from langchain_core.messages import AIMessageChunk
from langchain_core.runnables import RunnableGenerator


class OpenAIStream:
    """Stream generator that formats AIMessageChunk outputs into OpenAI-compatible Server‑Sent Events.

    The class selects appropriate content and chunk‑id extraction methods based on the target
    ('agents' for DeepAgents or 'llm' for plain language model outputs) and provides a callable
    interface returning an iterable of byte chunks suitable for HTTP streaming.
    """

    def __init__(
        self, model="default-text-large", system_fingerprint=None, target="agents"
    ):
        """Create an OpenAIStream.

        Args:
            model: The model identifier used for the response.
            system_fingerprint: Unique fingerprint for the system (passed to OpenAI format).
            target: Determines extraction logic – ``'agents'`` for DeepAgents output or ``'llm'`` for plain language model output.
        """
        self.model = model
        self.system_fingerprint = system_fingerprint
        if self.system_fingerprint is None:
            self.system_fingerprint = str(uuid4())

        if target == "agents":
            self.get_content = self.get_content_deepagents
            self.get_chunkid = self.get_chunkid_deepagents
        elif target == "llm":
            self.get_content = self.get_content_llm
            self.get_chunkid = self.get_chunkid_llm

    def __call__(self, chunks: Iterable[AIMessageChunk]) -> Iterable[bytes]:
        """Make the instance callable, delegating to ``stream_openai_response``.

        Args:
            chunks: An iterable of ``AIMessageChunk`` objects (or dicts) produced by the chain.

        Returns:
            An iterable of byte strings formatted as SSE data blocks.
        """
        return self.stream_openai_response(chunks)

    @staticmethod
    def format_data(chunk_id, model, system_fingerprint, content):
        """Format a single SSE data payload.

        Args:
            chunk_id: Identifier for the chunk (UUID if not provided).
            model: Model name to include in the payload.
            system_fingerprint: System fingerprint string.
            content: Text content of the chunk.

        Returns:
            JSON string representing the OpenAI SSE chunk.
        """
        return json.dumps(
            {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "system_fingerprint": system_fingerprint,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": content},
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
            }
        )

    @staticmethod
    def get_content_deepagents(chunk):
        """Extract content from a DeepAgents chunk.

        Handles both dict representations (where the content resides in ``chunk['model']['messages']``)
        and object representations (reading the ``content`` attribute). Returns an empty string
        if the expected structure is missing.
        """
        if isinstance(chunk, dict):
            try:
                content = chunk["model"]["messages"][-1].content
            except KeyError:
                content = ""
        else:
            content = getattr(chunk, "content", "")
        return content

    @staticmethod
    def get_content_llm(chunk):
        """Extract the content from a LLM chunck

        Args:
            chunk (_type_): _description_

        Returns:
            _type_: _description_
        """
        return chunk.content

    @staticmethod
    def get_chunkid_deepagents(chunk):
        """Extrat the ID from a agent chunk

        Args:
            chunk (_type_): _description_

        Returns:
            _type_: _description_
        """
        if isinstance(chunk, dict):
            chunk_id = chunk.get("id", str(uuid4()))
        else:
            chunk_id = getattr(chunk, "id", str(uuid4()))
        return chunk_id

    @staticmethod
    def get_chunkid_llm(chunk):
        """Extracts the id from a LLM chunk

        Args:
            chunk (_type_): _description_

        Returns:
            _type_: _description_
        """
        return chunk.id

    @RunnableGenerator
    def stream_openai_response(
        self, chunks: Iterable[AIMessageChunk]
    ) -> Iterable[bytes]:
        """Convert DeepAgents (or plain AIMessageChunk) output into OpenAI‑compatible SSE.

        DeepAgents may emit either ``AIMessageChunk`` objects or plain ``dict``
        structures.  We gracefully handle both by extracting ``content`` and an
        optional ``id``.  If an ``id`` is missing we generate a UUID so the client
        receives a well‑formed response.
        """
        for chunk in chunks:
            # Debug output – keep it simple to avoid leaking large data.
            # Support both object and dict representations.
            chunk_id = self.get_chunkid(chunk)
            content = self.get_content(chunk)
            data = self.format_data(
                chunk_id, self.model, self.system_fingerprint, content
            )
            # print('\ndata', data)
            yield bytes(f"data: {data}\n\n", "utf-8")
