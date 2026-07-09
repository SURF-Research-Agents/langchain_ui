import json
import time
from uuid import uuid4

from langchain_core.messages import AIMessageChunk


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


def get_content(chunk: AIMessageChunk) -> str:
    """_summary_

    Args:
        chunk (AIMessageChunk): _description_

    Returns:
        str: _description_

    Yields:
        Iterator[str]: _description_
    """
    if isinstance(chunk, dict):
        try:
            content = chunk["model"]["messages"][-1].content
        except KeyError:
            content = ""
    else:
        content = getattr(chunk, "content", "")

    return content


def get_chunkid(chunk: AIMessageChunk) -> str:
    """_summary_

    Args:
        chunk (AIMessageChunk): _description_

    Returns:
        str: _description_
    """
    if isinstance(chunk, dict):
        chunk_id = chunk.get("id", str(uuid4()))
    else:
        chunk_id = getattr(chunk, "id", str(uuid4()))
    return chunk_id
