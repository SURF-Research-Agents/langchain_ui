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
    """Extract all relevant information from an AIMessageChunk as a string.

    Captures content text, tool_calls, usage_metadata, and response_metadata.

    Args:
        chunk: AIMessageChunk (LangChain object or dict).

    Returns:
        str: All chunk information serialized into a single string.
    """
    parts = []

    if isinstance(chunk, dict):
        try:
            messages = chunk.get("model", {}).get("messages", [])
            msg = messages[-1] if messages else {}

            # Text content
            text = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "") or ""
            if text:
                parts.append("\n \n" + text)

            # Tool calls
            tool_calls = msg.get("tool_calls", []) if isinstance(msg, dict) else getattr(msg, "tool_calls", []) or []
            if tool_calls:
                for tc in tool_calls:
                    tc_str = f"\n🔧 *calling tool : ```{tc.get('name', 'unknown')}```: \n \t{json.dumps(tc.get('args', {}))}* \n"
                    parts.append(tc_str)

            # Usage metadata
            usage = chunk.get("usage_metadata")
            if usage:
                parts.append(f"usage: {json.dumps(usage)}")

            # Response metadata
            resp_meta = chunk.get("response_metadata")
            if resp_meta:
                parts.append(f"response_metadata: {json.dumps(resp_meta)}")

            # Tool outputs
            tool_outputs = chunk.get("tools")
            
            if tool_outputs:
                tool_outputs = tool_outputs['messages']
                for tool_out in tool_outputs:
                    if tool_out.name == 'read_file':
                        out_str = f"\t*📄 File read*\n"
                    else:
                        out_str = f"\t*📤 tool output: {json.dumps(tool_out.content)}*\n"
                    parts.append(out_str)

        except (KeyError, IndexError, AttributeError):
            pass
    else:
        # Text content
        text = getattr(chunk, "content", "") or ""
        if text:
            parts.append(text)

        # Tool calls
        tool_calls = getattr(chunk, "tool_calls", []) or []
        if tool_calls:
            for tc in tool_calls:
                tc_str = f"🔧 calling tool {tc.get('name', 'unknown')}: {json.dumps(tc.get('args', {}))} \n"
                parts.append(tc_str)

        # Usage metadata
        usage = getattr(chunk, "usage_metadata", None)
        if usage:
            parts.append(f"usage: {json.dumps(usage)}")

        # Response metadata
        resp_meta = getattr(chunk, "response_metadata", None)
        if resp_meta:
            parts.append(f"response_metadata: {json.dumps(resp_meta)}")

        # Tool outputs
        tool_outputs = getattr(chunk, "tools", None) or []
        if tool_outputs:
            for tool_out in tool_outputs:
                out_str = f"📤 tool output: {json.dumps(tool_out)}"
                parts.append(out_str)

    return "\n".join(parts)

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


