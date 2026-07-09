import json
import os
import time
from typing import Iterable, Literal
from uuid import uuid4

from deepagents import create_deep_agent
from dotenv import load_dotenv
from flask import Flask, Response, request
from langchain_core.messages import AIMessageChunk
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableGenerator
from langchain_surf import ChatWillma
from pydantic import BaseModel
from tavily import TavilyClient

load_dotenv('/Users/renau001/Documents/projects/ai/SRA/.env')

api_key = os.getenv("AIHUB_API_KEY")
model = "default-text-large"
tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

api_key = os.getenv("AIHUB_API_KEY")

system_fingerprint = str(uuid4())
app = Flask(__name__)


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class OpenAIRequest(BaseModel):
    model: str
    user: str
    stream: bool
    messages: list[Message]


def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search"""
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )

# System prompt to steer the agent to be an expert researcher
research_instructions = """You are an expert researcher. Your job is to conduct thorough research and then write a polished report.

You have access to an internet search tool as your primary means of gathering information.

## `internet_search`

Use this to run an internet search for a given query. You can specify the max number of results to return, the topic, and whether raw content should be included.
"""


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
        print('\nstreaming chunk:', chunk)
        # Support both object and dict representations.
        if isinstance(chunk, dict):
            # content = chunk.get("content", "")
            try:
                content = chunk['model']['messages'][-1].content
            except KeyError:
                content = ""
            chunk_id = chunk.get("id", str(uuid4()))
        else:
            content = getattr(chunk, "content", "")
            chunk_id = getattr(chunk, "id", str(uuid4()))
        data = json.dumps({
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "system_fingerprint": system_fingerprint,
            "choices": [{
                "index": 0,
                "delta": {"content": content},
                "logprobs": None,
                "finish_reason": None,
            }],
        })
        print('\ndata', data)
        yield bytes(f"data: {data}\n\n", "utf-8")

def build_chain():
    prompt = ChatPromptTemplate.from_template("Question: {input}")
    llm = ChatWillma(
        model=model,
        temperature=0.1,
        max_tokens=100000,
        timeout=30,
        api_key=api_key,
    )
    agent = create_deep_agent(
        model=llm,
        tools=[internet_search],
        system_prompt=research_instructions,
    )
    return prompt | agent

chain = build_chain() | stream_openai_response

@app.route("/chat/completions", methods=["POST"])
def chat():
    payload = request.get_json(force=True, silent=True) or {}
    print('\nrequest:', payload)
    chat_request = OpenAIRequest(**payload)
    print('\nchat_request:', chat_request)
    question = chat_request.messages[-1].content
    print('\nquestion:', question)
    return Response(chain.stream(question), mimetype="text/event-stream")

if __name__ == "__main__":
    app.run(debug=True, port=8000)
