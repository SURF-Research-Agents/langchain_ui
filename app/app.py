import os
from typing import Literal

from dotenv import load_dotenv
from flask import Flask, Response, request
from tavily import TavilyClient

from langchain_ui.chain import AgentWillma
from langchain_ui.message import OpenAIRequest
from langchain_ui.stream import OpenAIStream

load_dotenv("/Users/renau001/Documents/projects/ai/SRA/.env")

api_key = os.getenv("AIHUB_API_KEY")

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

app = Flask('AgentWillma')


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


openai_stream = OpenAIStream(target='llm')
willma_agent = AgentWillma(api_key=api_key, tools=[internet_search], target='llm')
chain = willma_agent() | openai_stream


@app.route("/chat/completions", methods=["POST"])
def chat():
    payload = request.get_json(force=True, silent=True) or {}

    chat_request = OpenAIRequest(**payload)

    question = chat_request.messages[-1].content

    return Response(chain.stream(question), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(debug=True, port=8000)
