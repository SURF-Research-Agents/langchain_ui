import logging
import os
from dotenv import load_dotenv
from flask import Flask, Response, request
from werkzeug.exceptions import BadRequest

from langfuse import get_client
from langfuse.langchain import CallbackHandler

from langchain_ui.message import OpenAIRequest
from langchain_ui.agents.agent import create_willma_agent
from langchain_ui.tools.tavily import internet_search
from langchain_ui.tools.slurm_calculator import slurm_calculator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
api_key = os.getenv("AIHUB_API_KEY")

langfuse = get_client()
langfuse_handler = CallbackHandler()

instructions = """
You are a helpful assistant. Be concise and accurate.

You have access to an internet search tool as your primary means of gathering information.
## `internet_search`
Use this to run an internet search for a given query. You can specify the max number of results to return, the topic, and whether raw content should be included.

You have access to a calculator tool to compute simple expression
##`slurm_calculator`
Always use the tool slurm_calculator to evaluate simple mathematical expressions
"""

app = Flask('WillmAgent')

willma_agent = create_willma_agent(api_key=api_key,
                                   tools=[internet_search,
                                          slurm_calculator],
                                    instructions=instructions)


@app.route("/chat/completions", methods=["POST"])
def chat():
    try:
        payload = request.get_json(force=True, silent=True)
        if not payload:
            raise BadRequest("Missing JSON body")
        
        chat_request = OpenAIRequest(**payload)
        question = chat_request.messages[-1].content
        
        if not question:
            raise BadRequest("Empty question")
    except (BadRequest, TypeError, KeyError) as e:
        return Response(
            f'data: {{"error": "{str(e)}"}}\n\n',
            mimetype="text/event-stream",
            status=400
        )

    def generate():
        try:
            for chunk in willma_agent.stream(question, 
                                             config={"callbacks": [langfuse_handler]}):
                yield chunk
        except Exception as e:
            logger.exception("Streaming error")
            yield f'data: {{"error": "{str(e)}"}}\n\n'

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
        direct_passthrough=True,
    )

if __name__ == "__main__":
    app.run(debug=True, port=8000)
