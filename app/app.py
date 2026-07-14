import os
from dotenv import load_dotenv
from flask import Flask, Response, request


from langchain_ui.message import OpenAIRequest
from langchain_ui.agents.agent import create_willma_agent
from langchain_ui.tools.tavily import internet_search
from langchain_ui.tools.slurm_calculator import slurm_calculator

load_dotenv('/Users/renau001/Documents/projects/ai/SRA/.env')
api_key = os.getenv("AIHUB_API_KEY")


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
    payload = request.get_json(force=True, silent=True) or {}
    print('\nrequest:', payload)
    chat_request = OpenAIRequest(**payload)
    print('\nchat_request:', chat_request)
    question = chat_request.messages[-1].content
    print('\nquestion:', question)
    return Response(willma_agent.stream(question), mimetype="text/event-stream")

if __name__ == "__main__":
    app.run(debug=True, port=8000)
