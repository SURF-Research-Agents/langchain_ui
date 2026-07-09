import os
from dotenv import load_dotenv
from flask import Flask, Response, request


from langchain_ui.message import OpenAIRequest
from langchain_ui.agents.agent import create_willma_agent
from langchain_ui.tools.tavily import internet_search

load_dotenv('/Users/renau001/Documents/projects/ai/SRA/.env')

api_key = os.getenv("AIHUB_API_KEY")


app = Flask('WillmAgent')


willma_agent = create_willma_agent(api_key=api_key,
                                   tools=[internet_search])



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
