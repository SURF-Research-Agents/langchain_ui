import os

from deepagents import create_deep_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_surf.chat_models.chat_willma import ChatWillma


class AgentWillma:

    def __init__(
        self,
        api_key=None,
        model="default-text-large",
        temperature=0.1,
        max_tokens=1000,
        timeout=30,
        tools=None,
        instructions=None,
        target="agents",
    ):
        """Initialize the Chain.

        Args:
            model: Identifier of the language model to use.
            api_key: API key for the model service.
            temperature: Sampling temperature for the model (default 0.1).
            max_tokens: Maximum tokens for model responses (default 1000).
            timeout: Request timeout in seconds (default 30).
            tools: Optional list of tools for the deep agent.
            instructions: System prompt instructions; defaults to a researcher prompt.
            target: Either 'agents' to build a deep‑agent chain or 'llm' for a plain LLM chain.
        """

        self.api_key = api_key
        if self.api_key is None:
            self.api_key = os.getenv("AIHUB_API_KEY")
        if self.api_key is None:
            raise ValueError("No API KEY FOUND")

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.target = target
        if tools is None:
            tools = []
        self.tools = tools
        if instructions is None:
            instructions = """You are an expert researcher. Your job is to conduct thorough research and then write a polished report.

You have access to an internet search tool as your primary means of gathering information.

## `internet_search`

Use this to run an internet search for a given query. You can specify the max number of results to return, the topic, and whether raw content should be included.
"""
        self.instructions = instructions

    def __call__(self):
        """Return the appropriate chain based on the target.

        Returns:
            A LangChain pipeline: either a deep‑agent chain or a plain LLM chain.
        """
        if self.target == "llm":
            return self.build_llm_chain()
        elif self.target == "agents":
            return self.build_deepagent_chain()
        else:
            raise ValueError("target must be llm or agents")

    def build_deepagent_chain(self):
        """Construct and return a deep‑agent chain.

        The chain consists of a prompt template followed by a deep agent created
        with the configured model, tools, and system instructions.
        """

        prompt = ChatPromptTemplate.from_template("Question: {input}")

        llm = ChatWillma(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            api_key=self.api_key,
        )
        agent = create_deep_agent(
            model=llm,
            tools=[self.tools],
            system_prompt=self.instructions,
        )
        return prompt | agent

    def build_llm_chain(self):
        """Construct and return a simple LLM chain.

        This chain consists of a prompt template followed directly by a ChatWillma
        language model configured with the provided parameters.
        """

        prompt = ChatPromptTemplate.from_template("Question: {input}")

        llm = ChatWillma(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            api_key=self.api_key,
        )

        return prompt | llm
