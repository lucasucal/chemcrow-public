from typing import Optional

from dotenv import load_dotenv
from langchain import PromptTemplate, chains
from pydantic import ValidationError
from rmrkl import ChatZeroShotAgent, RetryAgentExecutor

from chemcrow.llm import DEFAULT_MODEL, detect_provider, get_api_key, make_llm

from .prompts import FORMAT_INSTRUCTIONS, QUESTION_PROMPT, REPHRASE_TEMPLATE, SUFFIX
from .tools import make_tools


def _key_for_model(
    model: str,
    *,
    openai_api_key: Optional[str] = None,
    anthropic_api_key: Optional[str] = None,
    openrouter_api_key: Optional[str] = None,
) -> Optional[str]:
    provider = detect_provider(model)
    if provider == "anthropic":
        return get_api_key(provider, anthropic_api_key)
    if provider == "openrouter":
        return get_api_key(provider, openrouter_api_key)
    return get_api_key(provider, openai_api_key)


class ChemCrow:
    def __init__(
        self,
        tools=None,
        model=DEFAULT_MODEL,
        tools_model=DEFAULT_MODEL,
        temp=0.1,
        max_iterations=40,
        verbose=True,
        streaming: bool = False,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
        api_keys: dict = {},
        local_rxn: bool = True,
    ):
        """Initialize ChemCrow agent."""

        load_dotenv()
        model_api_key = _key_for_model(
            model,
            openai_api_key=openai_api_key,
            anthropic_api_key=anthropic_api_key,
            openrouter_api_key=openrouter_api_key,
        )
        try:
            self.llm = make_llm(model, temp, model_api_key, streaming)
        except ValidationError:
            raise ValueError("Invalid API key for the selected model provider")

        if tools is None:
            api_keys["OPENAI_API_KEY"] = openai_api_key
            api_keys["ANTHROPIC_API_KEY"] = anthropic_api_key
            api_keys["OPENROUTER_API_KEY"] = openrouter_api_key
            tools_api_key = _key_for_model(
                tools_model,
                openai_api_key=openai_api_key,
                anthropic_api_key=anthropic_api_key,
                openrouter_api_key=openrouter_api_key,
            )
            tools_llm = make_llm(tools_model, temp, tools_api_key, streaming)
            tools = make_tools(tools_llm, api_keys=api_keys, local_rxn=local_rxn, verbose=verbose)

        # Initialize agent
        self.agent_executor = RetryAgentExecutor.from_agent_and_tools(
            tools=tools,
            agent=ChatZeroShotAgent.from_llm_and_tools(
                self.llm,
                tools,
                suffix=SUFFIX,
                format_instructions=FORMAT_INSTRUCTIONS,
                question_prompt=QUESTION_PROMPT,
            ),
            verbose=True,
            max_iterations=max_iterations,
        )

        rephrase = PromptTemplate(
            input_variables=["question", "agent_ans"], template=REPHRASE_TEMPLATE
        )

        self.rephrase_chain = chains.LLMChain(prompt=rephrase, llm=self.llm)

    def run(self, prompt):
        outputs = self.agent_executor({"input": prompt})
        return outputs["output"]
