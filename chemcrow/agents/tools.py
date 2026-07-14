import os
from typing import Any, Optional, Type

from langchain import agents
from langchain.base_language import BaseLanguageModel
from langchain.tools import BaseTool
from overmind import tool
from pydantic import BaseModel

from chemcrow.tools import *


@tool("Name2SMILES")
def Name2SMILES(query: str, chemspace_api_key: str = None) -> str:
    return Query2SMILES(chemspace_api_key)._run(query)


class RobustTool(BaseTool):
    """Wraps another tool so it never crashes the agent loop.

    LangChain's ``handle_tool_error`` only catches ``ToolException``; raw
    exceptions (e.g. a flaky HTTP call raising ``JSONDecodeError``) otherwise
    propagate and abort the whole run. This wrapper turns any failure into an
    observation string the agent can react to, and coerces non-string results
    (e.g. a float molecular weight) into text.
    """

    inner_tool: Any = None

    def _run(self, *args: Any, **kwargs: Any) -> str:
        try:
            result = self.inner_tool._run(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 - deliberately broad for agent safety
            return (
                f"Error while running {self.name}: {type(e).__name__}: {e}. "
                "This may be a transient issue or invalid input. "
                "Consider trying a different tool, input, or approach."
            )
        if result is None:
            return f"{self.name} returned no result."
        return result if isinstance(result, str) else str(result)

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("Async not implemented.")


def _robustify(tool: BaseTool) -> RobustTool:
    """Wrap a tool instance, preserving its agent-facing metadata."""
    return RobustTool(
        name=tool.name,
        description=tool.description,
        inner_tool=tool,
        args_schema=getattr(tool, "args_schema", None),
        return_direct=getattr(tool, "return_direct", False),
    )


def _trace_langchain_tool(t: BaseTool) -> BaseTool:
    original_run = t._run

    @tool(t.name)
    def _traced(*args: Any, **kwargs: Any) -> str:
        return original_run(*args, **kwargs)

    t._run = _traced
    return t


def _safe_build(label: str, factory):
    """Build a tool, skipping it (with a warning) if construction fails.

    Some tools do network/setup work in ``__init__`` (e.g. downloading data).
    A failure there should disable that single tool, not the whole agent.
    """
    try:
        return factory()
    except Exception as e:  # noqa: BLE001
        print(f"[chemcrow] Skipping tool '{label}': {type(e).__name__}: {e}")
        return None


def make_tools(llm: BaseLanguageModel, api_keys: dict = {}, local_rxn: bool=False, verbose=True):
    serp_api_key = api_keys.get("SERP_API_KEY") or os.getenv("SERP_API_KEY")
    rxn4chem_api_key = api_keys.get("RXN4CHEM_API_KEY") or os.getenv("RXN4CHEM_API_KEY")
    openai_api_key = api_keys.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    chemspace_api_key = api_keys.get("CHEMSPACE_API_KEY") or os.getenv(
        "CHEMSPACE_API_KEY"
    )
    semantic_scholar_api_key = api_keys.get("SEMANTIC_SCHOLAR_API_KEY") or os.getenv(
        "SEMANTIC_SCHOLAR_API_KEY"
    )

    all_tools = [
        _trace_langchain_tool(t)
        for t in agents.load_tools(
            [
                "python_repl",
                # "ddg-search",
                "wikipedia",
                # "human"
            ]
        )
    ]

    core_tools = [
        _safe_build("Name2SMILES", lambda: Query2SMILES(chemspace_api_key)),
        _safe_build("Mol2CAS", lambda: Query2CAS()),
        _safe_build("SMILES2Name", lambda: SMILES2Name()),
        _safe_build("PatentCheck", lambda: PatentCheck()),
        _safe_build("MolSimilarity", lambda: MolSimilarity()),
        _safe_build("SMILES2Weight", lambda: SMILES2Weight()),
        _safe_build("FuncGroups", lambda: FuncGroups()),
        _safe_build("ExplosiveCheck", lambda: ExplosiveCheck()),
        _safe_build("ControlChemCheck", lambda: ControlChemCheck()),
        _safe_build("SimilarControlChemCheck", lambda: SimilarControlChemCheck()),
        _safe_build("SafetySummary", lambda: SafetySummary(llm=llm)),
        _safe_build(
            "LiteratureSearch",
            lambda: Scholar2ResultLLM(
                llm=llm,
                openai_api_key=openai_api_key,
                semantic_scholar_api_key=semantic_scholar_api_key,
            ),
        ),
    ]
    all_tools += [t for t in core_tools if t is not None]

    if chemspace_api_key:
        gmp = _safe_build("GetMoleculePrice", lambda: GetMoleculePrice(chemspace_api_key))
        if gmp is not None:
            all_tools += [gmp]
    if serp_api_key:
        ws = _safe_build("WebSearch", lambda: WebSearch(serp_api_key))
        if ws is not None:
            all_tools += [ws]
    if (not local_rxn) and rxn4chem_api_key:
        rxn_tools = [
            _safe_build("ReactionPredict", lambda: RXNPredict(rxn4chem_api_key)),
            _safe_build("ReactionRetrosynthesis", lambda: RXNRetrosynthesis(rxn4chem_api_key, llm)),
        ]
        all_tools += [t for t in rxn_tools if t is not None]
    elif local_rxn:
        rxn_tools = [
            _safe_build("ReactionPredict", lambda: RXNPredictLocal()),
            _safe_build("ReactionRetrosynthesis", lambda: RXNRetrosynthesisLocal(llm=llm)),
        ]
        all_tools += [t for t in rxn_tools if t is not None]

    return [_robustify(t) for t in all_tools]
