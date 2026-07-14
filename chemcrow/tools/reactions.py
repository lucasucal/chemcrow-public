"""Self-hosted reaction tools. Retrosynthesis, reaction forward prediction."""

import json
import os
from typing import Optional

import requests

RXN_PREDICT_URL = os.getenv("RXN_PREDICT_URL", "http://localhost:8051/api/v1/run")
RXN_RETRO_URL = os.getenv("RXN_RETRO_URL", "http://localhost:8052/api/v1/run")
from langchain.base_language import BaseLanguageModel
from langchain.schema import HumanMessage
from langchain.tools import BaseTool
from overmind import observe, tool

from chemcrow.utils import is_smiles

__all__ = ["RXNPredictLocal", "RXNRetrosynthesisLocal"]


class RXNPredictLocal(BaseTool):
    """Predict reaction."""

    name = "ReactionPredict"
    description = (
        "Predict the outcome of a chemical reaction. "
        "Takes as input the SMILES of the reactants separated by a dot '.', "
        "returns SMILES of the products."
    )

    @tool("ReactionPredict")
    def _run(self, reactants: str) -> str:
        """Run reaction prediction."""
        if not is_smiles(reactants):
            return "Incorrect input."

        product = self.predict_reaction(reactants)
        return product

    def predict_reaction(self, reactants: str) -> str:
        """Make api request."""
        try:
            response = requests.post(
                RXN_PREDICT_URL,
                headers={"Content-Type": "application/json"},
                data=json.dumps({"smiles": reactants})
            )
            return response.json()['product'][0]
        except:
            return "Error in prediction."


class RXNRetrosynthesisLocal(BaseTool):
    """Predict retrosynthesis."""

    name = "ReactionRetrosynthesis"
    description = (
        "Obtain the synthetic route to a chemical compound. "
        "Takes as input the SMILES of the product, returns recipe."
    )
    llm: Optional[BaseLanguageModel] = None

    def __init__(self, llm: Optional[BaseLanguageModel] = None):
        super().__init__()
        self.llm = llm

    @tool("ReactionRetrosynthesis")
    def _run(self, reactants: str) -> str:
        """Run reaction prediction."""
        if not is_smiles(reactants):
            return "Incorrect input."

        paths = self.retrosynthesis(reactants)
        procedure = self.get_action_sequence(paths[0])
        return procedure

    def retrosynthesis(self, reactants: str) -> str:
        """Make api request."""
        response = requests.post(
            RXN_RETRO_URL,
            headers={"Content-Type": "application/json"},
            data=json.dumps({"smiles": reactants})
        )
        return response.json()

    def get_action_sequence(self, path):
        """Get sequence of actions."""
        actions = path
        json_actions = self._preproc_actions(actions)
        llm_sum = self._summary_gpt(json_actions)
        return llm_sum

    def _preproc_actions(self, path):
        """Preprocess actions."""
        def _clean_actions(d):
            if 'metadata' in d:
                if 'mapped_reaction_smiles' in d['metadata']:
                    r = d['metadata']['mapped_reaction_smiles'].split(">>")
                    yield {"reactants": r[1], "products": r[0]}
            if 'children' in d:
                for c in d['children']:
                    yield from _clean_actions(c)

        rxns = list(_clean_actions(path))
        return rxns

    @observe("retrosynthesis_recipe_writer")
    def retrosynthesis_recipe_writer(self, json: dict) -> str:
        """Describe synthesis."""
        if self.llm is None:
            raise ValueError("LLM is required for retrosynthesis summarization.")
        prompt = (
            "Here is a chemical synthesis described as a json.\nYour task is "
            "to describe the synthesis, as if you were giving instructions for"
            "a recipe. Use only the substances, quantities, temperatures and "
            "in general any action mentioned in the json file. This is your "
            "only source of information, do not make up anything else. Also, "
            "add 15mL of DCM as a solvent in the first step. If you ever need "
            'to refer to the json file, refer to it as "(by) the tool". '
            "However avoid references to it. \nFor this task, give as many "
            f"details as possible.\n {str(json)}"
        )
        return self.llm([HumanMessage(content=prompt)]).content

    def _summary_gpt(self, json: dict) -> str:
        return self.retrosynthesis_recipe_writer(json)
