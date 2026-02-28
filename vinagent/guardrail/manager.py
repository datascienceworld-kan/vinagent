from typing import List, Union
import yaml
from langchain_together import ChatTogether
from langchain_core.language_models.base import BaseLanguageModel
from langchain_openai.chat_models.base import BaseChatOpenAI

from vinagent.guardrail import (
    AuthenticationGuardrail,
    GuardrailDecision,
    PIIGuardrail,
    ScopeGuardrail,
    ToxicityGuardrail,
    PromptInjectionGuardrail,
    OutputPIIGuardrail,
    OutputToxicityGuardrail,
    HallucinationGuardrail,
    OSPermissionGuardrail
)


class GuardrailManager:

    def __init__(self, yaml_path: str):
        self.yaml_path = yaml_path
        self.config = self._load_yaml()
        self.input_guardrails = []
        self.output_guardrails = []
        self.tool_guardrails = {}
        self._initialize_guardrails()

    def _load_yaml(self):
        with open(self.yaml_path, "r") as f:
            return yaml.safe_load(f)

    def _instantiate(self, item, tool_name: str | None = None):
        name = item["name"]
        params = item.get("params", {})
        if tool_name:
            params["name"] = tool_name

        try:
            cls = globals()[name]
        except KeyError:
            raise ValueError(f"Guardrail class {name} not found.")
        return cls(**params)

    def _initialize_guardrails(self):
        gr_config = self.config.get("guardrails", {})

        # Input
        for item in gr_config.get("input", []):
            self.input_guardrails.append(self._instantiate(item))

        # Output
        for item in gr_config.get("output", []):
            self.output_guardrails.append(self._instantiate(item))

        # Tools
        for tool_name, guardrails in gr_config.get("tools", {}).items():
            self.tool_guardrails[tool_name] = [
                self._instantiate(item, tool_name) for item in guardrails
            ]

    def add_guardrails(self, guardrails: List | None = None, **kwargs):
        DecisionModel = GuardrailDecision.add_guardrails(guardrails)
        return DecisionModel

    def validate_input(self, llm, user_input: str, **kwargs):
        DecisionModel = self.add_guardrails(self.input_guardrails)
        result = DecisionModel.validate(llm, user_input)
        return result

    def validate_tools(self, tool_name: str | None = None, **kwargs):
        def _validate(guardrail):
            if isinstance(guardrail, OSPermissionGuardrail):
                llm = kwargs.get("llm")
                user_input = kwargs.get("user_input")
                if llm is None or user_input is None:
                    missing = []
                    if llm is None:
                        missing.append("llm")
                    if user_input is None:
                        missing.append("user_input")
                    raise ValueError(
                        f"Missing required argument(s) for OSPermissionGuardrail: "
                        f"{', '.join(missing)}. "
                        f"Please call validate_tools(..., llm=..., user_input=...)"
                    )

                return guardrail.validate(
                    llm=kwargs.get("llm"),
                    user_input=kwargs.get("user_input"),
                )
            return guardrail.validate(**kwargs)
        
        if tool_name:
            return [_validate(g) for g in self.tool_guardrails.get(tool_name, [])]

        return {
            name: [_validate(g) for g in guardrails]
            for name, guardrails in self.tool_guardrails.items()
        }


    def validate_output(self, llm, output_text: str, **kwargs):
        DecisionModel = self.add_guardrails(self.output_guardrails)
        result = DecisionModel.validate(llm, output_text)
        return result
