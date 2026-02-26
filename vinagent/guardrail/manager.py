from typing import List
import yaml
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
        if tool_name:
            authen_guardrail = self.tool_guardrails[tool_name]
            result = authen_guardrail[0].validate()
            return result

        result = {}
        for tool_name, authen_list in self.tool_guardrails.items():
            result[tool_name] = authen_list[0].validate()
        return result

    def validate_output(self, llm, output_text: str, **kwargs):
        DecisionModel = self.add_guardrails(self.output_guardrails)
        result = DecisionModel.validate(llm, output_text)
        return result
