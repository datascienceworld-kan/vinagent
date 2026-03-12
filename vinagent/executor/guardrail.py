from abc import ABC, abstractmethod
from typing import Optional
from langchain_core.language_models.base import BaseLanguageModel
from vinagent.logger.logger import logger
from vinagent.guardrail import GuardrailManager, GuardRailBase


class GuardrailExecutorBase(ABC):
    @abstractmethod
    def check_input_guardrail(self, query: str) -> bool:
        pass

    @abstractmethod
    def check_output_guardrail(self, query: str) -> bool:
        pass

    @abstractmethod
    def check_tool_guardrail(self, query: str) -> bool:
        pass


class GuardrailExecutor(GuardrailExecutorBase):
    def __init__(
        self,
        guardrail_manager: GuardrailManager,
        input_guardrail: Optional[GuardRailBase] = None,
        output_guardrail: Optional[GuardRailBase] = None,
        llm: BaseLanguageModel = None,
    ):
        self.guardrail_manager = guardrail_manager
        self.input_guardrail = input_guardrail
        self.output_guardrail = output_guardrail
        self.llm = llm

    def check_input_guardrail(self, query: str):
        if self.guardrail_manager:
            decision = self.guardrail_manager.validate_input(self.llm, query)
            if not decision.allowed:
                logger.error(f"Input is not allowed: {decision.reason}")
                raise ValueError(decision.reason)
            return False
        else:
            if self.input_guardrail:
                decision = self.input_guardrail.validate(self.llm, query)
                if not decision.allowed:
                    raise ValueError(decision.reason)
                return False
        return True

    def check_output_guardrail(self, output_text: str):
        if self.guardrail_manager:
            decision = self.guardrail_manager.validate_output(self.llm, output_text)
            if not decision.allowed:
                logger.error(f"Output is not allowed: {decision.reason}")
                raise ValueError(decision.reason)
            return False
        else:
            if self.output_guardrail:
                decision = self.output_guardrail.validate(self.llm, output_text)
                logger.info(decision)
                if not decision.allowed:
                    logger.error(f"Output is not allowed: {decision.reason}")
                    raise ValueError(decision.reason)
                return False
        return True

    def check_tool_guardrail(self, llm, tool_name: str, user_input: str):
        if self.guardrail_manager:
            decisions = self.guardrail_manager.validate_tools(
                llm=self.llm, tool_name=tool_name, user_input=user_input
            )
            for decision in decisions:
                if not decision.allowed:
                    logger.error(f"Tool {tool_name} is not allowed: {decision.reason}")
                    raise ValueError(decision.reason)
                return False
        return True
