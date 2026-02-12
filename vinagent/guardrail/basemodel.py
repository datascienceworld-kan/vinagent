from typing import Optional, Literal, List
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod

class GuardRailBase(BaseModel, ABC):
    name: str = Field(
        description="The name of guardrail"
    )
    reason: Optional[str] = Field(
        description="Reason for guardrail decision",
        default=None
    )

    @abstractmethod
    def prompt_section(self) -> str:
        """Instruction injected into the LLM prompt"""

    @abstractmethod
    def result_field(self) -> str:
        """Field name used in GuardrailDecision output"""

class OutputGuardRailBase(BaseModel, ABC):
    name: str = Field(
        description="The name of guardrail"
    )
    reason: Optional[str] = Field(
        description="Reason for guardrail decision",
        default=None
    )

    @abstractmethod
    def prompt_section(self) -> str:
        pass

    @abstractmethod
    def result_field(self) -> str:
        pass
