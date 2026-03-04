import os
from typing import Literal, Optional
from pydantic import BaseModel
import logging
from vinagent.guardrail.basemodel import GuardRailBase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FileIntent(BaseModel):
    file_path: Optional[str] = None
    action: Optional[Literal["read", "write", "execute"]] = None


class OSPermissionGuardrailResult(BaseModel):
    allowed: bool
    file_path: Optional[str] = None
    permission_type: Optional[str] = None
    reason: Optional[str] = None

    def __str__(self) -> str:
        return (
            "OSPermissionGuardrailResult("
            f"allowed={self.allowed}, "
            f"file_path={self.file_path}, "
            f"permission_type={self.permission_type}, "
            f"reason={self.reason})"
        )


class OSPermissionGuardrail(GuardRailBase):
    file_name: Optional[str] = None
    action: Optional[Literal["read", "write", "execute"]] = None

    name: str = "os_permission"

    # ------------------------------------
    # 1. LLM Intent Extraction
    # ------------------------------------
    def intent_extraction_prompt(self, user_input: str):
        return f"""
You are a system security parser.
Extract file operation intent from the user request.
Return JSON ONLY with the following schema:
{{
"file_path": string | null,
"action": "read" | "write" | "execute" | null
}}

Rules:
- If user wants to view, list, open → action = "read"
- If user wants to modify, delete, create → action = "write"
- If user wants to run a file → action = "execute"
- If no file path is found → file_path = null
- If no operation found → action = null

User input:
{user_input}
"""

    def _extract_intent(self, llm, user_input: str) -> FileIntent:
        if self.file_name and self.action:
            logger.info(
                f"Using hardcoded file_name: {self.file_name} and action: {self.action}"
            )
            return FileIntent(file_path=self.file_name, action=self.action)
        prompt = self.intent_extraction_prompt(user_input)
        intent_llm = llm.with_structured_output(FileIntent)
        # Replace with your LLM call
        llm_response = intent_llm.invoke(prompt)
        # Expecting strict JSON
        return llm_response

    # ------------------------------------
    # 2. Deterministic Permission Check
    # ------------------------------------
    def _validate_permission(
        self, path: str, permission_type: str
    ) -> OSPermissionGuardrailResult:
        permission_map = {
            "read": os.R_OK,
            "write": os.W_OK,
            "execute": os.X_OK,
        }

        perm_flag = permission_map[permission_type]

        exists = os.path.exists(path)
        has_permission = os.access(path, perm_flag) if exists else False

        allowed = exists and has_permission

        if allowed:
            return OSPermissionGuardrailResult(
                allowed=True,
                file_path=path,
                permission_type=permission_type,
                reason=f"Permission '{permission_type}' granted.",
            )

        if not exists:
            return OSPermissionGuardrailResult(
                allowed=False,
                file_path=path,
                permission_type=permission_type,
                reason="Path does not exist.",
            )

        return OSPermissionGuardrailResult(
            allowed=False,
            file_path=path,
            permission_type=permission_type,
            reason=f"Permission '{permission_type}' denied.",
        )

    # ------------------------------------
    # 3. Public Validate
    # ------------------------------------
    def validate(self, llm, user_input: str, **kwargs) -> OSPermissionGuardrailResult:
        try:
            intent = self._extract_intent(llm, user_input)

            if not intent.file_path or not intent.action:
                return OSPermissionGuardrailResult(
                    allowed=False,
                    file_path=intent.file_path,
                    permission_type=intent.action,
                    reason="Could not extract file operation intent.",
                )
            validate_permission = self._validate_permission(
                intent.file_path, intent.action
            )
            return validate_permission

        except Exception as e:
            return OSPermissionGuardrailResult(
                allowed=False,
                reason=f"Guardrail execution failed: {str(e)}",
            )

    def prompt_section(self, llm, user_input: str) -> str:
        validate_result = self.validate(llm=llm, user_input=user_input)
        return f"""
OS PERMISSION CHECK
Determine whether OS permission violates based on validation result:
- OSPermissionGuardrailResult(allowed=False): Violate because OS permission is denied or path does not exist.
- OSPermissionGuardrailResult(allowed=True): Not violate because OS permission is granted.
Validation result:
{validate_result}
"""

    def result_field(self) -> str:
        return "os_permission"
