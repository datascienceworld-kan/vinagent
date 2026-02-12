from typing import Optional
from pydantic import BaseModel
from vinagent.guardrail.core import GuardRailBase


class AuthenticationGuardrailResult(BaseModel):
    allowed: bool
    reason: Optional[str] = None

    def __str__(self) -> str:
        return f"AuthenticationGuardrailResult(allowed={self.allowed}, reason={self.reason})"


from vinagent.oauth2.client import AuthenCard


class AuthenticationGuardrail(GuardRailBase):
    name: str = "authentication"
    secret_path: str | None = None
    access_token: str | None = None
    api_url: str | None = None

    def _build_auth_card(self) -> AuthenCard:
        if self.secret_path:
            return AuthenCard.from_config(self.secret_path)
        return AuthenCard(token=self.access_token, api_url=self.api_url)

    def validate(self, **kwargs) -> AuthenticationGuardrailResult:
        try:
            auth_card = self._build_auth_card()
            is_valid = auth_card.verify_access_token()
            if is_valid:
                return AuthenticationGuardrailResult(
                    allowed=True, reason="Valid access token."
                )

            return AuthenticationGuardrailResult(
                allowed=False, reason=f"Authentication failed"
            )

        except Exception as e:
            return AuthenticationGuardrailResult(
                allowed=False, reason=f"Authentication failed: {str(e)}"
            )

    def prompt_section(self) -> str:
        authen_result = self.validate()
        return f"""
AUTHENTICATION CHECK
Determine whether authentication violates based on authentication result:
- AuthenticationGuardrailResult(allowed=False): Violate because the access token is invalid.
- AuthenticationGuardrailResult(allowed=True): Not violate because the access token is valid.
Authentication result:
{authen_result}
"""

    def result_field(self) -> str:
        return "authentication"
