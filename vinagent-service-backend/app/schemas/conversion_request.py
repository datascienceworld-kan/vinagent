from typing import Literal

from pydantic import BaseModel


class ConversionRequest(BaseModel):
    markdown: str
    format: Literal["pdf","docx"]