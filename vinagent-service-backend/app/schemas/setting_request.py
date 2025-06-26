from typing import List

from pydantic import BaseModel


class SettingRequest(BaseModel):
    model_id: str
    description: str
    skills: List[str]