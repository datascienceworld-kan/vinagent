from pydantic import BaseModel
from typing import List

class AgentConfig(BaseModel):
    model: str
    tools: List[str]
    description: str
    skils: List[str]