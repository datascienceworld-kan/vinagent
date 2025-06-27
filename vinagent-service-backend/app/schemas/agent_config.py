from pydantic import BaseModel
from typing import List
from pathlib import Path

class AgentConfig(BaseModel):
    model: str
    tools: List[str]
    description: str
    skils: List[str]
    tools_path: Path = Path("templates/tools.json")