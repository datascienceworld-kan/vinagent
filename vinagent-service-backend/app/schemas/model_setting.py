from typing import List, Any, Dict
from pathlib import Path
from pydantic import BaseModel


class ModelSetting(BaseModel):
    models: List[Dict[str, Any]]
    agent_description: str
    agent_skills: List[str]
    tools_path: Path = Path("templates/tools.json")
    is_reset_tools: bool=False