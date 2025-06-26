from typing import List, Any, Dict

from pydantic import BaseModel


class ModelSetting(BaseModel):
    models: List[Dict[str, Any]]
    agent_description: str
    agent_skills: List[str]