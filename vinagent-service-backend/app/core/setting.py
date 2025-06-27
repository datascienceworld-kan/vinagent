import os
from typing import List
from dotenv import load_dotenv
from functools import lru_cache

load_dotenv()
class Settings:
    def __init__(self):
        self.llm_model: str = os.getenv("LLM_MODEL", "")
        self.agent_description: str = os.getenv("AGENT_DESCRIPTION", "")
        self.agent_skills: List[str] = self._parse_skills(
            os.getenv("AGENT_SKILLS", "")
        )
        self.textbin_path: str = os.getenv("TEXBIN_PATH","")
        self.frontend_origin: str = os.getenv("FRONTEND_ORIGIN","http://localhost:3000")
        self.tools_path: str = os.getenv("TOOLS_PATH","templates/tools.json")
        self.is_reset_tools = _parse_bool(os.getenv("IS_RESET_TOOLS", "False"))

    def _parse_skills(self, skills_str: str) -> List[str]:
        return [s.strip() for s in skills_str.split(",") if s.strip()]
    

@lru_cache()
def get_settings() -> Settings:
    return Settings()

def reset_settings_cache():
    get_settings.cache_clear()
    load_dotenv(override=True)

def _parse_bool(value: str)->bool:
    return value.lower() in ["1", "true", "yes"]