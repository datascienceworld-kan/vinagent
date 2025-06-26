
from app.core.setting import get_settings
from app.schemas.agent_config import AgentConfig

def get_config():
    config = get_settings()

    agent_description = config.agent_description
    agent_skills = config.agent_skills
    llm_model = config.llm_model
    return AgentConfig(model=llm_model,tools=[],description=agent_description, skils=agent_skills)
