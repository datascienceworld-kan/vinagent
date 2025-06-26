from app.core.setting import get_settings
from app.schemas.model_setting import ModelSetting

class ModelService:
    @staticmethod
    def filter_models(models: list[dict]) -> dict:
        settings = get_settings()

        filtered_models = []
        for model in models:
            pricing = model.get("pricing", {})

            filtered_model = {
                "id": model.get("id", ""),
                "type": model.get("type", ""),
                "display_name": model.get("display_name", ""),
                "organization": model.get("organization", ""),
                "pricing": {
                    "input": pricing.get("input", 0.0),
                    "output": pricing.get("output", 0.0),
                    "base": pricing.get("base", 0.0),
                },
                "running": model.get("id") == settings.llm_model,
            }

            filtered_models.append(filtered_model)
            filtered_models.sort(key=lambda m: not m["running"])

        return ModelSetting(models=filtered_models, 
                            agent_description=settings.agent_description, 
                            agent_skills=settings.agent_skills)