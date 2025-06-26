import os
from dotenv import load_dotenv
from typing import Any
from vinagent.config.logger_config import setup_logger
from app.core.setting import reset_settings_cache

logger = setup_logger(__name__, "vinagent.log")

class EnvConfigService:
    def __init__(self, env_file_path: str = ".env", auto_reload: bool = True):
        self.env_file_path = env_file_path
        self.auto_reload = auto_reload
        self._load_env()

    def _load_env(self):
        if os.path.exists(self.env_file_path):
            load_dotenv(self.env_file_path, override=True)
            logger.info(f"Loaded environment from {self.env_file_path}")
        else:
            logger.warning(f"{self.env_file_path} does not exist.")

    def _format_value(self, value: str) -> str:
        if isinstance(value, bool):
            value = "true" if value else "false"
        elif isinstance(value, (int, float)):
            value = str(value)
        elif isinstance(value, list):
            value = ",".join(map(str, value))
        elif not isinstance(value, str):
            raise ValueError(f"Unsupported value type: {type(value)}")

        if any(c in value for c in ' #"\n\t=,'):
            return f'"{value}"'
        return value

    def update_variable(self, key: str, value: Any):
        if not os.path.exists(self.env_file_path):
            raise FileNotFoundError(f"{self.env_file_path} not found.")

        formatted_value = self._format_value(value)

        with open(self.env_file_path, "r") as file:
            lines = file.readlines()

        was_updated = False
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            if stripped_line.startswith(f"{key}=") or stripped_line.startswith(f"export {key}="):
                prefix = "export " if stripped_line.startswith("export") else ""
                lines[i] = f"{prefix}{key}={formatted_value}\n"
                was_updated = True
                logger.info(f"Updated variable: {key}")
                break

        if not was_updated:
            lines.append(f"{key}={formatted_value}\n")
            logger.info(f"Added new variable: {key}")

        with open(self.env_file_path, "w") as file:
            file.writelines(lines)

        if self.auto_reload:
            self._load_env()
            reset_settings_cache() 

    def delete_variable(self, key: str):
        if not os.path.exists(self.env_file_path):
            raise FileNotFoundError(f"{self.env_file_path} not found.")

        with open(self.env_file_path, "r") as file:
            lines = file.readlines()

        new_lines = [
            line for line in lines
            if not line.strip().startswith(f"{key}=")
            and not line.strip().startswith(f"export {key}=")
        ]

        with open(self.env_file_path, "w") as file:
            file.writelines(new_lines)

        logger.info(f"Deleted variable: {key}")

        if self.auto_reload:
            self._load_env()
            reset_settings_cache() 

    def get_variable(self, key: str) -> str | None:
        value = os.getenv(key)
        if value is None:
            logger.warning(f"Variable {key} not found in environment.")
            return None
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            return value[1:-1]
        return value

    def get_variable_as_list(self, key: str) -> list[str]:
        raw = self.get_variable(key)
        if raw is None or raw.strip() == "":
            return []
        return [item.strip() for item in raw.split(",") if item.strip()]