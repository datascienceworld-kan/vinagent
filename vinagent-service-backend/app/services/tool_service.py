from vinagent.config.logger_config import setup_logger
import vinagent.tools
import importlib.resources as pkg_resources
from typing import List

logger = setup_logger(__name__, "vinagent.log")

class ToolService:
    @staticmethod
    def discover_tools() -> List[str]:
        """Scans the vinagent.tools package and returns a list of available tool module names."""
        tools_list = []

        try:
            tools_dir = pkg_resources.files(vinagent.tools)
            logger.debug(f"Scanning tools directory: {tools_dir}")

            if not tools_dir.is_dir():
                logger.warning(f"The tools directory {tools_dir} is not a valid directory.")
                return tools_list
            
            for item in tools_dir.iterdir():
                if item.name.endswith(".py") and item.name != "__init__.py":
                    module_name = f"vinagent.tools.{item.name[:-3]}"
                    logger.debug(f"Discovered tool module: {module_name}")
                    tools_list.append(module_name)

        except ModuleNotFoundError:
            logger.error("vinagent.tools package not found. Ensure it is installed and accessible.")
        except Exception as e:
            logger.error(f"Failed to discover tools: {e}")
            
        logger.info(f"Total tool modules discovered: {len(tools_list)}")
        return tools_list
