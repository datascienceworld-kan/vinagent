from typing import Any
import importlib
import ast
import json
import os
import inspect
import uuid
from vinagent.config.logger_config import setup_logger

logger = setup_logger(__name__,"vinagent_analysis.log")

class ToolMetadataExtractor:
    def __init__(self, module_path: str, save_path: str = "templates"):
        self.module_path = module_path
        self.save_path = save_path
        logger.info(f"Initializing ToolMetadataExtractor with module_path='{module_path}', save_path='{save_path}'")
        try:
            self.module = importlib.import_module(module_path)
            self.module_file = self.module.__file__
            logger.debug(f"Imported module '{module_path}' from file '{self.module_file}'")
        except Exception as e:
            logger.error(f"Failed to import module '{module_path}': {e}")
            raise
        
        
        

    def extract_imports(self) -> list[str]:
        logger.info(f"Extracting imports from '{self.module_file}'")
        try:
            with open(self.module_file, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read())
            imports = set()

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module.split('.')[0])

            sorted_imports = list(sorted(imports))
            logger.debug(f"Found imports: {sorted_imports}")
            return sorted_imports

        except Exception as e:
            logger.error(f"Failed to extract imports: {e}")
            return []
        
            
    def extract_metadata(self) -> dict[str, Any]:
        logger.info(f"Extracting metadata from module '{self.module_path}'")
        tool_metadata = {}
        try:
            for name, func in inspect.getmembers(self.module, inspect.isfunction):
                if func.__module__ != self.module_path:
                    logger.debug(f"Skipping function '{name}' from module '{func.__module__}'")
                    continue

                if name.startswith("_"):
                    logger.debug(f"Skipping private/internal function '{name}'")
                    continue

                logger.debug(f"Processing function '{name}'")
                sig = inspect.signature(func)
                args = {}
                for param_name, param in sig.parameters.items():
                    arg_type = (
                        str(param.annotation)
                        if param.annotation != inspect._empty else "Any"
                    )

                    default = (
                        param.default if param.default != inspect._empty else None
                    )
                    args[param_name] = default if default is not None else arg_type

                metadata = {
                    "tool_name": name,
                    "arguments": args,
                    "return": (
                        str(sig.return_annotation)
                        if sig.return_annotation != inspect._empty else "Any"
                    ),
                    "docstring": func.__doc__.strip() if func.__doc__ else "",
                    "dependencies": self.extract_imports(),
                    "module_path": self.module_path,
                    "tool_call_id": f"tool_{uuid.uuid4()}"
                }

                tool_metadata[name] = metadata
                logger.debug(f"Extracted metadata for tool '{name}': {metadata}")
        except Exception as e:
           logger.error(f"Failed to extract metadata for {name}: {e}")
        return tool_metadata
    

    def export_to_json(self):
        logger.info(f"Exporting metadata to JSON in '{self.save_path}'")
        try:
            os.makedirs(self.save_path, exist_ok=True)
            data = self.extract_metadata()
            file_name = self.module_path.split(".")[-1] + ".json"
            full_path = os.path.join(self.save_path, file_name)
            with open(full_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            logger.info(f"Exported metadata to '{full_path}'")
        
        except Exception as e:
            logger.error(f"Failed to export metadata to JSON: {e}")