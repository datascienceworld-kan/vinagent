from app.services.tool_service import ToolService

class VinagentController:
    
    def __init__(self):
        self.discover_tools = ToolService.discover_tools

    def list_available_tools(self):
        return self.discover_tools