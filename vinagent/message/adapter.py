from langchain_core.messages.base import BaseMessage


def adapter_ai_response_with_tool_calls(
    all_tools: dict, response: BaseMessage, tool_call: dict
):
    adapt_tool = {}
    selected_tool = all_tools[tool_call["tool_name"]]
    adapt_tool["name"] = selected_tool["tool_name"]
    adapt_tool["args"] = tool_call["arguments"]
    adapt_tool["type"] = "tool_call"
    adapt_tool["id"] = selected_tool["tool_call_id"]
    response.tool_calls = [adapt_tool]
    return response
