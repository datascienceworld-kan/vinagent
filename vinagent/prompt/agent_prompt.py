from typing import Union, List
from langchain_core.messages import (
    SystemMessage,
    ToolMessage,
    AIMessage,
    HumanMessage,
    BaseMessage,
)
import json


class PromptHandler:
    def format_tools_as_xml(self, tools: dict) -> str:
        parts = ["<tools>"]
        for name, tool in tools.items():
            parts.append(f'  <tool name="{name}">')
            parts.append(f'    <tool_name>{tool["tool_name"]}</tool_name>')
            parts.append(f'    <tool_type>{tool["tool_type"]}</tool_type>')
            parts.append(f'    <module_path>{tool["module_path"]}</module_path>')
            parts.append(f'    <is_runtime>{tool["is_runtime"]}</is_runtime>')
            parts.append(f'    <arguments>{json.dumps(tool["arguments"])}</arguments>')
            parts.append(f'    <return_type>{tool["return"]}</return_type>')
            parts.append(f'    <docstring>\n{tool["docstring"]}\n    </docstring>')
            parts.append(f"  </tool>")
        parts.append("</tools>")
        return "\n".join(parts)

    def build_prompt(
        self,
        user_id: str,
        message: str,
        tools: dict,
        memory: str = "",
    ) -> str:
        prompt = f"""You are a smart assistant that can answer questions and use tools to complete tasks.

## User
{user_id}

## Memory
{memory.replace("I ", f"{user_id} ") if memory else "No memory available."}

## Task
{message}

## Available Tools
{self.format_tools_as_xml(tools)}

---

## Instructions

### Step 1 — Decide if a tool is needed
- If the task is casual conversation or a simple factual question, answer directly as a helpful assistant. Do NOT invoke a tool.
- If you have already executed a tool and got a successful result, DO NOT call the tool again. Instead, answer the user directly with the final status.
- If the task requires file creation, data processing, or document manipulation and hasn't been done yet, select the most appropriate tool from the list above.

### Step 2 — If a tool IS needed, output ONLY this JSON (no explanation, no markdown). For example:
{{
"tool_name": "<exact tool name from the list>",
"tool_type": "<one of: function | module | mcp | agentskills>",
"module_path": "<module_path from the tool definition>",
"is_runtime": <true or false>,
"tool_call_id": "<tool_call_id from the tool definition>",
"arguments": {{
    "command": "<see rules below>"
}},
"return": "<return type from the tool definition>"
}}

### Step 3 — How to fill in `command`

The `command` value depends on `tool_type`:

| tool_type | What to write in `command` |
|---|---|
| `agentskills` | A **complete, agent-contained, runnable Python or bash script** that solves the user's task. Follow the code patterns in the tool's docstring. Save outputs to `/mnt/user-data/outputs/`. Do NOT copy docstring examples verbatim — write code specific to the task. |
| `function` | A function call string, e.g. `"get_weather(city='Hanoi')"` |
| `module` | A module-level invocation string appropriate for that module |
| `mcp` | The MCP action string as specified by that tool |

### Rules
- Never make up tool names, module paths, or arguments not present in the tool definitions.
- Never mix syntax from one tool into another.
- If you are unsure which tool to use, answer directly and suggest where the user might find help.
- Do not add explanation or commentary when outputting a tool call — output JSON only.
"""
        return prompt

    def action_prompt(self, prior_fix_str: str) -> str:
        _action_prompt = (
            "\n\n[TOOL ERROR RECOVERY REQUIRED]\n"
            "The previous tool call FAILED with the error shown above.\n"
            f"{prior_fix_str}\n\n"
            "You must FIRST analyze the root cause of the error.\n\n"
            "There are TWO possible recovery strategies:\n\n"
            "Step 1. CODE BUG (inside the tool command)\n"
            "- The Python code is incorrect\n"
            "- A runtime exception occurred (TypeError, ValueError, AttributeError, etc.)\n"
            "- Incorrect API usage\n"
            "- Logic bug in the script\n"
            "Action:\n"
            "  - Modify the Python code inside `tool_call.command`\n"
            "  - Set `requires_tool: True`\n"
            "  - Do NOT use `fix_bug_command`\n\n"
            "Step 2. ENVIRONMENT ISSUE\n"
            "- Missing library\n"
            "- Missing binary\n"
            "- PATH problem\n"
            "- Permission problem\n"
            "➡ Action:\n"
            "  - Provide a bash script in `fix_bug_command`\n"
            "  - Set `requires_tool: False`\n"
            "  - Do NOT modify the tool code\n\n"
            "IMPORTANT RULES:\n"
            "- Python exceptions almost always mean a CODE BUG.\n"
            "- Only install packages if the error explicitly says 'ModuleNotFoundError' or 'command not found'.\n"
            "- Do NOT repeat previous fixes.\n"
            "- The generated Python script MUST run the true procedure (not just idea/prompt to run the procedure)\n"
            "- Print the final result.\n"
            "- If creating output artifacts, print the artifact paths.\n"
            "- Scripts must never produce empty stdout.\n"
            "Respond in the structured JSON format with ONE of the two strategies."
        )
        return _action_prompt

    def system_prompt(
        self, skills: list[str], description: str, instruction: str
    ) -> str:
        skills = "- " + "- ".join(skills)
        content = (
            f"{description}\nYour skills:\n{skills}\nInstruction:\n{instruction}"
            if instruction
            else f"{description}\nYour skills:\n{skills}"
        )
        system_prompt = SystemMessage(content=content)
        return system_prompt

    def prompt_tool(
        self,
        query: str,
        tool_call: str,
        tool_message: ToolMessage,
        history: list[BaseMessage],
        *args,
        **kwargs,
    ) -> str:
        # Check if the tool execution resulted in an error
        is_error = tool_message.additional_kwargs.get("is_error", False)

        # Include BOTH the summary content AND the full artifact (STDERR traceback)
        # so the LLM has the complete picture to generate a comprehensive fix.
        content_value = tool_message.content or ""
        artifact_value = getattr(tool_message, "artifact", None) or ""
        if artifact_value and artifact_value != content_value:
            full_result = f"{content_value}\n\nFull output:\n{artifact_value}"
        else:
            full_result = content_value or artifact_value

        if is_error:
            # Collect prior fix attempts from history for context
            prior_fixes = []
            for msg in history:
                if (
                    isinstance(msg, ToolMessage)
                    and msg.additional_kwargs.get("is_error") is False
                ):
                    # successful fix attempt
                    prior_fixes.append(f"  - (success) {msg.content}")
                elif isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                    for tc in msg.tool_calls:
                        name = (
                            tc.get("name")
                            if isinstance(tc, dict)
                            else getattr(tc, "name", None)
                        )
                        if name == "fix_bug_command":
                            cmd = (
                                (tc.get("args") or {}).get("command", "")
                                if isinstance(tc, dict)
                                else ""
                            )
                            prior_fixes.append(f"  - (tried) {cmd}")

            prior_fix_str = ""
            if prior_fixes:
                prior_fix_str = (
                    "\n\nPrevious fix attempts (already tried, do NOT repeat):\n"
                    + "\n".join(prior_fixes)
                )

            _action_prompt = self.action_prompt(prior_fix_str)
        else:
            _action_prompt = (
                "The tool executed SUCCESSFULLY. Please review the Tool's Result above.\n"
                "If the task is complete, you MUST output a standard JSON response with `requires_tool`: false and provide a summary `answer` to the user. Do NOT call the same tool again."
            )

        tool_template = (
            f"- Question: {query}\n"
            f"- Tool Used: {tool_call}\n"
            f"""- Tool's Result:\n{full_result}\n"""
            f"{_action_prompt}"
        )

        return tool_template
