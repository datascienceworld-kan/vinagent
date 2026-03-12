# Add tools
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datascienceworld-kan/vinagent/blob/main/docs/docs/tutorials/get_started/add_tool.ipynb)

## Prerequisites

Install `vinagent` library

```python
%pip install vinagent
```

## Tool types

A tool is an important part of an AI agent. It allows your agent to connect to external data and perform tasks beyond the capabilities of an LLM. There are many ways to extend an agent with a new tool. However, Vinagent can connect to three types of tools that are available in its components.


- Function tools: These are integrated directly into your runtime code using the `@function_tool` decorator, without the need to store them in separate Python module files.

- Module tools: Gathering many tools into an unique Python module. Using decorator `@primary_function` to state which functions are selected as Agent tools. Once registered, the modules can be imported and used in your runtime environment.

- MCP tools: Connect Agent to thousands of third-party tools, enabling [MCP (Model Context Protocol) server](https://github.com/modelcontextprotocol/servers) like Google Drive, Gmail, Slack, Notion, Spotify,... to be used as external tools.


## Example of module tool

You can add module tools from a Python module path as follows:
- Initialize an LLM model, which can be any model wrapped by the [Langchain BaseLLM](https://python.langchain.com/api_reference/core/language_models/langchain_core.language_models.llms.BaseLLM.html) class. I use [TogetherAI](https://api.together.ai/signin) chat model in there, thus, you need to create `.env` environment with variable
```
TOGETHER_API_KEY="Your together API key"
```
You can use other LLM Provider API as long as it was initialized by Langchain `BaseLLM` class.


```python
from langchain_together import ChatTogether 
from vinagent.agent.agent import Agent
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv('.env'))

llm = ChatTogether(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
)
```

- Initialize an Agent with tools, which are wrapped inside the tools argument as a list of paths:



```python
import os
os.makedirs('./tools', exist_ok=True)
```


```python
%%writefile tools/hello.py
def hello_from_vinagent():
    '''A greet of Vinagent to everyone'''
    return "Hello my cute cute friend, I'm vinagent and I am here to play with you 😄!"
```

    Writing tools/hello.py



```python
# Step 1: Create Agent with tools
agent = Agent(
    description="You are a Vinagent",
    llm = llm,
    skills = [
        "Friendly talk with anyone"
    ],
    tools = ['tools/hello.py'],
    tools_path = 'templates/tools.json',
    is_reset_tools = True
)
```

    INFO:httpx:HTTP Request: POST https://api.together.xyz/v1/chat/completions "HTTP/1.1 200 OK"
    INFO:vinagent.register.tool:Registered hello_from_vinagent:
    {'tool_name': 'hello_from_vinagent', 'arguments': {}, 'return': "Hello my cute cute friend, I'm vinagent and I am here to play with you 😄!", 'docstring': 'A greet of Vinagent to everyone', 'dependencies': [], 'module_path': 'vinagent.tools.hello', 'tool_type': 'module', 'tool_call_id': 'tool_a25e45c3-81df-4b68-982d-d308c403a725'}
    INFO:vinagent.register.tool:Completed registration for module vinagent.tools.hello


!!! note
    `tools_path` is where the tools dictionary is saved. The default value is templates/tools.json.

!!! tip "Resetting Your Tools"
    If you set `is_reset_tools = True`, it will override the tool definitions every time an agent is reinitialized.


Asking tool

```python
# Step 2: invoke the agent
message = agent.invoke("Hi Vinagent, Can you greet by your style?")
print(message.content)
```

    
    Hello my friend, I'm vinagent, an AI smart assistant. I come here to help you 😄!


## Function Tool

In the following sections, let's explore how to register each type of tools in `vinagent` library. First, let's see how to register a function tool. You can customize any function in your runtime code as a powerful tool by using the `@agent.tools_manager.register_function_tool` decorator.

```python
from typing import List

@agent.tools_manager.register_function_tool # Note: agent must be initialized first
def sum_of_series(x: List[float]):
    return f"Sum of list is {sum(x)}"
```
```
INFO:root:Registered tool: sum_of_series (runtime)
```

```python
message = agent.invoke("Sum of this list: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]?")
message
```
```
ToolMessage(content="Completed executing tool sum_of_series({'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})", tool_call_id='tool_56f40902-33dc-45c6-83a7-27a96589d528', artifact='Sum of list is 55')
```

## Module Tool
Many complex tools cannot be implemented within a single function. In such cases, organizing the tool as a python module becomes necessary. To support this, `vinagent` allows tools to be registered via python module files placed in the `vinagent.tools` directory. This approach makes it easier to manage and execute more sophisticated tasks. Once registered, these modules can be imported and used directly in the runtime environment.

Let's write [websearch_tools](https://github.com/datascienceworld-kan/vinagent/blob/main/vinagent/tools/websearch_tools.py) module as follows:

```
%%writefile vinagent/tools/websearch_tools.py
import os
from dotenv import load_dotenv
from tavily import TavilyClient
from dataclasses import dataclass
from typing import Union, Any
from vinagent.register import primary_function

_ = load_dotenv()


@dataclass
class WebSearchClient:
    tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

    def call_api(self, query: Union[str, dict[str, str]]):
        if isinstance(query, dict):
            query_string = "\n".join([f"{k}: {v}" for (k, v) in query.items()])
        else:
            query_string = query
        result = self.tavily_client.search(query_string, include_answer=True)
        return result["answer"]

@primary_function
def search_api(query: Union[str, dict[str, str]]) -> Any:
    """
    Search for an answer from a query string
    Args:
        query (dict[str, str]):  The input query to search
    Returns:
        The answer from search query
    """
    client = WebSearchClient()
    answer = client.call_api(query)
    return answer
```

!!! note 
    If a module contains many functions but only a selected list of main functions should be considered as agent tools. To identify these, add the @primary_function decorator to mark them as agent tool methods. Otherwise, all functions in the module will be registered as tools.

```
from langchain_together import ChatTogether 
from vinagent.agent.agent import Agent
from dotenv import load_dotenv
load_dotenv()

llm = ChatTogether(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
)

agent = Agent(
    description="You are a Web Search Expert",
    llm = llm,
    skills = [
        "Search the information from internet", 
        "Give an in-deepth report",
        "Keep update with the latest news"
    ],
    tools = ['vinagent.tools.websearch_tools'],
    tools_path = 'templates/tools.json' # Place to save tools. The default path is also 'templates/tools.json',
    is_reset_tools = True # If True, will reset tools every time. Default is False
)
```

## MCP Tool

MCP (model context protocal) is a new AI protocal offfered by Anthropic that allows any AI model to interact with any tools distributed acrooss different platforms. These tools are provided by platform's [MCP Server](https://github.com/modelcontextprotocol/servers). There are many MCP servers available out there such as `google drive, gmail, slack, notions, spotify, etc.`, and `vinagent` can be used to connect to these servers and execute the tools within the agent.

You need to start a MCP server first. For example, start with [math MCP Server](vinagent/mcp/examples/math/README.md)

```
cd vinagent/mcp/examples/math
mcp dev main.py
```
```
⚙️ Proxy server listening on port 6277
🔍 MCP Inspector is up and running at http://127.0.0.1:6274 🚀
```

Next, you need to register the MCP server in the agent. You can do this by adding the server's URL to the `tools` list of the agent's configuration.

```
from vinagent.mcp.client import DistributedMCPClient
from vinagent.mcp import load_mcp_tools
from vinagent.agent.agent import Agent
from langchain_together import ChatTogether
from dotenv import load_dotenv

load_dotenv()

# Step 1: Initialize LLM
llm = ChatTogether(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
)

# Step 2: Initialize MCP client for Distributed MCP Server
client = DistributedMCPClient(
            {
                "math": {
                    "command": "python",
                    # Make sure to update to the full absolute path to your math_server.py file
                    "args": ["vinagent/mcp/examples/math/main.py"],
                    "transport": "stdio",
                }
             }
        )
server_name = "math"

# Step 3: Initialize Agent
agent = Agent(
    description="You are a Trending News Analyst",
    llm = llm,
    skills = [
        "You are Financial Analyst",
        "Deeply analyzing financial news"],
    tools = ['vinagent.tools.yfinance_tools'],
    tools_path="templates/tools.json",
    is_reset_tools=True,
    mcp_client=client, # MCP Client
    mcp_server_name=server_name, # MCP Server name to resgister. If not set, all tools from all MCP servers available
)

# Step 4: Register mcp_tools to agent
mcp_tools = await agent.connect_mcp_tool()
```

```
# Test sum
agent.invoke("What is the sum of 1993 and 709?")
```

```
# Test product
agent.invoke("Let's multiply of 1993 and 709?")
```
