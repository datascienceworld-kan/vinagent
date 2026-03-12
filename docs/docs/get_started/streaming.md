# Streaming Agent

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datascienceworld-kan/vinagent/blob/main/docs/docs/tutorials/get_started/streaming.ipynb)

## Install libraries

```python
%pip install vinagent
```

## Streaming
In addition to synchronous and asynchronous invocation, `Vinagent` also supports streaming invocation. This means that the response is generated in real-time on token-by-token basis, allowing for a more interactive and responsive experience. To use streaming, simply use `agent.stream`.

Setup environment variables:

```python
%%writefile .env
TOGETHER_API_KEY="Your together API key"
TAVILY_API_KEY="Your Tavily API key"
```

Initialize LLM and Agent:

```python
from vinagent.agent.agent import Agent
from langchain_together import ChatTogether
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv('.env'))

# Step 1: Initialize LLM
llm = ChatTogether(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
)

# Step 2: Initialize Agent
agent = Agent(
    description="You are a Weather Analyst",
    llm = llm,
    skills = [
        "Update weather at anywhere",
        "Forecast weather in the futher",
        "Recommend picnic based on weather"
    ],
    tools=['vinagent.tools.websearch_tools'],
    tools_path = 'templates/tools.json', # Place to save tools. Default is 'templates/tools.json'
    is_reset_tools = True # If True, it will reset tools every time reinitializing an agent. Default is False
)
```

Streaming provides a significant advantage in Agent invocation by delivering output token-by-token in runtime, allowing users to read a long-running answer as it exposures without waiting for the entire response to complete. 

- It greatly enhances the user experience, especially when integrating the agent into websites or mobile apps, where responsiveness and interactivity are critical. 

- Streaming is particularly effective for long outputs and I/O-bound tasks, enabling dynamic UI updates, early interruption, and a more natural, real-time interaction flow. 

You can conveniently use streaming in Vinagent by iterating over the generator returned by the `agent.stream()` method.


```python
content = ''
for chunk in agent.stream(query="What is the weather in New York today?"):
    content += chunk.content
    content += '|'
    print(content)
```

    INFO:vinagent.agent.agent:I am chatting with unknown_user
    INFO:httpx:HTTP Request: POST https://api.together.xyz/v1/chat/completions "HTTP/1.1 200 OK"


    To|


    INFO:vinagent.agent.agent:Tool call: {'tool_name': 'search_api', 'tool_type': 'module', 'arguments': {'query': 'New York weather today'}, 'module_path': 'vinagent.tools.websearch_tools'}


    To| find|
    To| find| the|
    To| find| the| current|
    To| find| the| current| weather|
    To| find| the| current| weather| in|
    To| find| the| current| weather| in| New|
    To| find| the| current| weather| in| New| York|
    To| find| the| current| weather| in| New| York|,|
    To| find| the| current| weather| in| New| York|,| I|
    To| find| the| current| weather| in| New| York|,| I| will|
    To| find| the| current| weather| in| New| York|,| I| will| use|
    To| find| the| current| weather| in| New| York|,| I| will| use| the|
    To| find| the| current| weather| in| New| York|,| I| will| use| the| search|
    To| find| the| current| weather| in| New| York|,| I| will| use| the| search|_api|
    To| find| the| current| weather| in| New| York|,| I| will| use| the| search|_api| tool|
    
    According to the search_api tool, the current weather in New York today is 72°F with mist. The wind is blowing at 6 mph from the west, and the humidity is relatively high at 94%.|

## Async Streaming
Vinagent offer an `astream` method that is async version of `stream` to accelerate the speed of response. It is particularly useful for I/O bound tasks that needs waiting for data from external sources.

```python
import asyncio
async def run_agent():
    content = ""

    async for chunk in agent.astream(query="Where is the capital of the Vietnam?"):
        content += chunk.content or ""
        print(chunk.content, end="", flush=True)

    print("\nFinal content:", content)

asyncio.run(run_agent())
```
