# 1. Introduction

`vinagent` is a lightweight and flexible library designed for building smart agent assistants across various industries. Whether you're creating an AI-powered customer service bot, a data analysis assistant, or a domain-specific automation agent, vinagent provides a simple yet powerful foundation.

With its modular tool system, you can easily extend your agent's capabilities by integrating a wide range of tools. Each tool is self-contained, well-documented, and can be registered dynamically—making it effortless to scale and adapt your agent to new tasks or environments.

![](https://imgur.com/s8SlWbR.png)

To install and use this library please following:

```
git@github.com:datascienceworld-kan/vinagent.git
cd vinagent
pip install -r requirements.txt
poetry install
```

Or you can install by pip command

```
pip install vinagent
```

To use a list of default tools inside [vinagent.tools](vinagent/tools/) you should set environment varibles inside `.env` including `TOGETHER_API_KEY` to use llm models at [togetherai](https://api.together.ai/signin) site and `TAVILY_API_KEY` to use tavily websearch tool at [tavily](https://app.tavily.com/home) site:

```
TOGETHER_API_KEY="Your together API key"
TAVILY_API_KEY="Your Tavily API key"
```
Let's create your acounts first and then create your relevant key for each website.

# 2. Set up Agent

`vinagent` is a flexible library for creating intelligent agents. You can configure your agent with tools, each encapsulated in a Python module under `vinagent.tools`. This provides a workspace of tools that agents can use to interact with and operate in the realistic world. Each tool is a Python file with full documentation and it can be independently ran. For example, the [vinagent.tools.websearch_tools](vinagent/tools/websearch_tools.py) module contains code for interacting with a search API.


```python
from langchain_together import ChatTogether 
from vinagent.agent.agent import Agent
from dotenv import load_dotenv
load_dotenv()

llm = ChatTogether(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
)

# Step 1: Create Agent with tools
agent = Agent(
    description="You are a Financial Analyst",
    llm = llm,
    skills = [
        "Deeply analyzing financial markets", 
        "Searching information about stock price",
        "Visualization about stock price"],
    tools = ['vinagent.tools.websearch_tools',
             'vinagent.tools.yfinance_tools'],
    tools_path = 'templates/tools.json', # Place to save tools. Default is 'templates/tools.json'
    is_reset_tools = True # If True, it will reset tools every time reinitializing an agent. Default is False
)

# Step 2: invoke the agent
message = agent.invoke("Who you are?")
```

If the answer is a normal message without using any tools, it will be an `AIMessage`. By contrast, it will have `ToolMessage` type. For examples:

```
message
```
```
AIMessage(content='I am a Financial Analyst.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 7, 'prompt_tokens': 308, 'total_tokens': 315, 'completion_tokens_details': None, 'prompt_tokens_details': None, 'cached_tokens': 0}, 'model_name': 'meta-llama/Llama-3.3-70B-Instruct-Turbo-Free', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-070f7431-7176-42a8-ab47-ed83657c9463-0', usage_metadata={'input_tokens': 308, 'output_tokens': 7, 'total_tokens': 315, 'input_token_details': {}, 'output_token_details': {}})
```
Access to `content` property to get the string content.

```
message.content
```
```
I am a Financial Analyst.
```

The following function need to use yfinancial tool, therefore the return value will be `ToolMessage` with the a stored pandas.DataFrame in `artifact` property.

```
df = agent.invoke("What is the price of Tesla stock in 2024?")
df
```
```
ToolMessage(content="Completed executing tool fetch_stock_data({'symbol': 'TSLA', 'start_date': '2024-01-01', 'end_date': '2024-12-31', 'interval': '1d'})", tool_call_id='tool_cde0b895-260a-468f-ac01-7efdde19ccb7', artifact=pandas.DataFrame)
```

To access `pandas.DataFrame` value:

```
df.artifact.head()
```

![png](asset/table.png)

Another example, if you visualize a stock price using a tool, the output message is a `ToolMessage` with the saved `artifact` is a plotly plot.

```
# return a ToolMessage which we can access to plot by plot.artifact and content by plot.content.
plot = agent.invoke("Let's visualize Tesla stock in 2024?")
```

![png](asset/test_4_1.png)
    

```
# return a ToolMessage which we can access to plot by plot.artifact and content by plot.content.
plot = agent.invoke("Let's visualize the return of Tesla stock in 2024?")
```
  
![png](asset/return_plot.png)
    

# 3. Register Tool

Vinagent stands out for its flexibility in registering different types of tools, including:

- Function tools: These are integrated directly into your runtime code using the @function_tool decorator, without the need to store them in separate Python module files.
- Module tools: These are added via Python module files placed in the vinagent.tools directory. Once registered, the modules can be imported and used in your runtime environment.
- MCP tools: These are tools registered through an [MCP (Model Context Protocol) server](https://github.com/modelcontextprotocol/servers), enabling external tool integration.

## 3.1. Function Tool

You can customize any function in your runtime code as a powerful tool by using the @function_tool decorator.

```python
from vinagent.register.tool import function_tool
from typing import List

@agent.function_tool # Note: agent must be initialized first
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

## 3.2. Module Tool
Many complex tools cannot be implemented within a single function. In such cases, organizing the tool as a python module becomes necessary. To support this, `vinagent` allows tools to be registered via python module files placed in the `vinagent.tools` directory. This approach makes it easier to manage and execute more sophisticated tasks. Once registered, these modules can be imported and used directly in the runtime environment.

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

## 3.3. MCP Tool

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

# 4. Invoke and streaming

## 4.1. Synchronous and Asynchronous Invocation
Vinagent offers both synchronous (agent.invoke) and asynchronous (agent.ainvoke) invocation methods. While synchronous calls halt the main thread until a response is returned, asynchronous calls enable the main thread to proceed without waiting. In practice, asynchronous invocations can be up to twice as fast as synchronous ones.

```
# Synchronous invocation
message = agent.invoke("What is the sum of 1993 and 709?")
```

```
# Asynchronous invocation
message = await agent.ainvoke("What is the sum of 1993 and 709?")
```

## 4.2. Streaming Invocation
In addition to synchronous and asynchronous invocation, vinagent also supports streaming invocation. This means that the response is generated in real-time on token-by-token basis, allowing for a more interactive and responsive experience. To use streaming, simply use `agent.stream`:

```
for chunk in agent.stream("Where is the capital of the Vietnam?"):
    print(chunk)
```

# 5. Advance Features

## 5.1. Deep Search

With vinagent, you can invent a complex workflow by combining multiple tools into a single agent. This allows you to create a more sophisticated and flexible agent that can adapt to different task. Let's see how an agent can be created to help with financial analysis by using `deepsearch` tool, which allows you to search for information in a structured manner. This tool is particularly useful for tasks that require a deep understanding of the data and the ability to navigate through complex information.

```
from langchain_together import ChatTogether 
from vinagent.agent import Agent
from dotenv import load_dotenv
load_dotenv()

llm = ChatTogether(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
)

agent = Agent(
    description="You are a Financial Analyst",
    llm = llm,
    skills = [
        "Deeply analyzing financial markets", 
        "Searching information about stock price",
        "Visualization about stock price"],
    tools = ['vinagent.tools.deepsearch']
)

message = agent.invoke("Let's analyze Tesla stock in 2025?")
print(message.artifact)
```

[![Watch the video](https://img.youtube.com/vi/MUOg7MYGUzE/0.jpg)](https://youtu.be/MUOg7MYGUzE)

The output is available at [vinagent/examples/deepsearch.md](vinagent/examples/deepsearch.md)

## 5.2. Trending Search

Exceptionally, vinagent also offers a feature to summarize and highlight the top daily news on the internet based on any topic you are looking for, regardless of the language used. This is achieved by using the `trending_news` tool.

```
from langchain_together import ChatTogether 
from vinagent.agent.agent import Agent
from dotenv import load_dotenv
load_dotenv()

llm = ChatTogether(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
)

agent = Agent(
    description="You are a Trending News Analyst",
    llm = llm,
    skills = [
        "Searching the trending news on realtime from google news",
        "Deeply analyzing top trending news"],
    tools = ['vinagent.tools.trending_news']
)
    
message = agent.invoke("Tìm 5 tin tức nổi bật về tình hình giá vàng sáng hôm nay")
print(message.artifact)
```

[![Watch the video](https://img.youtube.com/vi/c8ylwGDYl2c/0.jpg)](https://youtu.be/c8ylwGDYl2c?si=D7aMgY5f_WJqPbFm)

The output is available at [vinagent/examples/todaytrend.md](vinagent/examples/todaytrend.md)


# 6. Agent with Memory

There is a special feature that allows to adhere Memory for each Agent. This is useful when you want to keep track of the user's behavior and conceptualize them as a knowledge graph for the agent. Therefore, it helps agent become more intelligent and capable of understanding personality and responding to user queries with greater accuracy.

The following code to save each conversation into short-memory.

```
from langchain_together.chat_models import ChatTogether
from dotenv import load_dotenv
from vinagent.memory import Memory

load_dotenv()

llm = ChatTogether(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
)

memory = Memory(
    memory_path="templates/memory.jsonl",
    is_reset_memory=True, # will reset the memory every time the agent is invoked
    is_logging=True
)

text_input = """Hi, my name is Kan. I was born in Thanh Hoa Province, Vietnam, in 1993.
My motto is: "Make the world better with data and models". That’s why I work as an AI Solution Architect at FPT Software and as an AI lecturer at NEU.
I began my journey as a gifted student in Mathematics at the High School for Gifted Students, VNU University, where I developed a deep passion for Math and Science.
Later, I earned an Excellent Bachelor's Degree in Applied Mathematical Economics from NEU University in 2015. During my time there, I became the first student from the Math Department to win a bronze medal at the National Math Olympiad.
I have been working as an AI Solution Architect at FPT Software since 2021.
I have been teaching AI and ML courses at NEU university since 2022.
I have conducted extensive research on Reliable AI, Generative AI, and Knowledge Graphs at FPT AIC.
I was one of the first individuals in Vietnam to win a paper award on the topic of Generative AI and LLMs at the Nvidia GTC Global Conference 2025 in San Jose, USA.
I am the founder of DataScienceWorld.Kan, an AI learning hub offering high-standard AI/ML courses such as Build Generative AI Applications and MLOps – Machine Learning in Production, designed for anyone pursuing a career as an AI/ML engineer.
Since 2024, I have participated in Google GDSC and Google I/O as a guest speaker and AI/ML coach for dedicated AI startups.
"""

memory.save_short_term_memory(llm, text_input)
memory_message = memory.load_memory('string')
memory_message
```

```
Kan -> BORN_IN[in 1993] -> Thanh Hoa Province, Vietnam
Kan -> WORKS_FOR[since 2021] -> FPT Software
Kan -> WORKS_FOR[since 2022] -> NEU
Kan -> STUDIED_AT -> High School for Gifted Students, VNU University
Kan -> STUDIED_AT[graduated in 2015] -> NEU University
Kan -> RESEARCHED_AT -> FPT AIC
Kan -> RECEIVED_AWARD[at Nvidia GTC Global Conference 2025] -> paper award on Generative AI and LLMs
Kan -> FOUNDED -> DataScienceWorld.Kan
Kan -> PARTICIPATED_IN[since 2024] -> Google GDSC
Kan -> PARTICIPATED_IN[since 2024] -> Google I/O
DataScienceWorld.Kan -> OFFERS -> Build Generative AI Applications
DataScienceWorld.Kan -> OFFERS -> MLOps – Machine Learning in Production
Kan -> OWNS -> house
Kan -> OWNS -> garden
Kan -> HAS_MOTTO -> stay hungry and stay foolish
Kan -> RECEIVED_AWARD_AT[in 2025] -> Nvidia GTC Global Conference
```

To adhere Memmory to each Agent

```
import os
import sys
from langchain_together import ChatTogether 
from vinagent.agent import Agent
from vinagent.memory.memory import Memory
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

llm = ChatTogether(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
)

# Step 1: Create Agent with tools
agent = Agent(
    llm = llm,
    description="You are my close friend",
    skills=[
        "You can remember all memory related to us",
        "You can remind the memory to answer questions",
        "You can remember the history of our relationship"
    ],
    memory_path='templates/memory.json',
    is_reset_memory=True # Will reset memory each time re-initialize agent. Default is False
)

# Step 2: invoke the agent
message = agent.invoke("Hello how are you?")
message.content
```

# 7. Design Agent flow 

The Vinagent library enables the integration of workflows built upon the nodes and edges of LangGraph. What sets it apart is our major improvement in representing a LangGraph workflow through a more intuitive syntax for connecting nodes using the right shift operator (`>>`). All agent patterns such as ReAct, chain-of-thought, and reflection can be easily constructed using this simple and readable syntax.

We support two styles of creating a workflow:
- `FlowStateGraph`: Create nodes by concrete class nodes inherited from class Node of vinagent.
- `FunctionStateGraph`: Create a workflow from function, which are decorated with @node to convert this function as a node. 

## 7.1. FlowStateGraph

These are steps to create a workflow:

1. Define General Nodes
Create your workflow nodes by inheriting from the base Node class. Each node typically implements two methods:

- `exec`: Executes the task associated with the node and returns a partial update to the shared state.

- `branching` (optional): For conditional routing. It returns a string key indicating the next node to be executed.

2. Connect Nodes with `>>` Operator
Use the right shift operator (`>>`) to define transitions between nodes. For branching, use a dictionary to map conditions to next nodes.

```
from typing import Annotated, TypedDict
from vinagent.graph.operator import FlowStateGraph, END, START
from vinagent.graph.node import Node
from langgraph.checkpoint.memory import MemorySaver
from langgraph.utils.runnable import coerce_to_runnable

# Define a reducer for message history
def append_messages(existing: list, update: dict) -> list:
    return existing + [update]

# Define the state schema
class State(TypedDict):
    messages: Annotated[list[dict], append_messages]
    sentiment: str

# Optional config schema
class ConfigSchema(TypedDict):
    user_id: str

# Define node classes
class AnalyzeSentimentNode(Node):
    def exec(self, state: State) -> dict:
        message = state["messages"][-1]["content"]
        sentiment = "negative" if "angry" in message.lower() else "positive"
        return {"sentiment": sentiment}

    def branching(self, state: State) -> str:
        return "human_escalation" if state["sentiment"] == "negative" else "chatbot_response"

class ChatbotResponseNode(Node):
    def exec(self, state: State) -> dict:
        return {"messages": {"role": "bot", "content": "Got it! How can I assist you further?"}}

class HumanEscalationNode(Node):
    def exec(self, state: State) -> dict:
        return {"messages": {"role": "bot", "content": "I'm escalating this to a human agent."}}

# Define the Agent with graph and flow
class Agent:
    def __init__(self):
        self.checkpoint = MemorySaver()
        self.graph = FlowStateGraph(State, config_schema=ConfigSchema)
        self.analyze_sentiment_node = AnalyzeSentimentNode()
        self.human_escalation_node = HumanEscalationNode()
        self.chatbot_response_node = ChatbotResponseNode()

        self.flow = [
            self.analyze_sentiment_node >> {
                "chatbot_response": self.chatbot_response_node,
                "human_escalation": self.human_escalation_node
            },
            self.human_escalation_node >> END,
            self.chatbot_response_node >> END
        ]

        self.compiled_graph = self.graph.compile(checkpointer=self.checkpoint, flow=self.flow)

    def invoke(self, input_state: dict, config: dict) -> dict:
        return self.compiled_graph.invoke(input_state, config)

# Test the agent
agent = Agent()
input_state = {
    "messages": {"role": "user", "content": "I'm really angry about this!"}
}
config = {"configurable": {"user_id": "123"}, "thread_id": "123"}
result = agent.invoke(input_state, config)
print(result)
```

Output:

```
{
  'messages': [
    {'role': 'user', 'content': "I'm really angry about this!"},
    {'role': 'bot', 'content': "I'm escalating this to a human agent."}
  ],
  'sentiment': 'negative'
}
```

We can visualize the graph workflow on jupyternotebook

```
agent.compiled_graph
```

![](asset/langgraph_output.png)


## 7.2. FunctionStateGraph

We can simplify the coding style of a graph by converting each function into a node and assigning it a name.

1. Each node will be a function with the same name as the node itself. However, you can override this default by using the `@node(name="your_node_name")` decorator.

2. If your node is a conditionally branching node, you can use the `@node(branching=fn_branching)` decorator, where `fn_branching` is a function that determines the next node(s) based on the return value of current state of node.

3. In the Agent class constructor, we define a flow as a list of routes that connect these node functions.

```
from typing import Annotated, TypedDict
from vinagent.graph.operator import END, START
from vinagent.graph.function_graph import node, FunctionStateGraph
from vinagent.graph.node import Node
from langgraph.checkpoint.memory import MemorySaver
from langgraph.utils.runnable import coerce_to_runnable

# Define a reducer for message history
def append_messages(existing: list, update: dict) -> list:
    return existing + [update]

# Define the state schema
class State(TypedDict):
    messages: Annotated[list[dict], append_messages]
    sentiment: str

# Optional config schema
class ConfigSchema(TypedDict):
    user_id: str

def branching(state: State) -> str:
    return "human_escalation" if state["sentiment"] == "negative" else "chatbot_response"

@node(branching=branching, name='AnalyzeSentiment')
def analyze_sentiment_node(state: State) -> dict:
    message = state["messages"][-1]["content"]
    sentiment = "negative" if "angry" in message.lower() else "positive"
    return {"sentiment": sentiment}

@node(name='ChatbotResponse')
def chatbot_response_node(state: State) -> dict:
    return {"messages": {"role": "bot", "content": "Got it! How can I assist you further?"}}

@node(name='HumanEscalation')
def human_escalation_node(state: State) -> dict:
    return {"messages": {"role": "bot", "content": "I'm escalating this to a human agent."}}

# Define the Agent with graph and flow
class Agent:
    def __init__(self):
        self.checkpoint = MemorySaver()
        self.graph = FunctionStateGraph(State, config_schema=ConfigSchema)

        self.flow = [
            analyze_sentiment_node >> {
                "chatbot_response": chatbot_response_node,
                "human_escalation": human_escalation_node
            },
            human_escalation_node >> END,
            chatbot_response_node >> END
        ]

        self.compiled_graph = self.graph.compile(checkpointer=self.checkpoint, flow=self.flow)

    def invoke(self, input_state: dict, config: dict) -> dict:
        return self.compiled_graph.invoke(input_state, config)

# Test the agent
agent = Agent()
input_state = {
    "messages": {"role": "user", "content": "I'm really angry about this!"}
}
config = {"configurable": {"user_id": "123"}, "thread_id": "123"}
result = agent.invoke(input_state, config)
print(result)
```

Output:

```
{'messages': [{'role': 'user', 'content': "I'm really angry about this!"}, {'role': 'bot', 'content': "I'm escalating this to a human agent."}], 'sentiment': 'negative'}
```

Visualize the workflow:

```
agent.compiled_graph
```

![](asset/langgraph_function_output.png)

# 10. Agent Observability

Vinagent provides a local server that can be used to visualize the workflow and debug the agent. We leverage mlflow observability to track the agent's progress and performance. To use the local server, run the following command:

- Step 1: Start the local mlflow server.

```
mlflow ui
```
This command will deploy a local loging server on port 5000 to your agent connect to.

- Step 2: Initialize an experiment to auto-log messages for agent

```
import mlflow
from vinagent.mlflow import autolog

# Enable Vinagent autologging
autolog.autolog()

# Optional: Set tracking URI and experiment
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Vinagent")
```

After this step, one hooking function will be registered after agent invoking.

- Step 3: Run your agent

```
from langchain_together import ChatTogether 
from vinagent.agent.agent import Agent
from dotenv import load_dotenv
load_dotenv()

llm = ChatTogether(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
)

agent = Agent(
    description="You are an Expert who can answer any general questions.",
    llm = llm,
    skills = [
        "Searching information from external search engine\n",
        "Summarize the main information\n"],
    tools = ['vinagent.tools.websearch_tools'],
    tools_path = 'templates/tools.json',
    memory_path = 'templates/memory.json'
)

result = agent.invoke(query="What is the weather today in Ha Noi?")
```

You can watch the video below to learn more about Agent observability feature:

[![Watch the video](https://img.youtube.com/vi/UgZLhoIgc94/0.jpg)](https://youtu.be/UgZLhoIgc94?si=qidf9fX3i4Cf0ETp)


# 9. License
`vinagent` is released under the MIT License. You are free to use, modify, and distribute the code for both commercial and non-commercial purposes.

# 10. Contributing
We welcome contributions from the community. If you would like to contribute, please read our [Contributing Guide](https://github.com/datascienceworld-kan/vinagent/blob/main/CONTRIBUTING.md). If you have any questions or need help, feel free to join [Discord Channel](https://discord.com/channels/1036147288994758717/1358017320970358864).

# 11. Credits

We acknowledge the contributions of previous open-source library and platform that inspired the development of `vinagent`:

- [LangChain](https://github.com/langchain-ai/langchain) – for offering standardizing base classes.
- [Langgraph](https://github.com/langchain-ai/langgraph) – for building workflow.
- [LlamaIndex](https://github.com/jerryjliu/llama_index) – for advanced features like deepsearch, trending search.
- [Together.ai](https://api.together.ai) – for providing a powerful and free generative AI models.
- [Tavily](https://app.tavily.com) – for providing a powerful web search tool.
