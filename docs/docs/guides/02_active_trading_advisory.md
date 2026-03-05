# Intermediate - Active Trading & Advisory

_Contributor: Gia Bao; Reviewed & Extended by: Kan Pham_

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datascienceworld-kan/vinagent/blob/main/cookbook/multi_agent_in_finance/02_active_trading_advisory.ipynb)

Modern financial markets move across multiple asset classes simultaneously. Equities, cryptocurrencies, commodities, and macroeconomic signals interact continuously, often influencing each other in subtle ways. Investors who focus on a single market frequently miss critical signals emerging from others.

For example:

- Semiconductor stocks may surge following strong GPU demand.

- Cryptocurrency markets may react to liquidity shifts or macroeconomic sentiment.

- Institutional capital may rotate between equities and digital assets.

Successful investment strategies therefore rely on cross-market awareness, not isolated analysis. However, monitoring multiple markets in real time presents a challenge. Most traditional investment workflows still rely on sequential analysis pipelines:

- Collect market data

- Analyze signals

- Produce a recommendation

This approach becomes inefficient when data sources grow. Analysts must wait for each step to complete before the next begins, creating latency in a world where markets react within seconds.

To address this limitation, we introduce Active Trading Advisory, an AI-driven system designed to continuously observe multiple markets and synthesize trading recommendations dynamically.

### System Architecture
To achieve these goals, we evolve from simple sequential workflows toward a Parallel Agent Execution model.

Instead of collecting and analyzing data one source at a time, the system deploys specialized agents that operate concurrently, each responsible for a different market.

```
          ┌→ Market Data Provider →┐
START ----┤                        ├→ Recommender Agent → END
          └→ Binance Tracker ------→┘

```

In this architecture:

- Independent agents gather insights from separate markets.

- Analysis occurs in parallel, reducing latency.

- A synthesis agent aggregates signals into a final recommendation.

| Agent | Role | Parallel Status |
|---|---|---|
| **MarketDataProvider** | Analyzes traditional stock equity (e.g., NVDA) | Yes |
| **BinanceTracker** | Analyzed crypto assets (e.g., BTC-USD) | Yes |
| **RecommenderAgent** | Synthesizes a multi-market strategy | No (Wait for both) |

This design reflects a key principle of modern AI systems: specialized agents collaborating to solve complex tasks efficiently.

### Implementing Parallel Agents with Vinagent

In the following sections, we demonstrate how to build this Active Trading Advisory system using the Vinagent framework. Vinagent can provide structured agent orchestration, tool integration for external data sources into flexible multi-agent workflows that native support for parallel execution. By using these capabilities, we can transform traditional trading pipelines into adaptive AI advisory systems capable of real-time multi-market analysis.

## Environment Setup


```python
%pip install vinagent==0.0.6.post6
```

```python
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv('.env'))

llm = ChatOpenAI(model="gpt-4o-mini")
print("LLM initialized.")
```

    LLM initialized.


## Define Multi-Market State

Next, let's define a shared state object that all agents read from and write to. This is the "memory" of the system, it holds partial results (equity insight, crypto insight) that flow between agents and accumulate into the final recommendation.

The state contains specific keys for each market stream, plus a running messages log for traceability.


```python
from typing import Annotated, TypedDict

def append_messages(existing: list, update: dict) -> list:
    return existing + [update]

class TradingState(TypedDict):
    messages: Annotated[list[dict], append_messages]
    equity_insight: str
    crypto_insight: str
    final_recommendation: str
```

By leveraging this structure: `equity_insight` and `crypto_insight` are written by parallel agents independently.
`final_recommendation` is only written by RecommenderAgent after both insights are available. The `append_messages` reducer ensures every specialized agent's message entry stacked into history.


## Define Data Fetching Tools

Creating tools that agents can invoke to retrieve real market data. `get_current_time` lets agents timestamp their analysis; `fetch_btc_market_data` pulls live BTC metrics from CoinDesk. Wrapping them with @primary_function makes them compatible with Vinagent's tool registry.

```python
from datetime import datetime
import requests
from vinagent.register import primary_function


@primary_function
def get_current_time() -> str:
    """
    Get the current date and time. Use this to know 'today's' date.
    Returns:
        str: Current date and time in YYYY-MM-DD HH:MM:SS format.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print("Custom time tool defined.")

# You need to have your CoinDesk API in your environment variables
@primary_function
def fetch_btc_market_data() -> str:
    """
    Fetch comprehensive Bitcoin (BTC) market data including price, changes (daily/weekly/monthly), and moving averages from the CoinDesk API.
    Returns:
        str: A comprehensive summary of BTC market metrics.
    """
    try:
        response = requests.get('https://data-api.coindesk.com/index/cc/v1/latest/tick',
                params={"market":"cadli","instruments":"BTC-USD","apply_mapping":"true"},
                headers={"Content-type":"application/json; charset=UTF-8"}
        )
        data = response.json()['Data']['BTC-USD']
        
        summary = (
            f"--- BTC-USD Market Data ---\n"
            f"Current Price: ${data['VALUE']:.2f} USD\n"
            f"Day Change: {data['CURRENT_DAY_CHANGE']:.2f} ({data['CURRENT_DAY_CHANGE_PERCENTAGE']:.2f}%)\n"
            f"Week Change: {data['CURRENT_WEEK_CHANGE']:.2f} ({data['CURRENT_WEEK_CHANGE_PERCENTAGE']:.2f}%)\n"
            f"Month Change: {data['CURRENT_MONTH_CHANGE']:.2f} ({data['CURRENT_MONTH_CHANGE_PERCENTAGE']:.2f}%)\n"
            f"24h Range: ${data['MOVING_24_HOUR_LOW']:.2f} - ${data['MOVING_24_HOUR_HIGH']:.2f}\n"
            f"Year Change: {data['CURRENT_YEAR_CHANGE_PERCENTAGE']:.2f}%\n"
            f"Volatility Context: Day High ${data['CURRENT_DAY_HIGH']:.2f} / Low ${data['CURRENT_DAY_LOW']:.2f}"
        )
        return summary
    except Exception as e:
        return f"Error fetching BTC price: {e}"

print(fetch_btc_market_data())
```

    Custom time tool defined.
    --- BTC-USD Market Data ---
    Current Price: $72498.60 USD
    Day Change: -192.65 (-0.27%)
    Week Change: 6712.50 (10.20%)
    Month Change: 5458.29 (8.14%)
    24h Range: $67469.42 - $74036.08
    Year Change: -17.18%
    Volatility Context: Day High $73264.09 / Low $72350.48


## Implement Parallel Agents

Three agent classes: `MarketDataProvider`, `BinanceTracker`, and `RecommenderAgent` will join in a branching static workflow. The first two inherit from `AgentNode` and are designed to run concurrently (they do not depend on each other). `RecommenderAgent` reads both their outputs from the shared state to synthesize a final recommendation. Each agent's exec method defines its logic and what it writes back to the state.

```python
from vinagent.multi_agent import AgentNode
from vinagent.logger.logger import logging_message

class MarketDataProvider(AgentNode):
    """Fetches and analyzes stock market data."""
    @logging_message
    def exec(self, state: TradingState) -> dict:
        # In a real scenario, this would use fetch_stock_data tool
        prompt = "Analyze FPT's current market sentiment and technical outlook. Provide a summary for a trader."
        output = self.invoke(prompt)
        return {
            "messages": {"role": "MarketDataProvider", "content": output.content if hasattr(output, "content") else str(output)},
            "equity_insight": output.content if hasattr(output, "content") else str(output)
        }

class BinanceTracker(AgentNode):
    """Fetches and analyzes crypto market data."""
    @logging_message
    def exec(self, state: TradingState) -> dict:
        # Instruction to use the dedicated BTC tool for performance metrics
        prompt = (
            "Analyze Bitcoin (BTC-USD) current performance, trends, and volatility. "
            "IMPORTANT: Use the 'fetch_btc_market_data' tool to get comprehensive metrics "
            "(daily/weekly/monthly changes and ranges). Provide a professional summary for a trader."
        )
        output = self.invoke(prompt)
        return {
            "messages": {"role": "BinanceTracker", "content": output.content if hasattr(output, "content") else str(output)},
            "crypto_insight": output.content if hasattr(output, "content") else str(output)
        }

class RecommenderAgent(AgentNode):
    """Merges insights into a unified strategy."""
    @logging_message
    def exec(self, state: TradingState) -> dict:
        equity = state.get("equity_insight", "")
        crypto = state.get("crypto_insight", "")
        
        prompt = (
            f"Equity Analysis: \n{equity}\n\n"
            f"Crypto Analysis: \n{crypto}\n\n"
            f"Based on both markets, provide a unified multi-asset trading recommendation."
        )
        output = self.invoke(prompt)
        return {
            "messages": {"role": "Recommender", "content": output.content if hasattr(output, "content") else str(output)},
            "final_recommendation": output.content if hasattr(output, "content") else str(output)
        }

```

In there, `MarketDataProvider` and `BinanceTracker` are stateless with respect to each other, they only read the initial query from `state["messages"]` and write to separate keys. This is what makes safe parallelism possible. `RecommenderAgent` explicitly reads `equity_insight` and `crypto_insight` from state that it only runs after both are populated (enforced by the graph topology below).

## Assemble the Parallel Flow

Instantiating the agents with their LLM, tools, and instructions, then wiring them together into a directed graph using Vinagent's `FlowStateGraph`. The key here is the `START >> market_agent` and `START >> crypto_agent` edges, which tell the graph to launch both agents simultaneously from the initial state.

```python
from vnstock_tools import fetch_stock_data_vn
from vinagent.multi_agent import CrewAgent
from vinagent.graph.operator import FlowStateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

instr = "CRITICAL: Format tool arguments as JSON object. NEVER use plain strings."

market_agent = MarketDataProvider(
    name="market_agent", 
    llm=llm, 
    tools=[],
    instruction=instr
)

crypto_agent = BinanceTracker(
    name="crypto_agent", 
    llm=llm, 
    tools=['vinagent.tools.websearch_tools'], 
    instruction=instr
)

# 1. FIX: Give the recommender a specific instruction so it stops trying to act like a data agent
recommender = RecommenderAgent(
    name="recommender", 
    llm=llm, 
    tools=[], 
    instruction="You are a Portfolio Manager. Synthesize the provided market insights into a final recommendation." \
    " DO NOT fetch new data or use tools."
)

global_tools = [
    get_current_time,
    fetch_btc_market_data,
    primary_function(fetch_stock_data_vn)
]

# 2. FIX: Register tools to ALL agents as a safety net to prevent NoneType crashes
for agent in [market_agent, crypto_agent, recommender]:
    for tool in global_tools:
        agent.tools_manager.register_function_tool(tool)

print("Agents instantiated with standardized tool registration.")

# 3. Build the crew
crew = CrewAgent(
    llm=llm,
    checkpoint=MemorySaver(),
    graph=FlowStateGraph(TradingState),
    flow=[
        START >> market_agent,    # ─┐ Run in parallel
        START >> crypto_agent,    # ─┘
        market_agent >> recommender,   # ─┐ Both must complete
        crypto_agent >> recommender,   # ─┘ before recommender runs
        recommender >> END
    ]
    ]
)
print("Crew Assembled with Direct Parallel Flow.")
```

    INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
    INFO:vinagent.register.tool:Registered search_api:
    {'tool_name': 'search_api', 'arguments': 'query (Union[str, dict[str, str]])', 'return': 'The answer from search query', 'docstring': 'Search for an answer from a query string\n    Args:\n        query (dict[str, str]):  The input query to search\n    Returns:\n        The answer from search query', 'dependencies': ['os', 'dotenv', 'tavily', 'dataclasses', 'typing', 'vinagent'], 'module_path': 'vinagent.tools.websearch_tools', 'tool_type': 'module', 'tool_call_id': 'tool_d9ac2cfe-f13a-47be-a8cc-093bf0575ef'}
    INFO:vinagent.register.tool:Completed registration for module vinagent.tools.websearch_tools
    INFO:vinagent.register.tool:Registered tool: get_current_time (runtime)
    INFO:vinagent.register.tool:Registered tool: fetch_btc_market_data (runtime)
    INFO:vinagent.register.tool:Registered tool: fetch_stock_data_vn (runtime)
    INFO:vinagent.register.tool:Registered tool: get_current_time (runtime)
    INFO:vinagent.register.tool:Registered tool: fetch_btc_market_data (runtime)
    INFO:vinagent.register.tool:Registered tool: fetch_stock_data_vn (runtime)
    INFO:vinagent.register.tool:Registered tool: get_current_time (runtime)
    INFO:vinagent.register.tool:Registered tool: fetch_btc_market_data (runtime)
    INFO:vinagent.register.tool:Registered tool: fetch_stock_data_vn (runtime)


    Agents instantiated with standardized tool registration.
    Phase 2 Crew Assembled with Direct Parallel Flow.

Here is how the graph executes in technical viewpoint:

1. `START` fans out to `market_agent` and `crypto_agent` simultaneously.
2. The graph waits until both write their insights into `TradingState`.
3. recommender is unblocked and receives the fully populated state.
4. The final recommendation is written to `state["final_recommendation"]`.

## Execute Multi-Market Analysis

Run the system to see parallel processing in action.


```python
from IPython.display import display, Markdown

query = "Compare FPT and BTC performance for this week and suggest a 70/30 split strategy."

result = crew.invoke(
    query=query,
    user_id="analyst",
    thread_id=23
)
```
??? note "Crew Running Logs"
    INFO:vinagent.multi_agent.crew:No authentication card provided, skipping authentication
    INFO:vinagent.agent.agent:No authentication card provided, skipping authentication
    INFO:vinagent.agent.agent:I'am chatting with unknown_user
    INFO:vinagent.agent.agent:Tool calling iteration 1/10


    {'input': {'messages': {'role': 'user', 'content': 'Compare FPT and BTC performance for this week and suggest a 70/30 split strategy.'}}, 'config': {'configurable': {'user_id': 'analyst'}, 'thread_id': 23}}


    INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
    INFO:vinagent.agent.agent:Executing tool call: {'tool_name': 'fetch_btc_market_data', 'arguments': {}, 'return': "<class 'str'>", 'module_path': '__runtime__', 'tool_type': 'function', 'tool_call_id': 'tool_8fe201a1-7333-4d34-8260-b75a25e88b4', 'is_runtime': True}
    INFO:vinagent.register.tool:Completed executing function tool fetch_btc_market_data({})
    INFO:vinagent.agent.agent:Tool calling iteration 2/10
    INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
    INFO:vinagent.agent.agent:No more tool calls needed. Completed in 2 iterations.
    INFO:vinagent.logger.logger:
    
    ============ BinanceTracker Response ============
    {'messages': {'role': 'BinanceTracker', 'content': 'Bitcoin (BTC-USD) is currently trading at $72,480.28 USD. \n\n**Market Performance:**\n- **Daily Change:** -$210.97 (-0.29%)\n- **Weekly Change:** +$6,694.17 (+10.18%)\n- **Monthly Change:** +$5,439.97 (+8.11%)\n- **Yearly Change:** -17.20%\n\n**Volatility Analysis:**\n- The price has ranged between $67,469.42 and $74,036.08 in the last 24 hours.\n- The highest price reached in the last day was $73,264.09, while the lowest was $72,350.48.\n\n**Summary for Traders:**\nOverall, Bitcoin has shown a moderate recovery over the past week and month after facing significant losses over the year. The volatility remains evident with a daily fluctuation, suggesting both opportunities and risks for traders in the market.'}, 'crypto_insight': 'Bitcoin (BTC-USD) is currently trading at $72,480.28 USD. \n\n**Market Performance:**\n- **Daily Change:** -$210.97 (-0.29%)\n- **Weekly Change:** +$6,694.17 (+10.18%)\n- **Monthly Change:** +$5,439.97 (+8.11%)\n- **Yearly Change:** -17.20%\n\n**Volatility Analysis:**\n- The price has ranged between $67,469.42 and $74,036.08 in the last 24 hours.\n- The highest price reached in the last day was $73,264.09, while the lowest was $72,350.48.\n\n**Summary for Traders:**\nOverall, Bitcoin has shown a moderate recovery over the past week and month after facing significant losses over the year. The volatility remains evident with a daily fluctuation, suggesting both opportunities and risks for traders in the market.'}
    
    
    INFO:vinagent.agent.agent:No authentication card provided, skipping authentication
    INFO:vinagent.agent.agent:I'am chatting with unknown_user
    INFO:vinagent.agent.agent:Tool calling iteration 1/10
    INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
    INFO:vinagent.agent.agent:No more tool calls needed. Completed in 1 iterations.
    INFO:vinagent.logger.logger:
    
    ============ RecommenderAgent Response ============
    {'messages': {'role': 'Recommender', 'content': 'Based on the current market insights, the following multi-asset trading recommendation can be made:\n\n1. **Equities**: Given the recent volatility in the stock market, it would be prudent to adopt a cautious stance. Look for diversified equities that have shown moderate recovery and are less susceptible to short-term market fluctuations. Focus on sectors like technology and consumer goods that tend to perform well in recovery phases.\n\n2. **Cryptocurrency (Bitcoin)**: Bitcoin has demonstrated a recovery trend over the past week with a notable +10.18% increase, despite its yearly decline of -17.20%. Given the significant daily volatility, traders should consider short-term trading strategies, capitalizing on the price fluctuations within the range of $67,469.42 to $74,036.08. A cautious approach is advised, with potential allocations in Bitcoin but also setting clear exit strategies to manage risks.\n\nIn conclusion, maintain a balanced approach by including both equities and Bitcoin in your portfolio, while monitoring market movements closely and being prepared to adjust positions based on short-term trends.'}, 'final_recommendation': 'Based on the current market insights, the following multi-asset trading recommendation can be made:\n\n1. **Equities**: Given the recent volatility in the stock market, it would be prudent to adopt a cautious stance. Look for diversified equities that have shown moderate recovery and are less susceptible to short-term market fluctuations. Focus on sectors like technology and consumer goods that tend to perform well in recovery phases.\n\n2. **Cryptocurrency (Bitcoin)**: Bitcoin has demonstrated a recovery trend over the past week with a notable +10.18% increase, despite its yearly decline of -17.20%. Given the significant daily volatility, traders should consider short-term trading strategies, capitalizing on the price fluctuations within the range of $67,469.42 to $74,036.08. A cautious approach is advised, with potential allocations in Bitcoin but also setting clear exit strategies to manage risks.\n\nIn conclusion, maintain a balanced approach by including both equities and Bitcoin in your portfolio, while monitoring market movements closely and being prepared to adjust positions based on short-term trends.'}
    
    
Display the result:

```
display(Markdown(f"""
### 📈 Multi-Asset Trading Recommendation
---
{result['final_recommendation']}
"""))
```

??? note "Result of Multi-Asset Trading Recommendation"
    ### 📈 Multi-Asset Trading Recommendation
    ---
    Based on the current market insights, the following multi-asset trading recommendation can be made:

    1. **Equities**: Given the recent volatility in the stock market, it would be prudent to adopt a cautious stance. Look for diversified equities that have shown moderate recovery and are less susceptible to short-term market fluctuations. Focus on sectors like technology and consumer goods that tend to perform well in recovery phases.

    2. **Cryptocurrency (Bitcoin)**: Bitcoin has demonstrated a recovery trend over the past week with a notable +10.18% increase, despite its yearly decline of -17.20%. Given the significant daily volatility, traders should consider short-term trading strategies, capitalizing on the price fluctuations within the range of $67,469.42 to $74,036.08. A cautious approach is advised, with potential allocations in Bitcoin but also setting clear exit strategies to manage risks.

    In conclusion, maintain a balanced approach by including both equities and Bitcoin in your portfolio, while monitoring market movements closely and being prepared to adjust positions based on short-term trends.


## Single-market vs Multi-market Analysis

This table compares the result from the single-market analysis agent (on equity only) and multi-market analysis on the same query: `Compare FPT and BTC performance for this week and suggest a 70/30 split strategy.`.

| Single-Market Analysis (Equity Only) | Multi-Market Analysis (Equity + Crypto) |
|--------------------------------------|------------------------------------------|
| **Market Focus**<br><br> The equity market is experiencing **moderate volatility**, suggesting a cautious but constructive outlook.<br><br> **Key observations:**<br> - Technology stocks tend to perform well during early recovery phases.<br> - **FPT**, as a leading technology company, aligns well with this trend.<br><br> **Implications for this week:**<br> - FPT may provide **stable and predictable returns**.<br> - Strong long-term fundamentals in **digital transformation and IT services**.<br> - **Lower short-term volatility**, making it suitable as a **core portfolio allocation**. | **Cross-Market Perspective**<br><br> **1. Equities**<br> - Market volatility suggests maintaining a **cautious stance**.<br> - Prefer **diversified equities** with moderate recovery signals.<br> - Focus on sectors such as **technology** and **consumer goods**.<br><br> **2. Cryptocurrency (Bitcoin)**<br> - Bitcoin showed a **+10.18% weekly recovery**, despite **-17.20% yearly decline**.<br> - High daily volatility creates **short-term trading opportunities**.<br> - Observed trading range: **$67,469 – $74,036**.<br><br> **Strategy Insight**<br> - Combine **equity stability** with **crypto upside potential**.<br> - Maintain diversification and **clear exit strategies** to manage risk. |

While single-market analysis provides focused insights into equity performance, it may overlook opportunities emerging in other financial markets. Multi-market analysis expands the perspective by incorporating signals from both equities and cryptocurrencies. This broader view enables more diversified portfolio strategies, better risk balancing, and the ability to capture short-term opportunities such as Bitcoin’s recent volatility-driven gains. As a result, multi-market analysis produces more robust and adaptable investment recommendations, especially when designing strategies like a 70/30 portfolio allocation across different asset classes.


## Key Takeaways

1. Parallel execution reduces latency. Market analysis that would take N seconds sequentially now takes max(agent_time), which is dramatically faster as the number of data sources scales.

2. Shared state is the coordination mechanism. Agents communicate not by calling each other, but by writing to and reading from a typed state object. This makes the system easy to extend (add a new asset class → add a new agent and key).

3. Agent specialization improves quality. Each agent is given a narrow, well-defined role and is instructed to stay within it.

4. Cross-market recommendations are qualitatively richer. A single-asset system can only say "buy FPT." An active advisory system can say "allocate 70% to FPT for stability and 30% to BTC to capture crypto momentum, with a stop-loss at..." — a fundamentally more actionable insight.
