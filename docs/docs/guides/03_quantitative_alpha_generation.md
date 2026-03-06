# Advanced - Quantitative Alpha Generation

_Contributor: Gia Bao; Reviewed & Extended by: Kan Pham_

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datascienceworld-kan/vinagent/blob/main/cookbook/multi_agent_in_finance/03_quantitative_alpha_generation.ipynb)


In this phase, we implement a **Quantitative Refinement Loop**. This is a standard professional workflow where agents collaborate to profile data, propose predictive signals (alphas), verify the logic, and finally backtest the performance.

Imagine you are a quantitative analyst at a mid-sized hedge fund. Every morning, your team manually scans dozens of Vietnamese stocks like FPT, VNM, HPG, looking for patterns that might hint at a profitable trade. One analyst proposes a momentum signal. Another questions whether it will hold out-of-sample. A third runs a backtest over the weekend. By Monday, the market has already moved.

This is the bottleneck that kills alpha generation at scale: too many humans, too much latency, and no standardized pipeline to go from raw data to a validated, backtested trading signal.

The Vinagent library addresses this with a Quantitative Refinement Loop — a multi-agent pipeline where specialized AI agents collaborate in sequence, each performing one focused job: profiling the data, proposing a signal, verifying its logic, and running a simulation. What would take a team of analysts several days can now be orchestrated in a single Python script.

In this tutorial, we will walk through building this pipeline end-to-end, targeting the Vietnamese tech stock FPT as our case study.


## System Architecture
The pipeline is linear by design. Each agent receives the output of the previous one, so errors and biases compound the same way they do in real research workflows, and can be caught at each gate.

```
Data Profiler → Alpha Builder → Signal Verifier → Backtester Agent → Performance Report
```

| Agent | Role |
|---|---|
| **DataProfiler** | Extracts statistical features (volatility, skew, kurtosis) |
| **AlphaBuilder** | Proposes a mathematical trading signal |
| **SignalVerifier** | Checks for logic errors (overfitting, bias) |
| **BacktesterAgent** | Simulates performance and calculates Sharpe Ratio |

## Environment Setup

We use OpenAI as the LLM backbone and load credentials from a .env file. The `gpt-4o-mini` model is sufficient for reasoning-heavy tasks like signal verification and alpha proposal, keeping costs low while maintaining quality.

```python
%pip install -U vinagent==0.0.6.post6
%pip install --no-cache-dir "numpy<2.0" matplotlib==3.7.1
%pip install python-dotenv==1.0.0 tavily-python==0.7.7 plotly==5.22.0 Vnstock==3.4.2 -q
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


## Define Quant State

As previous tutorials, before writing any agents, we define a shared state object `QuantState` that acts as the single source of truth passed between agents. Think of it as the research dossier that gets handed from analyst to analyst down the pipeline.

Each field corresponds to a stage of the workflow: raw data profile, proposed alpha, verification findings, and final metrics. The messages field uses a custom `append_messages` reducer so that each agent can append to the conversation log without overwriting prior entries.

```python
import operator
from typing import Annotated, List, TypedDict

def append_messages(existing: list, update: dict) -> list:
    return existing + [update]

class QuantState(TypedDict):
    """Coherent state for the quantitative refinement loop."""
    # List reducers for automatic merging
    messages: Annotated[list[dict], append_messages]
    data_profile: str
    alpha_signal: str
    verification_report: str
    backtest_metrics: str
    final_report: str
```

## Implement Quantitative Agents

Each agent is a subclass of `AgentNode`, overriding the exec method to define its specific job. The `@logging_message` decorator captures each agent's output automatically for debugging and audit trails, essential in any production quant workflow.

Notice that `AlphaBuilder` and `SignalVerifier` receive the instruction `no_tool_instr`, which explicitly forbids them from calling external tools. This is intentional: these two agents are pure reasoning agents. Giving them tool access would introduce latency and unnecessary API calls when all they need to do is think critically about the data already in state.

```python
from vinagent.multi_agent import AgentNode
from vinagent.logger.logger import logging_message

class DataProfiler(AgentNode):
    """Analyzes statistical properties of the asset."""
    @logging_message
    def exec(self, state: QuantState) -> dict:
        print(f"[{self.name}] Analyzing market data and profiling distributions...")
        symbol = state.get("symbol", "FPT")
        prompt = f"""
        Calculate statistical metrics for {symbol} over the last 2 years.
        Use the `calculate_stock_statistics` tool.
        Explain the implications of volatility, skewness, and drawdown.
        """
        output = self.invoke(prompt, max_iterations=5)
        return {
            "messages": [{"role": "assistant", "content": output.content if hasattr(output, "content") else str(output)}],
            "data_profile": output.content if hasattr(output, "content") else str(output)
        }

class AlphaBuilder(AgentNode):
    """Proposes a mathematical signal based on the data profile."""
    @logging_message
    def exec(self, state: QuantState) -> dict:
        print(f"[{self.name}] Proposing trading alpha based on profile...")
        profile = state.get("data_profile", "")
        prompt = f"""
        Based on this data profile: \n{profile}\n\n
        Propose a specific trading strategy ('momentum' or 'mean_reversion') and parameters (e.g., window size).
        Explain why it fits the profile. Format your choice clearly for the next agent.
        """
        output = self.invoke(prompt, max_iterations=3)
        return {
            "messages": [{"role": "assistant", "content": output.content if hasattr(output, "content") else str(output)}],
            "alpha_signal": output.content if hasattr(output, "content") else str(output)
        }

class SignalVerifier(AgentNode):
    """Challenges the logic to prevent overfitting."""
    @logging_message
    def exec(self, state: QuantState) -> dict:
        print(f"[{self.name}] Verifying alpha logic...")
        signal = state.get("alpha_signal", "")
        prompt = f"Validate this alpha signal for common hedge fund pitfalls: \n{signal}\n\nProvide a verification report and approve or suggest adjustments."
        output = self.invoke(prompt, max_iterations=3)
        return {
            "messages": [{"role": "assistant", "content": output.content if hasattr(output, "content") else str(output)}],
            "verification_report": output.content if hasattr(output, "content") else str(output)
        }

class BacktesterAgent(AgentNode):
    """Simulates the signal and provides Sharpe/Drawdown metrics."""
    @logging_message
    def exec(self, state: QuantState) -> dict:
        print(f"[{self.name}] Running backtest simulation...")
        symbol = state.get("symbol", "FPT")
        signal = state.get("alpha_signal", "")
        prompt = f"""
        Simulate the strategy described here: \n{signal}\n\n

        Identify the symbol ({symbol}), strategy_type (momentum/mean_reversion), and parameters.
        Use the `backtest_alpha_strategy` tool to get metrics.
        
        CRITICAL JSON RULE: If passing arguments to `kwargs`, format it strictly as JSON using double quotes. 
        Example: {{"window_size": 20}}
        DO NOT use single quotes.
        
        Provide a final performance report in Markdown.
        """
        output = self.invoke(prompt, max_iterations=5)
        return {
            "messages": [{"role": "assistant", "content": output.content if hasattr(output, "content") else str(output)}],
            "backtest_metrics": output.content if hasattr(output, "content") else str(output)
        }

```

## Assemble the Refinement Pipeline

Here we wire everything together. Tools are registered once and shared across all agents that need them. The `primary_function` decorator wraps raw functions so Vinagent can introspect their signatures and expose them as callable tools.

The `FlowStateGraph` with a linear flow list enforces strict sequencing. Unlike a parallel or conditional graph, every agent in this pipeline waits for its predecessor to complete before running. This mirrors how a real research review process works: you do not backtest a signal before verifying it.



```python
from vinagent.register import primary_function

from vnstock_tools import (
    fetch_stock_data_vn as fetch_stock_data, 
    visualize_stock_data_vn as visualize_stock_data, 
    plot_returns_vn as plot_returns,
    calculate_stock_statistics,
    backtest_alpha_strategy
)
from vinagent.tools.websearch_tools import search_api

@primary_function
def get_current_time() -> str:
    """Get current system time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print("Core Infrastructure Initialized (Direct Registration).")
```

    Core Infrastructure Initialized (Direct Registration).

Registering tools for agents:

```python
from vinagent.multi_agent import CrewAgent
from vinagent.graph.operator import FlowStateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from vinagent.register import primary_function

# 1. Instantiate the Agents
instr = 'CRITICAL: Format tool arguments as strictly valid JSON. You MUST use double quotes (") for all property names and string values. NEVER use single quotes (\').'
no_tool_instr = instr + " IMPORTANT: DO NOT use any tools. Your only job is to reason and output text."

profiler = DataProfiler(name="profiler", llm=llm, instruction=instr)
builder = AlphaBuilder(name="builder", llm=llm, instruction=no_tool_instr)
verifier = SignalVerifier(name="verifier", llm=llm, instruction=no_tool_instr)
backtester = BacktesterAgent(name="backtester", llm=llm, instruction=instr)

all_tools = [
    primary_function(fetch_stock_data), 
    primary_function(visualize_stock_data), 
    primary_function(calculate_stock_statistics),
    get_current_time,
    primary_function(backtest_alpha_strategy)
]

for agent in [profiler, builder, verifier, backtester]:
    for tool in all_tools:
        agent.tools_manager.register_function_tool(tool)

print("Agents instantiated and tools registered with safety net.")

# 3. Assemble the Linear Pipeline (Crew)
crew = CrewAgent(
    llm=llm,
    checkpoint=MemorySaver(),
    graph=FlowStateGraph(QuantState),
    flow=[
        START >> profiler,
        profiler >> builder,
        builder >> verifier,
        verifier >> backtester,
        backtester >> END
    ]
)
print("Phase 3 Quant Pipeline Assembled successfully.")
```

    INFO:vinagent.register.tool:Registered tool: fetch_stock_data_vn (runtime)
    INFO:vinagent.register.tool:Registered tool: visualize_stock_data_vn (runtime)
    INFO:vinagent.register.tool:Registered tool: calculate_stock_statistics (runtime)
    INFO:vinagent.register.tool:Registered tool: get_current_time (runtime)
    INFO:vinagent.register.tool:Registered tool: backtest_alpha_strategy (runtime)
    INFO:vinagent.register.tool:Registered tool: fetch_stock_data_vn (runtime)
    INFO:vinagent.register.tool:Registered tool: visualize_stock_data_vn (runtime)
    INFO:vinagent.register.tool:Registered tool: calculate_stock_statistics (runtime)
    INFO:vinagent.register.tool:Registered tool: get_current_time (runtime)
    INFO:vinagent.register.tool:Registered tool: backtest_alpha_strategy (runtime)
    INFO:vinagent.register.tool:Registered tool: fetch_stock_data_vn (runtime)
    INFO:vinagent.register.tool:Registered tool: visualize_stock_data_vn (runtime)
    INFO:vinagent.register.tool:Registered tool: calculate_stock_statistics (runtime)
    INFO:vinagent.register.tool:Registered tool: get_current_time (runtime)
    INFO:vinagent.register.tool:Registered tool: backtest_alpha_strategy (runtime)
    INFO:vinagent.register.tool:Registered tool: fetch_stock_data_vn (runtime)
    INFO:vinagent.register.tool:Registered tool: visualize_stock_data_vn (runtime)
    INFO:vinagent.register.tool:Registered tool: calculate_stock_statistics (runtime)
    INFO:vinagent.register.tool:Registered tool: get_current_time (runtime)
    INFO:vinagent.register.tool:Registered tool: backtest_alpha_strategy (runtime)


    Agents instantiated and tools registered with safety net.
    Phase 3 Quant Pipeline Assembled successfully.


## Run the Search for Alpha

With the pipeline assembled, we kick off the workflow by passing an initial state containing our target symbol and a natural-language query. The `user_id` and `thread_id` parameters enable Vinagent's checkpoint system to persist state across runs — useful when you want to resume a failed pipeline or audit a previous run.


```python
initial_state = {"symbol": "FPT"}
result = crew.invoke(query="Find a momentum-based alpha for FPT.", user_id='admin', thread_id=67)

print("\n--- FINAL QUANT REPORT ---\n")
print(result.get("backtest_metrics", "No metrics found in state."))
```

??? note "Crew Running Logs"
    INFO:vinagent.agent.agent:Tool calling iteration 1/5
    {'input': {'messages': {'role': 'user', 'content': 'Find a momentum-based alpha for FPT.'}}, 'config': {'configurable': {'user_id': 'admin'}, 'thread_id': 67}}
    [profiler] Analyzing market data and profiling distributions...
    INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
    INFO:vinagent.agent.agent:No more tool calls needed. Completed in 1 iterations.
    INFO:vinagent.logger.logger:

    ============ DataProfiler Response ============
    {'messages': [{'role': 'assistant', 'content': "To calculate the statistical metrics for FPT over the last 2 years, I will first need to fetch the historical stock data for FPT. Then, I will use that data to calculate the required statistics.\n\nLet's start by fetching the stock data for FPT.\n\nI will assume the date range is from today back to 2 years ago and use a daily interval for the data.\n\nFirst, I will get the current date and compute the start date."}], 'data_profile': "To calculate the statistical metrics for FPT over the last 2 years, I will first need to fetch the historical stock data for FPT. Then, I will use that data to calculate the required statistics.\n\nLet's start by fetching the stock data for FPT.\n\nI will assume the date range is from today back to 2 years ago and use a daily interval for the data.\n\nFirst, I will get the current date and compute the start date."}


    ============ AlphaBuilder Response ============
    {'messages': [{'role': 'assistant', 'content': 'To propose a specific trading strategy for FPT, I recommend using a "momentum" strategy with a window size of 20 days.\n\nThis strategy is suitable because momentum trading aims to capture the continuation of existing trends over short to medium timeframes. Given that FPT has shown significant price movements historically, leveraging a 20-day lookback period can help capitalize on these trends effectively. \n\nBy focusing on this window size, the strategy can identify and follow the prevailing upward or downward trends, potentially leading to profitable trading opportunities.'}], 'alpha_signal': 'To propose a specific trading strategy for FPT, I recommend using a "momentum" strategy with a window size of 20 days.\n\nThis strategy is suitable because momentum trading aims to capture the continuation of existing trends over short to medium timeframes. Given that FPT has shown significant price movements historically, leveraging a 20-day lookback period can help capitalize on these trends effectively. \n\nBy focusing on this window size, the strategy can identify and follow the prevailing upward or downward trends, potentially leading to profitable trading opportunities.'}


    ============ SignalVerifier Response ============
    {'messages': [{'role': 'assistant', 'content': "To validate the proposed momentum trading strategy for FPT, it's essential to consider several common pitfalls often encountered in hedge fund strategies. Here’s a verification report:\n\n1. **Market Trends**: Ensure that the current market conditions reflect a strong trend. A momentum strategy thrives in trending markets; if the market is ranging, the strategy may underperform.\n\n2. **Window Size**: The choice of a 20-day lookback period is generally reasonable for capturing short to medium-term price movements. However, it should be validated against historical performance to determine if this window size has historically yielded profitable signals for FPT.\n\n3. **Volatility**: Assess the volatility of FPT. High volatility can lead to false signals in momentum trading, potentially resulting in losses.\n\n4. **Liquidity Considerations**: Ensure that FPT has sufficient liquidity to enter and exit positions without significant slippage.\n\n5. **Risk Management**: Implement risk management measures, such as stop-loss orders, to mitigate potential losses from sudden market reversals.\n\n6. **Data Quality**: Verify that the historical price data used for backtesting is clean and free of errors, as inaccuracies can lead to misleading results.\n\n7. **Overfitting**: Be cautious of overfitting the strategy to historical data, which can result in poor future performance.\n\n8. **Transaction Costs**: Consider the impact of transaction costs on the performance of the strategy, especially for high-frequency trading.\n\nRecommendation: Backtest the momentum strategy using historical data to evaluate its performance under various market conditions. If the results are promising, proceed with caution and consider starting with a small allocation. Additionally, monitor the market conditions closely and be prepared to adjust the strategy if necessary.\n\nIf you would like to perform a backtest or gather historical data for further analysis, please let me know!"}], 'verification_report': "To validate the proposed momentum trading strategy for FPT, it's essential to consider several common pitfalls often encountered in hedge fund strategies. Here’s a verification report:\n\n1. **Market Trends**: Ensure that the current market conditions reflect a strong trend. A momentum strategy thrives in trending markets; if the market is ranging, the strategy may underperform.\n\n2. **Window Size**: The choice of a 20-day lookback period is generally reasonable for capturing short to medium-term price movements. However, it should be validated against historical performance to determine if this window size has historically yielded profitable signals for FPT.\n\n3. **Volatility**: Assess the volatility of FPT. High volatility can lead to false signals in momentum trading, potentially resulting in losses.\n\n4. **Liquidity Considerations**: Ensure that FPT has sufficient liquidity to enter and exit positions without significant slippage.\n\n5. **Risk Management**: Implement risk management measures, such as stop-loss orders, to mitigate potential losses from sudden market reversals.\n\n6. **Data Quality**: Verify that the historical price data used for backtesting is clean and free of errors, as inaccuracies can lead to misleading results.\n\n7. **Overfitting**: Be cautious of overfitting the strategy to historical data, which can result in poor future performance.\n\n8. **Transaction Costs**: Consider the impact of transaction costs on the performance of the strategy, especially for high-frequency trading.\n\nRecommendation: Backtest the momentum strategy using historical data to evaluate its performance under various market conditions. If the results are promising, proceed with caution and consider starting with a small allocation. Additionally, monitor the market conditions closely and be prepared to adjust the strategy if necessary.\n\nIf you would like to perform a backtest or gather historical data for further analysis, please let me know!"}


    ============ BacktesterAgent Response ============
    {'messages': [{'role': 'assistant', 'content': 'To proceed with the trading strategy for FPT using a momentum approach with a 20-day window size, I will first need to backtest the strategy. I will use the `backtest_alpha_strategy` tool with the specified parameters. \n\nLet\'s get started by backtesting the strategy. I will fetch the historical stock data for FPT and define the necessary parameters.\n\n### Parameters:\n- **Symbol**: "FPT"\n- **Strategy Type**: "momentum"\n- **Start Date**: I\'ll determine a suitable start date for our analysis.\n- **End Date**: I\'ll also determine a suitable end date.\n\nLet me fetch the current date to define the time period for our backtest. After that, I\'ll formulate the backtest request.'}], 'backtest_metrics': 'To proceed with the trading strategy for FPT using a momentum approach with a 20-day window size, I will first need to backtest the strategy. I will use the `backtest_alpha_strategy` tool with the specified parameters. \n\nLet\'s get started by backtesting the strategy. I will fetch the historical stock data for FPT and define the necessary parameters.\n\n### Parameters:\n- **Symbol**: "FPT"\n- **Strategy Type**: "momentum"\n- **Start Date**: I\'ll determine a suitable start date for our analysis.\n- **End Date**: I\'ll also determine a suitable end date.\n\nLet me fetch the current date to define the time period for our backtest. After that, I\'ll formulate the backtest request.'}


??? note "Final Quant Report"
    --- FINAL QUANT REPORT ---

    To proceed with the trading strategy for FPT using a momentum approach with a 20-day window size, I will first need to backtest the strategy. I will use the `backtest_alpha_strategy` tool with the specified parameters. 

    Let's get started by backtesting the strategy. I will fetch the historical stock data for FPT and define the necessary parameters.

    ### Parameters:
    - **Symbol**: "FPT"
    - **Strategy Type**: "momentum"
    - **Start Date**: I'll determine a suitable start date for our analysis.
    - **End Date**: I'll also determine a suitable end date.

    Let me fetch the current date to define the time period for our backtest. After that, I'll formulate the backtest request.

## Conclusion
The `Vinagent Quantitative Refinement Loop` demonstrates a concrete and practical pattern for automating the most time-consuming parts of quant research. Rather than treating the LLM as a single black box that does everything, the pipeline assigns each agent a narrow, auditable responsibility: `profile, propose, verify, simulate`.

A few things are worth internalizing from this tutorial. First, the division between tool-calling agents (`profiler, backtester`) and pure-reasoning agents (`builder, verifier`) is a deliberate architectural choice that keeps costs low and outputs focused.

Second, the shared `QuantState` object is what makes the pipeline coherent, every agent reads from and writes to the same dossier, so there is no ambiguity about what data is in scope at each step.

Third, the explicit JSON formatting instruction inside `BacktesterAgent` is a small but important detail: LLMs in tool-calling contexts will occasionally produce malformed arguments, and a single line of prompt guidance prevents an otherwise confusing failure.

From here, natural extensions include adding a conditional edge that routes back to `AlphaBuilder` when `SignalVerifier` rejects a signal, integrating a portfolio construction layer after the backtest, or parallelizing the pipeline to evaluate multiple symbols simultaneously. The linear graph structure used here is the foundation, everything more complex builds on top of it.
