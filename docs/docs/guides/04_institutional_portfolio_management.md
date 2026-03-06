# Super Advanced - Institutional Portfolio Management

_Contributor: Gia Bao; Reviewed & Extended by: Kan Pham_

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datascienceworld-kan/vinagent/blob/main/cookbook/multi_agent_in_finance/03_quantitative_alpha_generation.ipynb)


There's a quiet but important distinction between how retail investors and institutional portfolio managers (PM) think about allocation. A retail investor might ask: "Which stock should I buy?" An institutional PM asks something far more nuanced: "Given what the market collectively believes, and given my firm's specific views, what is the optimal allocation, and can I defend it to my investment committee?"

That distinction is not just philosophical. It shapes the entire architecture of how decisions get made.
Most algorithmic trading tutorials teach you to optimize a portfolio once, output some weights, and call it done. But real institutional workflows don't work that way. They iterate. They incorporate expert judgment. They have review gates. They produce portfolios that are not only mathematically optimal but explainable and auditable.
This tutorial builds exactly that kind of system using Vinagent, a multi-agent framework, and applies it to a Tech-Heavy portfolio in the Vietnam Stock Market. By the end, you'll have a working pipeline that:

- Derives what the market implicitly believes through equilibrium returns
- Blends that market consensus with your own expert views using the Black-Litterman model
- Optimizes portfolio weights using mean-variance optimization
- Routes the result through a simulated Portfolio Manager who can reject the proposal and trigger a revision loop

Let's unpack why each of these pieces matters before we write a single line of code.

### The Core Idea: Markets Are Smarter Than Any Single Model

Before diving into code, it's worth understanding the intellectual foundation here. Classic mean-variance optimization, the kind taught in every finance textbook, has a well-known problem: it's brutally sensitive to inputs. Feed it slightly different return estimates and you get wildly different allocations. In practice, this often produces portfolios that are mathematically optimal but practically absurd: 80% in a single asset, near-zero weights in most others.

The Black-Litterman model was developed at Goldman Sachs in the early 1990s specifically to fix this. Its key insight is elegant: instead of guessing expected returns from scratch, start from what the market already believes.

The logic goes like this. If you take the current market-cap weights of all assets and reverse-engineer what expected returns would justify those weights under an efficient market assumption, you get what are called equilibrium returns, the market's implicit view on each asset's risk-adjusted return. These become your prior beliefs.

Then, you layer in your own subjective views. Maybe your research team believes FPT will outperform the sector by 5%. Maybe you're cautious on VGI given recent regulatory changes. Black-Litterman gives you a principled, mathematically rigorous way to blend those views with the market prior, producing a posterior estimate that respects both the wisdom of market prices and the value of expert judgment.

The result: more stable, more diversified allocations that don't blow up when inputs change slightly.

### System Architecture

Automating the math is the easy part. The harder question is: who checks the machine?
In institutional investing, no algorithmic output goes directly to execution. There is always a Portfolio Manager review step. This isn't bureaucracy for its own sake, it's risk management. Models can be well-calibrated and still produce outputs that violate constraints the model didn't know about: regulatory limits, client mandates, liquidity considerations, or simply the PM's judgment that the macro environment makes a particular bet unwise right now.

The Human-in-the-Loop (HITL) pattern we implement here captures this dynamic. After the optimizer produces weights, a review agent evaluates them against intuitive criteria. In this case, diversification. If the portfolio is too concentrated, the review agent doesn't just reject it; it provides specific feedback that gets fed back into the Black-Litterman step, forcing a revised set of views and a new optimization cycle.

This creates a feedback loop that mirrors real institutional workflows:

```
START → Data Agent → Market Posterior (BL) → Optimizer → Human Review
                                 ↑                          |
                                 └------- [Rejected] -------┘
                                 |
                                 └------- [Approved] -------→ END
```

The system iterates until the PM is satisfied. This is not a workaround, it's the design.

Before we write code, here's how the agents divide responsibilities:

| Agent | Role | Logic |
|---|---|---|
| **InstitutionalDataAgent** | Fetches market equilibrium (Priors) | Sequential |
| **MarketPosteriorAgent** | Merges market data with subjective Views | Iterative |
| **PortfolioOptimizer** | Generates target weights | Sequential |
| **HumanReviewAgent** | Simulates a Portfolio Manager's approval | Branching |

Each agent is a node in a directed graph. The edges between them define the workflow. Most edges are unconditional (Agent A always feeds Agent B), but the edge out of `HumanReviewAgent` is conditional, it either routes to `END` (approved) or loops back to `MarketPosteriorAgent` (rejected, with revised views).

## Environment Setup

Install dependencies and initialize the LLM.

```python
%pip install -U vinagent==0.0.6.post7
%pip install --no-cache-dir "numpy<2.0" matplotlib==3.7.1
%pip install python-dotenv==1.0.0 tavily-python==0.7.7 plotly==5.22.0 Vnstock==3.4.2 -q
```

```
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv('.env'))

llm = ChatOpenAI(model="gpt-4o-mini")
print("LLM initialized.")
```


## Define Institutional State

In a multi-agent system, agents need to share information. Vinagent (built on LangGraph) uses a typed state dictionary that gets passed between nodes, with each node reading what it needs and writing back what it produces.

Think of this as the system's working memory. Every piece of information, market priors, posterior returns, optimal weights, approval status, and PM feedback lives here.

```python
import operator
from typing import Annotated, List, TypedDict

def append_messages(existing: list, update: dict) -> list:
    return existing + [update]

class InstitutionalState(TypedDict):
    """State for professional institutional rebalancing."""
    messages: Annotated[list[dict], append_messages]
    market_priors: str
    optimal_weights: str
    approval_status: str
    pm_feedback: str
```

The `pm_feedback` field is especially important: when the review agent rejects a proposal, its feedback is stored here and picked up by the `MarketPosteriorAgent` in the next iteration. This is how the feedback loop propagates information.


## Implement Institutional Agents

**The Data Agent: Grounding in Market Reality**

The `InstitutionalDataAgent` calculates equilibrium returns, the implied returns that would justify current market-cap weights if the market were efficient. For our Vietnam tech portfolio, we use five tickers: FPT, CMG, VGI, ELC, and ITD.
These equilibrium returns are our prior beliefs, the starting point before any expert views are applied. Using market-implied priors rather than arbitrary return forecasts is what makes Black-Litterman more robust than naive optimization.

```python
from vinagent.multi_agent import AgentNode
from vinagent.logger.logger import logging_message

class InstitutionalDataAgent(AgentNode):
    """Fetches market equilibrium (Priors)."""
    @logging_message
    def exec(self, state: InstitutionalState) -> dict:
        print(f"[{self.name}] Fetching market priors...")
        prompt = """
        Calculate implied market returns (priors) for a Tech-Heavy portfolio in the Vietnam Stock Market.
        CRITICAL: You MUST use these exact Vietnamese tickers: ['FPT', 'CMG', 'VGI', 'ELC', 'ITD'].
        Use the `calculate_equilibrium_returns` tool.
        """
        output = self.invoke(prompt)
        return {
            "messages": [{"role": "assistant", "content": output.content if hasattr(output, "content") else str(output)}],
            "market_priors": output.content if hasattr(output, "content") else str(output)
        }
```

**The Posterior Agent: Blending Knowledge with Judgment**

This is where Black-Litterman does its work. The MarketPosteriorAgent takes the equilibrium priors and combines them with subjective views — either the default (no strong views) or the revised views from a rejected proposal.

Notice how `pm_feedback` flows in here. On the first pass, it's empty. If the PM rejects and says "favor FPT heavily," that instruction becomes the input views for the next Black-Litterman calculation. The model is thus _learning from its own review process._


```python
class MarketPosteriorAgent(AgentNode):
    """Merges market data with subjective Views using Black-Litterman."""
    @logging_message
    def exec(self, state: InstitutionalState) -> dict:
        print(f"[{self.name}] Applying Black-Litterman model...")
        priors = state.get("market_priors", "")
        feedback = state.get("pm_feedback", "No previous feedback. Assume standard positive views on tech.")
        
        prompt = f"""
        Priors: {priors}
        PM Feedback/Views: {feedback}
        
        Calculate Black-Litterman posterior returns using the `calculate_black_litterman` tool.
        Make sure to apply the views mentioned in the PM Feedback.
        """
        output = self.invoke(prompt)
        return {
            "messages": [{"role": "assistant", "content": output.content if hasattr(output, "content") else str(output)}],
            "market_priors": output.content if hasattr(output, "content") else str(output) 
        }
```

**The Optimizer: Translating Returns into Allocations**
Given the posterior expected returns, the `PortfolioOptimizer` solves for weights that maximize risk-adjusted return. The `risk_aversion` parameter controls how aggressively the optimizer concentrates in high-returning assets, higher values produce more diversified portfolios.

```python
class PortfolioOptimizer(AgentNode):
    """Performs mean-variance optimization."""
    @logging_message
    def exec(self, state: InstitutionalState) -> dict:
        print(f"[{self.name}] Generating optimal weights...")
        returns = state.get("market_priors", "")
        
        prompt = f"""
        Calculate optimal weights based on these returns: {returns}
        Use the `optimize_portfolio` tool.
        """
        output = self.invoke(prompt)
        return {
            "messages": [{"role": "assistant", "content": output.content if hasattr(output, "content") else str(output)}],
            "optimal_weights": output.content if hasattr(output, "content") else str(output)
        }
```

**The Human Review Agent: The Governance Gate**
This is the most conceptually interesting agent. Rather than executing a mathematical operation, it exercises judgment. It evaluates the proposed weights against a qualitative criterion: is this allocation sufficiently diversified for an institutional mandate?

The rule is simple but meaningful: no single stock should exceed 40% of the portfolio. If the allocation fails this test, the agent explains why it's rejecting and what change it wants, effectively writing the views for the next BL iteration.

```python
class HumanReviewAgent(AgentNode):
    """Simulates a Portfolio Manager's approval."""
    @logging_message
    def exec(self, state: InstitutionalState) -> dict:
        print(f"[{self.name}] Reviewing portfolio weights...")
        weights = state.get("optimal_weights", "")
        
        prompt = f"""
        Review these proposed weights: {weights}. 
        Do they look reasonable for an institutional tech-heavy portfolio in Vietnam? 
        If the weights are sufficiently diversified (no single stock > 40%), explicitly say 'APPROVED'. 
        If they are poorly distributed or completely equal (e.g., all 0.20), explicitly say 'REJECTED' and instruct the BL agent to apply stronger subjective views (e.g., favor FPT heavily) to fix the distribution.
        """
        output = self.invoke(prompt)
        content = output.content if hasattr(output, "content") else str(output)
        
        status = "Approved" if "APPROVED" in content.upper() else "Rejected"
        
        return {
            "messages": [{"role": "assistant", "content": content}],
            "approval_status": status,
            "pm_feedback": content
        }
```


## Assemble the Graph

Now we wire everything together. The key design decision is the conditional edge out of `HumanReviewAgent`: it reads `approval_status` and routes either to `END` or back to `bl_agent`.

**Initialize Tools**

```python
# Tools imported directly
from vinagent.tools.yfinance_tools import fetch_stock_data, visualize_stock_data, plot_returns
from vinagent.tools.websearch_tools import search_api
from customize_tools import calculate_black_litterman, optimize_portfolio, calculate_equilibrium_returns

@primary_function
def get_current_time() -> str:
    """Get current system time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print("Institutional Core Infrastructure Initialized (Direct Registration).")

```

    Institutional Core Infrastructure Initialized (Direct Registration).


**Initialize Agents**
```python
instr = 'CRITICAL: Format tool arguments as strictly valid JSON. Use double quotes (") for all property names and string values.'
no_tool_instr = "You are a Portfolio Manager evaluating output. DO NOT use tools. Just read and respond."

data_agent = InstitutionalDataAgent(name="data_agent", llm=llm, instruction=instr)
bl_agent = MarketPosteriorAgent(name="bl_agent", llm=llm, instruction=instr)
optimizer = PortfolioOptimizer(name="optimizer", llm=llm, instruction=instr)
human_review = HumanReviewAgent(name="human_review", llm=llm, instruction=no_tool_instr)

print("Phase 4 Agents defined and instantiated successfully!")
```

    Phase 4 Agents defined and instantiated successfully!


**Initialize Crew of Agents**

```python
global_tools = [
    primary_function(fetch_stock_data),
    primary_function(visualize_stock_data),
    primary_function(plot_returns),
    primary_function(search_api),
    primary_function(calculate_black_litterman),
    primary_function(optimize_portfolio),
    primary_function(calculate_equilibrium_returns),
    get_current_time
]

for agent in [data_agent, bl_agent, optimizer, human_review]:
    for tool in global_tools:
        agent.tools_manager.register_function_tool(tool)

from vinagent.multi_agent import CrewAgent
from vinagent.graph.operator import FlowStateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

def human_review_router(state):
    """Routes back to the Black-Litterman agent if rejected, or ends if approved."""
    if state.get("approval_status") == "Approved":
        return END
    return bl_agent.name

institutional_graph = FlowStateGraph(InstitutionalState)
institutional_graph.add_conditional_edges(human_review.name, human_review_router)

crew = CrewAgent(
    llm=llm,
    checkpoint=MemorySaver(),
    graph=institutional_graph,
    flow=[
        START >> data_agent,
        data_agent >> bl_agent,
        bl_agent >> optimizer,
        optimizer >> human_review
    ]
)

print("Phase 4 HITL Crew Assembled successfully!")
```


    Phase 4 HITL Crew Assembled successfully!

A note on `MemorySaver`: this enables checkpointing, meaning the state is persisted at each node. In a production system, you'd replace this with a database-backed checkpoint store, allowing workflows to be paused, inspected, and resumed across sessions.


## Execute Institutional Rebalance

The system will iterate until the Human Review node grants approval.


```python
from IPython.display import display, Markdown

query = "Perform an institutional rebalance for the Tech-Heavy portfolio in Vietnam Stock Market. Ensure it goes through PM review."

result = crew.invoke(query=query, user_id="admin", thread_id=10)

display(Markdown(f"""
## 🏦 Institutional Portfolio Advice
---
**Approval Status:** {result.get('approval_status', 'Unknown')}

**Optimized Allocation Insights:**
{result.get('optimal_weights', 'No data collected.')}
"""))
```

??? note "Execution Logs"
    ============ InstitutionalDataAgent Response ============
    {'messages': [{'role': 'assistant', 'content': "The implied market returns for a Tech-Heavy portfolio in the Vietnam Stock Market using the tickers ['FPT', 'CMG', 'VGI', 'ELC', 'ITD'] are as follows:\n\n- FPT: 10.7%\n- CMG: 10.0%\n- VGI: 10.0%\n- ELC: 10.0%\n- ITD: 10.0%"}], 'market_priors': "The implied market returns for a Tech-Heavy portfolio in the Vietnam Stock Market using the tickers ['FPT', 'CMG', 'VGI', 'ELC', 'ITD'] are as follows:\n\n- FPT: 10.7%\n- CMG: 10.0%\n- VGI: 10.0%\n- ELC: 10.0%\n- ITD: 10.0%"}
    
    INFO:vinagent.agent.agent:Tool calling iteration 1/10


    [bl_agent] Applying Black-Litterman model...


    INFO:vinagent.agent.agent:Executing tool call: {'tool_name': 'calculate_black_litterman', 'tool_type': 'function', 'arguments': {'symbols': ['FPT', 'CMG', 'VGI', 'ELC', 'ITD'], 'priors': [10.7, 10.0, 10.0, 10.0, 10.0], 'views': [], 'tau': 0.05}, 'module_path': '__runtime__'}
    INFO:vinagent.register.tool:Completed executing function tool calculate_black_litterman({'symbols': ['FPT', 'CMG', 'VGI', 'ELC', 'ITD'], 'priors': [10.7, 10.0, 10.0, 10.0, 10.0], 'views': [], 'tau': 0.05})

    
    ============ MarketPosteriorAgent Response ============
    {'messages': [{'role': 'assistant', 'content': 'The Black-Litterman posterior returns for the Tech-Heavy portfolio in the Vietnam Stock Market are as follows:\n\n- FPT: 10.7%\n- CMG: 10.0%\n- VGI: 10.0%\n- ELC: 10.0%\n- ITD: 10.0%'}], 'market_priors': 'The Black-Litterman posterior returns for the Tech-Heavy portfolio in the Vietnam Stock Market are as follows:\n\n- FPT: 10.7%\n- CMG: 10.0%\n- VGI: 10.0%\n- ELC: 10.0%\n- ITD: 10.0%'}

    INFO:vinagent.agent.agent:Tool calling iteration 1/10


    [optimizer] Generating optimal weights...


    INFO:vinagent.agent.agent:Executing tool call: {'tool_name': 'optimize_portfolio', 'tool_type': 'function', 'arguments': {'symbols': ['FPT', 'CMG', 'VGI', 'ELC', 'ITD'], 'expected_returns': {'FPT': 0.107, 'CMG': 0.1, 'VGI': 0.1, 'ELC': 0.1, 'ITD': 0.1}, 'risk_aversion': 2.0}, 'module_path': '__runtime__'}
    INFO:vinagent.register.tool:Completed executing function tool optimize_portfolio({'symbols': ['FPT', 'CMG', 'VGI', 'ELC', 'ITD'], 'expected_returns': {'FPT': 0.107, 'CMG': 0.1, 'VGI': 0.1, 'ELC': 0.1, 'ITD': 0.1}, 'risk_aversion': 2.0})
    
    ============ PortfolioOptimizer Response ============
    {'messages': [{'role': 'assistant', 'content': 'The optimal weights for the Tech-Heavy portfolio in the Vietnam Stock Market based on the Black-Litterman posterior returns are as follows:\n\n- FPT: 21.1%\n- CMG: 19.72%\n- VGI: 19.72%\n- ELC: 19.72%\n- ITD: 19.72%'}], 'optimal_weights': 'The optimal weights for the Tech-Heavy portfolio in the Vietnam Stock Market based on the Black-Litterman posterior returns are as follows:\n\n- FPT: 21.1%\n- CMG: 19.72%\n- VGI: 19.72%\n- ELC: 19.72%\n- ITD: 19.72%'}
    

    ============ HumanReviewAgent Response ============
    {'messages': [{'role': 'assistant', 'content': 'APPROVED'}], 'approval_status': 'Approved', 'pm_feedback': 'APPROVED'}
    

When you run this, you'll see the workflow execute step by step. The typical output on the first pass looks like this:

```
[data_agent] Fetching market priors...
  → FPT: 10.7%, CMG: 10.0%, VGI: 10.0%, ELC: 10.0%, ITD: 10.0%

[bl_agent] Applying Black-Litterman model...
  → Posterior returns (first pass, no strong views applied)

[optimizer] Generating optimal weights...
  → FPT: 21.1%, CMG: 19.72%, VGI: 19.72%, ELC: 19.72%, ITD: 19.72%

[human_review] Reviewing portfolio weights...
  → APPROVED (diversification criterion met)
```

In this case, the optimizer naturally produces a reasonably diversified result and the PM approves on the first pass. If the weights were more skewed, which can happen when posterior returns differ more significantly, the system would loop back, apply stronger views, and try again.

## Conclusion

What we've built here is more than a portfolio optimizer. It's a small example of a broader principle: that AI agents are most useful not when they replace human judgment, but when they structure the process through which human judgment is applied.

The Black-Litterman model gives us a principled way to blend market wisdom with expert views. The multi-agent workflow gives us the infrastructure to apply that model consistently, at scale, with governance. The HITL feedback loop ensures that no algorithmic output escapes scrutiny and that scrutiny is productive, feeding back into the next iteration rather than simply blocking progress.

For anyone building serious AI applications in finance or in any domain where decisions carry real stakes this architecture offers a template worth studying:

- Ground your priors in reality (equilibrium returns, market data, established baselines)
- Encode expert knowledge explicitly (views, constraints, qualitative criteria)
- Build review gates that generate feedback, not just approvals and rejections
- Make the workflow auditable - every decision should trace back to interpretable inputs

Vinagent makes all of this surprisingly accessible. The graph abstraction maps cleanly onto institutional workflow thinking, and the agent pattern lets you separate concerns data fetching, model calculation, optimization, review in a way that's easy to maintain and extend.

The next steps from here are natural extensions: replace the simulated PM with a real human-in-the-loop interrupt, add portfolio constraints (turnover limits, sector caps, liquidity filters), integrate live market data, or extend the model to multi-period rebalancing. Each of those is a node in a slightly more complex graph but the architecture you've built here scales to all of them.
