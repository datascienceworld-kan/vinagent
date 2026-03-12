# Vinagent Document

With the target "make the AI community better by inventing AI libraries", I share how [vinagent site](https://datascienceworld-kan.github.io/vinagent) is created with you. I wish that more-and-more useful libraries will be invented from community with a careful and detailed documents.

## Install dependencies
To deploy on your local website on your computer, you need to install mkdoc library first.

```
pip install mkdocs==1.6.1
# Check version
mkdocs --version
```

## Structuring website
Then, structuring your site at `mkdocs.yml` template. For instance, I want to create a header bar including four tabs: `Get started, Guidlines, Reference, Contributing` with their relevant sub sections on the left side. Let's configure `mkdocs.yml` as following:

```
nav:
  - Get started: 
    - index.md
    - Quick Start:
      - Start with basic Agent: get_started/basic_agent.md
      - Build a customization: 
        - 1. Add tools: get_started/add_tool.md
        - 2. Add memory: get_started/add_memory.md
        - 3. Async Invoking: get_started/async_invoke.md
        - - 4. Streaming & Async Streaming: get_started/streaming.md
        - 5. Prebuilt Agent: get_started/react_agent.md
        - 6. Security: get_started/authen_layer.md
        - 7. Observability: get_started/observability.md
      - Run local ReactJS App: get_started/local_run.md
    - Agent Development:
      - Workflow & Agent: get_started/workflow_and_agent.md
      - Agent RAG: get_started/agent_rag.md
    - Multi-Agent Development: 
      - Start with basic Multi-Agent: get_started/multi_agent.md

  - Guidelines: 
    - Financial Usercase:
      - Visualize and Analyze Stock: guides/analyze_stock_trending.md
      - Find Trending New: guides/trending_news.md
    - Banking Agent:
      - Banking SQL: guides/banking_agent.md
    - Research Usercase:
      - Research Agent: guides/paper_research.md
    - Legal Field:
      - Legal Agent: guides/legal_assistant.md
    - Ecommerce Usercase:
      - Customer Care Multi-Agent: guides/customer_care.md
    - RAG:
      - Agent RAG: guides/agent_rag.md
    
  - Reference: 
    - Agent: reference/agent.md
    - Tool: reference/tool.md
    - Memory: reference/memory.md
    - Graph: reference/graph.md
    - MCP: reference/mcp.md
    - Tracing: reference/tracing.md
    - Authenticate: reference/authenticate.md

  - Contributing:
    - How to Contribute: contributing/contributing.md
```

You can add new blogs into  `mkdocs.yml`. 

## Rendering website on local
To test your website rendering on local, let's deploy it as following:

```
mkdocs --serve
```


    INFO    -  Documentation built in 3.15 seconds
    INFO    -  [10:36:51] Watching paths for changes: 'docs', 'mkdocs.yml'
    INFO    -  [10:36:51] Serving on http://127.0.0.1:8000/vinagent/


You can access a new website at `http://127.0.0.1:8000/vinagent`


## Deploy on website
A github action [CI/CD deployment](https://github.com/datascienceworld-kan/vinagent/blob/main/.github/workflows/ci_docs.yaml) pipeline takes responsibility to track new changing in `docs` folder and proceeds auto deployment.
