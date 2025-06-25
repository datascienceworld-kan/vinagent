import os
from dotenv import load_dotenv
from tavily import TavilyClient
from dataclasses import dataclass
from typing import Union, Any

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
