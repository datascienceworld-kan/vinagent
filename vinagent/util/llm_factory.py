from langchain_together import ChatTogether

class LLMFactory:
    @staticmethod
    def create_llm(model_name: str):
        return ChatTogether(model=model_name)