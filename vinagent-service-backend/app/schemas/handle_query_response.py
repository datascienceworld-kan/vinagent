from pydantic import BaseModel
from typing import Optional, Any

class ChatMessage(BaseModel):
    from_: str 
    text: str

class QueryResponse(BaseModel):
    query_id: str
    chat_message: ChatMessage
    artifact_data: Optional[Any]