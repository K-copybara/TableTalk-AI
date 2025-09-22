from pydantic import BaseModel

class ChatRequest(BaseModel):
    session_id: str
    user_input: str
    store_id: int

class ChatResponse(BaseModel):
    answer: str