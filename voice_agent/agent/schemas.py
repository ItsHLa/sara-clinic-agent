from pydantic import BaseModel, Field


class AgentChatRequest(BaseModel):
    question: str = Field(min_length=1)
    session_id: str | None = None
