from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field


class ChatContext(BaseModel):
    district_id: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=100,
    )
    zone_id: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=100,
    )


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
    )
    context: Optional[ChatContext] = None


class ToolCall(BaseModel):
    tool: str
    output: Dict[str, Any]


class ChatResponse(BaseModel):
    answer: str
    tool_calls: List[ToolCall]
    evidence: List[Dict[str, Any]]
