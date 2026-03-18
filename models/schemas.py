from pydantic import BaseModel


class ProcessRequest(BaseModel):
    """Request body for POST /process."""

    text: str


class ProcessResponse(BaseModel):
    """Response body for POST /process."""

    received_text: str
    status: str
    answer: str
    num_tokens: int
    source_documents: list[str]


class MessagesResponse(BaseModel):
    """Response body for GET /messages."""

    messages: list[str]
