import logging

from fastapi import FastAPI

from models import MessagesResponse, ProcessRequest, ProcessResponse
from services import AIService

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="AI Solutions API")
ai_service = AIService()


@app.get("/")
def hello_world():
    """Hello World GET endpoint."""
    return {"message": "Hello World"}


@app.post("/process", response_model=ProcessResponse)
def process(request: ProcessRequest):
    """POST endpoint that accepts JSON with a 'text' field and returns a chain response with source_documents."""
    answer, num_tokens, source_documents = ai_service.process(request.text)
    return ProcessResponse(
        received_text=request.text,
        status="processed",
        answer=answer,
        num_tokens=num_tokens,
        source_documents=source_documents,
    )


@app.get("/messages", response_model=MessagesResponse)
def get_messages():
    """GET endpoint that returns all messages saved via /process."""
    return MessagesResponse(messages=ai_service.get_messages())
