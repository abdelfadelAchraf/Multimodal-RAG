from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=1000
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20
    )

    document_id: str | None = None


class ChatResponse(BaseModel):
    question: str
    answer: str

    text_sources: list[dict]
    image_sources: list[dict]