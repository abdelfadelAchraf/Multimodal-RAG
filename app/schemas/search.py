from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
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


class TextSearchResult(BaseModel):
    text: str
    document_id: str
    filename: str
    page_number: int
    chunk_index: int
    distance: float


class ImageSearchResult(BaseModel):
    document_id: str
    filename: str
    page_number: int
    image_path: str
    distance: float


class SearchResponse(BaseModel):
    question: str
    text_results: list[TextSearchResult]
    image_results: list[ImageSearchResult]