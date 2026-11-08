
from pydantic import BaseModel


class DocumentUploadResponse(BaseModel):

    document_id: str
    filename: str

    status: str

    page_count: int
    text_blocks_count: int
    images_count: int

    message: str
