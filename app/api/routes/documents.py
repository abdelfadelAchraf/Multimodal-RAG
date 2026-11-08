
import logging

from fastapi import APIRouter, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.core.config import settings
from app.schemas.document import DocumentUploadResponse
from app.services.document_service import document_service
from app.utils.file_utils import validate_pdf_upload


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/documents",
    tags=["Documents"]
)


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=201
)
async def upload_document(
    file: UploadFile = File(...)
):
    try:
        # 1. Read the uploaded file with a size limit.
        max_size = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024

        content = bytearray()

        while True:
            chunk = await file.read(1024 * 1024)

            if not chunk:
                break

            content.extend(chunk)

            if len(content) > max_size:
                raise ValueError(
                    "File exceeds the maximum allowed size."
                )

        # 2. Validate the uploaded PDF.
        filename = file.filename or ""

        validate_pdf_upload(
            filename=filename,
            content_type=file.content_type,
            file_content=bytes(content),
            file_size=len(content),
            max_size=max_size
        )

        # 3. Process the PDF outside the async event loop.
        result = await run_in_threadpool(
            document_service.ingest_pdf,
            bytes(content),
            filename
        )

        # 4. Return a structured API response.
        return DocumentUploadResponse(
            **result,
            message="PDF uploaded and processed successfully."
        )

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception:

        logger.exception("PDF ingestion failed")

        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing the PDF."
        )

    finally:
        await file.close()
