import logging

from fastapi import APIRouter, HTTPException
from starlette.concurrency import run_in_threadpool

from app.schemas.search import SearchRequest, SearchResponse
from app.services.retrieval_service import retrieval_service


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/search",
    tags=["Search"]
)


@router.post(
    "",
    response_model=SearchResponse
)
async def search_documents(request: SearchRequest):

    try:
        results = await run_in_threadpool(
            retrieval_service.search,
            question=request.question,
            top_k=request.top_k,
            document_id=request.document_id
        )

        return SearchResponse(**results)

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception:
        logger.exception("Document search failed")

        raise HTTPException(
            status_code=500,
            detail="An error occurred while searching documents."
        )