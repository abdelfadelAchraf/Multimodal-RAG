import logging

from fastapi import APIRouter, HTTPException
from starlette.concurrency import run_in_threadpool

from app.schemas.chat import ChatRequest, ChatResponse
from app.services.rag_service import rag_service


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/chat",
    tags=["Chat"]
)


@router.post(
    "",
    response_model=ChatResponse
)
async def chat(request: ChatRequest):

    try:
        logger.info("CHAT: Starting RAG pipeline")
        result = await run_in_threadpool(
            rag_service.answer,
            question=request.question,
            top_k=request.top_k,
            document_id=request.document_id
        )
        logger.info("CHAT: RAG pipeline completed successfully")

        return ChatResponse(**result)

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception:
        logger.exception("RAG answer generation failed")

        raise HTTPException(
            status_code=500,
            detail="An error occurred while generating the answer."
        )