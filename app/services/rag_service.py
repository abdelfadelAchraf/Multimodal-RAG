from pathlib import Path

from app.services.retrieval_service import retrieval_service
from app.services.gemini_service import gemini_service


class RAGService:

    def __init__(self):
        self.project_root = Path(__file__).resolve().parents[2]

    def answer(
        self,
        question: str,
        top_k: int = 5,
        document_id: str | None = None
    ) -> dict:

        # STEP 1: Retrieve relevant information.
        retrieval_results = retrieval_service.search(
            question=question,
            top_k=top_k,
            document_id=document_id
        )

        text_context = retrieval_results["text_results"]
        image_context = retrieval_results["image_results"]

        # STEP 2: Generate an answer using retrieved context.
        if not text_context and not image_context:
            return {
                "question": question,
                "answer": (
                    "I couldn't find relevant information "
                    "in the indexed documents."
                ),
                "text_sources": [],
                "image_sources": []
            }

        answer = gemini_service.generate_answer(
            question=question,
            text_context=text_context,
            image_context=image_context,
            project_root=self.project_root
        )

        # STEP 3: Return the answer and its sources.
        return {
            "question": question,
            "answer": answer,
            "text_sources": text_context,
            "image_sources": image_context
        }


rag_service = RAGService()