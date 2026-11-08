from pathlib import Path

from app.services.clip_service import clip_service
from app.repositories.chroma_repository import chroma_repository


class RetrievalService:

    def search(
        self,
        question: str,
        top_k: int = 5,
        document_id: str | None = None
    ) -> dict:
        """
        Search relevant text chunks and images using CLIP.
        """

        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        if top_k < 1:
            raise ValueError("top_k must be greater than zero.")

        # STEP 1: Generate the question embedding.
        query_embedding = clip_service.embed_text(question)

        # STEP 2: Retrieve relevant text chunks.
        text_results = self._search_text(
            query_embedding=query_embedding,
            top_k=top_k,
            document_id=document_id
        )

        # STEP 3: Retrieve relevant images.
        image_results = self._search_images(
            query_embedding=query_embedding,
            top_k=top_k,
            document_id=document_id
        )

        return {
            "question": question,
            "text_results": text_results,
            "image_results": image_results
        }

    def _search_text(
        self,
        query_embedding: list[float],
        top_k: int,
        document_id: str | None
    ) -> list[dict]:

        collection = chroma_repository.text_collection

        collection_count = collection.count()

        if collection_count == 0:
            return []

        results = chroma_repository.search_text(
            query_embedding=query_embedding,
            n_results=min(top_k, collection_count),
            document_id=document_id
        )

        return [
            {
                "text": document,
                "document_id": metadata["document_id"],
                "filename": metadata["filename"],
                "page_number": metadata["page_number"],
                "chunk_index": metadata["chunk_index"],
                "distance": distance
            }
            for document, metadata, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )
        ]

    def _search_images(
        self,
        query_embedding: list[float],
        top_k: int,
        document_id: str | None
    ) -> list[dict]:

        collection = chroma_repository.image_collection

        collection_count = collection.count()

        if collection_count == 0:
            return []

        results = chroma_repository.search_images(
            query_embedding=query_embedding,
            n_results=min(top_k, collection_count),
            document_id=document_id
        )

        return [
            {
                "document_id": metadata["document_id"],
                "filename": metadata["filename"],
                "page_number": metadata["page_number"],
                "image_path": metadata["image_path"],
                "distance": distance
            }
            for metadata, distance in zip(
                results["metadatas"][0],
                results["distances"][0]
            )
        ]


retrieval_service = RetrievalService()