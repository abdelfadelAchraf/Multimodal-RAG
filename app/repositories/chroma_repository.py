
from pathlib import Path
from typing import Any

import chromadb

from app.core.config import settings


class ChromaRepository:
    TEXT_COLLECTION = "text_embeddings"
    IMAGE_COLLECTION = "image_embeddings"

    def __init__(self):
        project_root = Path(__file__).resolve().parents[2]

        database_path = (
            project_root / settings.CHROMA_PERSIST_DIRECTORY
        )

        database_path.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(database_path)
        )

        # Both collections use cosine distance.
        self.text_collection = self.client.get_or_create_collection(
            name=self.TEXT_COLLECTION,
            metadata={"hnsw:space": "cosine"}
        )

        self.image_collection = self.client.get_or_create_collection(
            name=self.IMAGE_COLLECTION,
            metadata={"hnsw:space": "cosine"}
        )

    def upsert_text_chunks(
        self,
        chunks: list[dict[str, Any]],
        embeddings: list[list[float]]
    ) -> None:
        """Store text chunk embeddings and their source information."""

        if not chunks:
            return

        if len(chunks) != len(embeddings):
            raise ValueError(
                "The number of chunks must match the number of embeddings."
            )

        ids = [chunk["chunk_id"] for chunk in chunks]

        documents = [chunk["text"] for chunk in chunks]

        metadatas = [
            {
                "document_id": chunk["document_id"],
                "filename": chunk["filename"],
                "page_number": chunk["page_number"],
                "chunk_index": chunk["chunk_index"]
            }
            for chunk in chunks
        ]

        self.text_collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

    def upsert_image_embeddings(
        self,
        images: list[dict[str, Any]],
        embeddings: list[list[float]]
    ) -> None:
        """Store image embeddings and their file references."""

        if not images:
            return

        if len(images) != len(embeddings):
            raise ValueError(
                "The number of images must match the number of embeddings."
            )

        ids = [
            (
                f"{image['document_id']}_"
                f"page_{image['page_number']}_"
                f"image_{image['xref']}"
            )
            for image in images
        ]

        metadatas = [
            {
                "document_id": image["document_id"],
                "filename": image["filename"],
                "page_number": image["page_number"],
                "image_path": image["image_path"]
            }
            for image in images
        ]

        self.image_collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def search_text(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        document_id: str | None = None
    ) -> dict:
        """Find the most similar text chunks."""

        query_args = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"]
        }

        if document_id:
            query_args["where"] = {
                "document_id": document_id
            }

        return self.text_collection.query(**query_args)

    def search_images(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        document_id: str | None = None
    ) -> dict:
        """Find the most similar images."""

        query_args = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["metadatas", "distances"]
        }

        if document_id:
            query_args["where"] = {
                "document_id": document_id
            }

        return self.image_collection.query(**query_args)

    def delete_document(self, document_id: str) -> None:
        """Delete all vectors belonging to a document."""

        self.text_collection.delete(
            where={"document_id": document_id}
        )

        self.image_collection.delete(
            where={"document_id": document_id}
        )


chroma_repository = ChromaRepository()
