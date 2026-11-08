
import json
import logging
import shutil
import uuid
from pathlib import Path

from app.core.config import settings
from app.pipelines.ingestion import extract_pdf_content
from app.pipelines.chunking import create_text_chunks
from app.services.clip_service import clip_service
from app.repositories.chroma_repository import chroma_repository


logger = logging.getLogger(__name__)


class DocumentService:
    def __init__(self):
        self.project_root = Path(__file__).resolve().parents[2]

        self.upload_directory = (
            self.project_root / settings.UPLOAD_DIRECTORY
        )

        self.extracted_directory = (
            self.project_root / settings.EXTRACTED_DIRECTORY
        )

    def ingest_pdf(
        self,
        file_content: bytes,
        original_filename: str
    ) -> dict:

        document_id = str(uuid.uuid4())

        pdf_path = self.upload_directory / f"{document_id}.pdf"

        document_directory = (
            self.extracted_directory / document_id
        )

        image_directory = (
            self.extracted_directory / "images" / document_id
        )

        try:
            self.upload_directory.mkdir(
                parents=True,
                exist_ok=True
            )

            pdf_path.write_bytes(file_content)

            # STEP 1: Extract PDF text and images.
            extraction_result = extract_pdf_content(
                pdf_path=pdf_path,
                image_output_dir=image_directory,
                document_id=document_id,
                original_filename=original_filename
            )

            # Convert image paths to project-relative paths.
            for image in extraction_result["images"]:
                image["image_path"] = str(
                    Path(image["image_path"]).relative_to(
                        self.project_root
                    )
                )

            # STEP 2: Generate text chunks.
            chunks = create_text_chunks(
                pages=extraction_result["pages"],
                document_id=document_id,
                original_filename=original_filename
            )

           # STEP 3: Generate text embeddings in batches.
            texts = [chunk["text"] for chunk in chunks]

            text_embeddings = clip_service.embed_texts(
                 texts=texts,
                 batch_size=16
            )

# STEP 4: Generate image embeddings in batches.
            images = extraction_result["images"]

            image_paths = [
              self.project_root / image["image_path"]
              for image in images
            ]

            image_embeddings = clip_service.embed_images(
                 image_paths=image_paths,
                 batch_size=8
            )

            # STEP 5: Store text vectors.
            chroma_repository.upsert_text_chunks(
                chunks=chunks,
                embeddings=text_embeddings
            )

            # STEP 6: Store image vectors.
            chroma_repository.upsert_image_embeddings(
                images=images,
                embeddings=image_embeddings
            )

            # STEP 7: Save extraction metadata.
            extraction_result["chunks"] = chunks
            extraction_result["chunks_count"] = len(chunks)

            document_directory.mkdir(
                parents=True,
                exist_ok=True
            )

            extraction_file = (
                document_directory / "extraction.json"
            )

            extraction_file.write_text(
                json.dumps(
                    extraction_result,
                    indent=4,
                    ensure_ascii=False
                ),
                encoding="utf-8"
            )

            return {
                "document_id": document_id,
                "filename": original_filename,
                "page_count": extraction_result["page_count"],
                "text_blocks_count": (
                    extraction_result["text_blocks_count"]
                ),
                "images_count": extraction_result["images_count"],
                "chunks_count": len(chunks),
                "status": "processed"
            }

        except Exception:
            logger.exception(
                "Document ingestion failed: %s",
                document_id
            )

            # Roll back any vectors inserted before the failure.
            try:
                chroma_repository.delete_document(document_id)
            except Exception:
                logger.exception(
                    "Failed to clean up vectors for %s",
                    document_id
                )

            # Clean up files created for this document.
            pdf_path.unlink(missing_ok=True)

            shutil.rmtree(
                document_directory,
                ignore_errors=True
            )

            shutil.rmtree(
                image_directory,
                ignore_errors=True
            )

            raise


document_service = DocumentService()
