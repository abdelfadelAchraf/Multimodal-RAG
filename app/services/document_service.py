
import json
import shutil
import uuid

from pathlib import Path

from app.core.config import settings
from app.pipelines.ingestion import extract_pdf_content


class DocumentService:

    def __init__(self):
        # Resolve paths relative to the project root.
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

        pdf_path = (
            self.upload_directory / f"{document_id}.pdf"
        )

        document_directory = (
            self.extracted_directory / document_id
        )

        image_directory = (
            self.extracted_directory
            / "images"
            / document_id
        )

        try:
            # 1. Create required directories.
            self.upload_directory.mkdir(
                parents=True,
                exist_ok=True
            )

            # 2. Save the original PDF.
            pdf_path.write_bytes(file_content)

            # 3. Extract PDF content.
            extraction_result = extract_pdf_content(
                pdf_path=pdf_path,
                image_output_dir=image_directory,
                document_id=document_id,
                original_filename=original_filename
            )

            # 4. Prepare image paths relative to project root.
            for image in extraction_result["images"]:
                image["image_path"] = str(
                    Path(image["image_path"]).relative_to(
                        self.project_root
                    )
                )

            # 5. Save extracted content as JSON.
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

            # 6. Return document information.
            return {
                "document_id": document_id,
                "filename": original_filename,
                "page_count": extraction_result["page_count"],
                "text_blocks_count": (
                    extraction_result["text_blocks_count"]
                ),
                "images_count": extraction_result["images_count"],
                "status": "processed"
            }

        except Exception:
            # Remove partial files if processing fails.
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
