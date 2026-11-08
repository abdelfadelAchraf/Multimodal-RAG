
from pathlib import Path


def validate_pdf_upload(
    filename: str,
    content_type: str | None,
    file_content: bytes,
    file_size: int,
    max_size: int
) -> None:
    """
    Validate an uploaded PDF before processing.
    """

    if not filename:
        raise ValueError("Filename is required.")

    if Path(filename).suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    if content_type not in (
        None,
        "",
        "application/pdf",
        "application/octet-stream"
    ):
        raise ValueError("Invalid content type.")

    if file_size == 0:
        raise ValueError("The uploaded file is empty.")

    if file_size > max_size:
        raise ValueError("File exceeds the maximum allowed size.")

    # Verify the PDF signature in its first 1024 bytes.
    if b"%PDF-" not in file_content[:1024]:
        raise ValueError("The uploaded file is not a valid PDF.")
