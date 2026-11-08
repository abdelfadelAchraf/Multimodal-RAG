
import fitz

from pathlib import Path


def extract_pdf_content(
    pdf_path: Path,
    image_output_dir: Path,
    document_id: str,
    original_filename: str
) -> dict:
    """
    Extract text and embedded images from a PDF.

    Text and images retain their original page numbers.
    """

    image_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    pages = []
    images = []

    with fitz.open(pdf_path) as pdf:

        if pdf.is_encrypted:
            raise ValueError("Encrypted PDFs are not supported.")

        if pdf.page_count == 0:
            raise ValueError("The PDF contains no pages.")

        for page_index, page in enumerate(pdf):

            # PDF pages are zero-indexed.
            # We use one-based page numbers for users.
            page_number = page_index + 1

            # 1. Extract text
            page_text = page.get_text("text").strip()

            pages.append({
                "page_number": page_number,
                "text": page_text,
                "character_count": len(page_text)
            })

            # 2. Extract embedded images
            extracted_xrefs = set()

            for image_index, image_info in enumerate(
                page.get_images(full=True),
                start=1
            ):

                xref = image_info[0]

                # Avoid extracting the same image twice
                # on the same page.
                if xref in extracted_xrefs:
                    continue

                extracted_xrefs.add(xref)

                extracted = pdf.extract_image(xref)

                if not extracted:
                    continue

                image_bytes = extracted["image"]
                image_extension = extracted["ext"].lower()

                # Only accept supported image formats.
                if image_extension not in {
                    "png", "jpg", "jpeg", "bmp"
                }:
                    continue

                image_filename = (
                    f"page_{page_number:04d}_"
                    f"image_{image_index:03d}."
                    f"{image_extension}"
                )

                image_path = (
                    image_output_dir / image_filename
                )

                image_path.write_bytes(image_bytes)

                images.append({
                    "document_id": document_id,
                    "filename": image_filename,
                    "image_path": str(image_path),
                    "page_number": page_number,
                    "xref": xref
                })

    return {
        "document_id": document_id,
        "original_filename": original_filename,
        "page_count": len(pages),
        "pages": pages,
        "images": images,
        "text_blocks_count": sum(
            1 for page in pages if page["text"]
        ),
        "images_count": len(images)
    }
