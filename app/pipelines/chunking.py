
from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_text_chunks(
    pages: list[dict],
    document_id: str,
    original_filename: str
) -> list[dict]:
    """
    Split extracted PDF pages into smaller text chunks.

    Each chunk retains its source document and page number.
    """

    text_splitter = RecursiveCharacterTextSplitter( 
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )

    chunks = []

    for page in pages:
        page_text = page["text"].strip()

        if not page_text:
            continue

        page_chunks = text_splitter.split_text(page_text)

        for chunk_index, chunk_text in enumerate(page_chunks):
            chunks.append({
                "chunk_id": (
                    f"{document_id}_"
                    f"page_{page['page_number']}_"
                    f"chunk_{chunk_index}"
                ),
                "document_id": document_id,
                "filename": original_filename,
                "page_number": page["page_number"],
                "chunk_index": chunk_index,
                "text": chunk_text
            })

    return chunks
