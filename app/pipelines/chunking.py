
from functools import lru_cache

from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import CLIPTokenizerFast




MODEL_NAME = "openai/clip-vit-base-patch32"

@lru_cache(maxsize=1)
def get_clip_tokenizer():
    """Load and cache the tokenizer."""
    return CLIPTokenizerFast.from_pretrained(MODEL_NAME)


def create_text_chunks(
    pages: list[dict],
    document_id: str,
    original_filename: str
) -> list[dict]:

    tokenizer = get_clip_tokenizer()

    def count_tokens(text: str) -> int:
        return len(
            tokenizer.encode(
                text,
                add_special_tokens=False
            )
        )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=65,
        chunk_overlap=10,
        length_function=count_tokens,
        separators=["\n\n", "\n", ". ", " ", ""]
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
