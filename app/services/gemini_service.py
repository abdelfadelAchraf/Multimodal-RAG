from pathlib import Path

from google import genai
from google.genai import types

from app.core.config import settings


class GeminiService:

    MODEL_NAME = "gemini-3.8-flash"

    def __init__(self):
        if not settings.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is not configured.")

        self.client = genai.Client(
            api_key=settings.GOOGLE_API_KEY
        )

    def generate_answer(
        self,
        question: str,
        text_context: list[dict],
        image_context: list[dict],
        project_root: Path
    ) -> str:

        contents = []

        # Add the user's question.
        contents.append(
            f"USER QUESTION:\n{question}"
        )

        # Add retrieved text.
        contents.append("\nRETRIEVED TEXT CONTEXT:\n")

        if text_context:
            for index, item in enumerate(text_context, start=1):
                contents.append(
                    f"""
                    Source {index}:
                    Document: {item['filename']}
                    Page: {item['page_number']}
                    Content: {item['text']}
                    """
                )
        else:
            contents.append("No relevant text was retrieved.")

        # Add instructions for the visual context.
        contents.append("\nRETRIEVED IMAGE CONTEXT:\n")

        image_parts = []

        for index, item in enumerate(image_context, start=1):
            image_path = project_root / item["image_path"]

            if not image_path.is_file():
                continue

            image_bytes = image_path.read_bytes()

            contents.append(
                f"""
                Image {index}:
                Document: {item['filename']}
                Page: {item['page_number']}
                """
            )

            image_parts.append(
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=self._get_mime_type(image_path)
                )
            )

        contents.extend(image_parts)

        system_instruction = """
        You are OmniKnowledge, a multimodal document assistant.

        Answer the user's question using the supplied text and images.

        Rules:
        1. Ground your answer in the provided document context.
        2. Use images to understand diagrams, charts and visual information.
        3. Do not invent facts that are not supported by the context.
        4. If the context is insufficient, clearly say so.
        5. Cite document sources using their filename and page number.
        6. Distinguish information found in text from information
           interpreted from images.
        """

        response = self.client.models.generate_content(
            model=self.MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2
            )
        )

        if not response.text:
            raise RuntimeError("Gemini returned an empty response.")

        return response.text

    @staticmethod
    def _get_mime_type(image_path: Path) -> str:
        extension = image_path.suffix.lower()

        mime_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".bmp": "image/bmp"
        }

        mime_type = mime_types.get(extension)

        if mime_type is None:
            raise ValueError(
                f"Unsupported image format: {extension}"
            )

        return mime_type


gemini_service = GeminiService()