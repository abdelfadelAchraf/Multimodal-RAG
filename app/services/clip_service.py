from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class CLIPEmbeddingService:
    MODEL_NAME = "openai/clip-vit-base-patch32"

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None

    def _load_model(self):
        """Load CLIP once and reuse it."""

        if self.model is None:
            print(f"Loading CLIP on {self.device}...")

            self.processor = CLIPProcessor.from_pretrained(
                self.MODEL_NAME
            )

            self.model = CLIPModel.from_pretrained(
                self.MODEL_NAME
            )

            self.model.to(self.device)
            self.model.eval()

            print("CLIP loaded successfully.")

    def embed_texts(
        self,
        texts: Sequence[str],
        batch_size: int = 16
    ) -> list[list[float]]:
        """Generate normalized embeddings for multiple text inputs."""

        if batch_size <= 0:
            raise ValueError("Batch size must be greater than zero.")

        if not texts:
            return []

        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("All texts must be non-empty strings.")

        self._load_model()

        embeddings = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]

            inputs = self.processor(
                text=list(batch),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            )

            inputs = {
                key: value.to(self.device)
                for key, value in inputs.items()
            }

            with torch.inference_mode():
                text_outputs = self.model.text_model(**inputs)

                features = self.model.text_projection(
                    text_outputs.pooler_output
                )

                features = F.normalize(
                    features,
                    p=2,
                    dim=-1
                )

            embeddings.extend(features.cpu().tolist())

        return embeddings

    def embed_text(self, text: str) -> list[float]:
        """Generate one normalized text embedding."""

        return self.embed_texts([text])[0]

    def embed_images(
        self,
        image_paths: Sequence[str | Path],
        batch_size: int = 8
    ) -> list[list[float]]:
        """Generate normalized embeddings for multiple images."""

        if batch_size <= 0:
            raise ValueError("Batch size must be greater than zero.")

        if not image_paths:
            return []

        self._load_model()

        embeddings = []

        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start:start + batch_size]
            batch_images = []

            try:
                # Open only the images in the current batch.
                for image_path in batch_paths:
                    image_path = Path(image_path)

                    if not image_path.is_file():
                        raise FileNotFoundError(
                            f"Image not found: {image_path}"
                        )

                    with Image.open(image_path) as image:
                        batch_images.append(image.convert("RGB"))

                inputs = self.processor(
                    images=batch_images,
                    return_tensors="pt"
                )

                pixel_values = inputs["pixel_values"].to(self.device)

                with torch.inference_mode():
                    image_outputs = self.model.vision_model(
                        pixel_values=pixel_values
                    )

                    features = self.model.visual_projection(
                        image_outputs.pooler_output
                    )

                    features = F.normalize(
                        features,
                        p=2,
                        dim=-1
                    )

                embeddings.extend(features.cpu().tolist())

            finally:
                # Release image memory after processing this batch.
                for image in batch_images:
                    image.close()

        return embeddings

    def embed_image(self, image_path: str | Path) -> list[float]:
        """Generate one normalized image embedding."""

        return self.embed_images([image_path])[0]


clip_service = CLIPEmbeddingService()