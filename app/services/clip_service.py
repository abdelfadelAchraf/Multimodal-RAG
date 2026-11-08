
from pathlib import Path

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

    def embed_text(self, text: str) -> list[float]:
        """Generate a normalized text embedding."""

        if not text or not text.strip():
            raise ValueError("Text cannot be empty.")

        self._load_model()

        inputs = self.processor(
            text=[text],
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
            # Extract pooled text representation.
            text_outputs = self.model.text_model(**inputs)

            # Project into CLIP's shared embedding space.
            features = self.model.text_projection(
                text_outputs.pooler_output
            )

            features = F.normalize(
                features,
                p=2,
                dim=-1
            )

        return features[0].cpu().tolist()

    def embed_image(self, image_path: Path) -> list[float]:
        """Generate a normalized image embedding."""

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        self._load_model()

        with Image.open(image_path) as image:
            image = image.convert("RGB")

            inputs = self.processor(
                images=image,
                return_tensors="pt"
            )

        pixel_values = inputs["pixel_values"].to(self.device)

        with torch.inference_mode():
            # Extract pooled visual representation.
            image_outputs = self.model.vision_model(
                pixel_values=pixel_values
            )

            # Project into CLIP's shared embedding space.
            features = self.model.visual_projection(
                image_outputs.pooler_output
            )

            features = F.normalize(
                features,
                p=2,
                dim=-1
            )

        return features[0].cpu().tolist()


clip_service = CLIPEmbeddingService()
