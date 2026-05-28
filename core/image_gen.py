import os
import io
from huggingface_hub import InferenceClient
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

IMAGE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"


def generate_christian_image(refined_prompt: str) -> Image.Image | None:
    """
    Generate a Christian-themed image using SDXL via HF Inference API.
    The prompt must already be safety-checked and refined before calling this.
    Returns a PIL Image or None on failure.
    """
    client = InferenceClient(token=os.getenv("HF_TOKEN"))
    try:
        image_bytes = client.text_to_image(refined_prompt, model=IMAGE_MODEL)
        if isinstance(image_bytes, Image.Image):
            return image_bytes
        return Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        print(f"Image generation error: {e}")
        return None
