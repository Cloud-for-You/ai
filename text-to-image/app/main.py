from fastapi import FastAPI
from pydantic import BaseModel
from io import BytesIO
import base64
import logging

from .model import UltraFluxModel

# Nastavení loggingu
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

app = FastAPI(title="UltraFlux API")

logger = logging.getLogger(__name__)

model = UltraFluxModel()

class GenerateRequest(BaseModel):
    prompt: str
    height: int = 4096
    width: int = 4096
    guidance_scale: float = 4.0
    num_inference_steps: int = 50
    max_sequence_length: int = 512
    seed: int = 0

@app.post("/generate")
def generate(req: GenerateRequest):
    logger.info(f"Začínám generování obrázku pro prompt: '{req.prompt}' s parametry: height={req.height}, width={req.width}, guidance_scale={req.guidance_scale}, num_inference_steps={req.num_inference_steps}, seed={req.seed}")
    image = model.generate(
        prompt=req.prompt,
        height=req.height,
        width=req.width,
        guidance_scale=req.guidance_scale,
        num_inference_steps=req.num_inference_steps,
        max_sequence_length=req.max_sequence_length,
        seed=req.seed
    )
    logger.info(f"Generování obrázku dokončeno pro prompt: '{req.prompt}'")

    buffer = BytesIO()
    image.save(buffer, format="PNG")

    return {
        "image_base64": base64.b64encode(buffer.getvalue()).decode()
    }