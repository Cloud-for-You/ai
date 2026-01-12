from fastapi import FastAPI
from pydantic import BaseModel
from io import BytesIO
import base64

from .model import UltraFluxModel

app = FastAPI(title="UltraFlux API")

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
    image = model.generate(
        prompt=req.prompt,
        height=req.height,
        width=req.width,
        guidance_scale=req.guidance_scale,
        num_inference_steps=req.num_inference_steps,
        max_sequence_length=req.max_sequence_length,
        seed=req.seed
    )

    buffer = BytesIO()
    image.save(buffer, format="PNG")

    return {
        "image_base64": base64.b64encode(buffer.getvalue()).decode()
    }