from fastapi import FastAPI
from pydantic import BaseModel
from io import BytesIO
import base64

from .model import UltraFluxModel

app = FastAPI(title="UltraFlux API")

model = UltraFluxModel()

class GenerateRequest(BaseModel):
    prompt: str
    steps: int = 30
    guidance_scale: float = 7.5

@app.post("/generate")
def generate(req: GenerateRequest):
    image = model.generate(
        prompt=req.prompt,
        steps=req.steps,
        guidance_scale=req.guidance_scale
    )

    buffer = BytesIO()
    image.save(buffer, format="PNG")

    return {
        "image_base64": base64.b64encode(buffer.getvalue()).decode()
    }