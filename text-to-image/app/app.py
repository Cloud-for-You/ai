from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
from inf_ultraflux import setup_pipe, generate_images

app = FastAPI(title="UltraFlux Text-to-Image API", description="API for generating images using UltraFlux model")

# Global pipe variable
pipe = None

@app.on_event("startup")
async def startup_event():
    global pipe
    pipe = setup_pipe()

class GenerateRequest(BaseModel):
    prompts: List[str]
    height: int = 4096
    width: int = 4096
    guidance_scale: float = 4.0
    num_inference_steps: int = 50
    max_sequence_length: int = 512
    seed: int = 0

@app.post("/generate")
async def generate(request: GenerateRequest):
    if pipe is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    try:
        results = generate_images(
            pipe,
            request.prompts,
            height=request.height,
            width=request.width,
            guidance_scale=request.guidance_scale,
            num_inference_steps=request.num_inference_steps,
            max_sequence_length=request.max_sequence_length,
            seed=request.seed
        )
        return {"images": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)