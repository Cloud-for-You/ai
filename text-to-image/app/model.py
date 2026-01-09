import torch
from diffusers import DiffusionPipeline

MODEL_ID = "Owen777/UltraFlux-v1"

class UltraFluxModel:
    def __init__(self):
        self.pipe = DiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16
        ).to("cuda")

        self.pipe.enable_attention_slicing()

    def generate(self, prompt: str, steps: int, guidance_scale: float):
        return self.pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale
        ).images[0]