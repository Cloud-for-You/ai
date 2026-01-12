import os
import torch
import mlflow
import mlflow.transformers
from diffusers import DiffusionPipeline

class UltraFluxModel:
    def __init__(self):
        hf_token = os.environ.get("HF_TOKEN")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = DiffusionPipeline.from_pretrained(
            "Owen777/UltraFlux-v1",
            torch_dtype=torch.float16,
            token=hf_token
        ).to(device)

        # Kontrola, zda je model na GPU
        print(f"Model loaded on device: {self.pipe.device}")

        # Pokud potřebujete specifické nastavení
        self.pipe.enable_attention_slicing()

    def generate(self, prompt: str, steps: int, guidance_scale: float):
        return self.pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale
        ).images[0]