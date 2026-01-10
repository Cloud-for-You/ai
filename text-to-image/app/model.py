import torch
import mlflow
import mlflow.transformers
from diffusers import DiffusionPipeline

MODEL_ID = "Owen777/UltraFlux-v1"

class UltraFluxModel:
    def __init__(self):
        # Načtení modelu z MLflow registry místo přímého z HF
        # Předpokládá se, že model je registrovaný pod názvem "ultraflux-model"
        model_uri = "models:/ultraflux-model/latest"  # Nebo specifická verze, např. "models:/ultraflux-model/1"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = mlflow.transformers.load_model(model_uri).to(device)

        # Pokud potřebujete specifické nastavení
        self.pipe.enable_attention_slicing()

    def generate(self, prompt: str, steps: int, guidance_scale: float):
        return self.pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale
        ).images[0]