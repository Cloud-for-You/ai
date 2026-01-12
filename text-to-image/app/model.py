import os
import torch
from diffusers import DiffusionPipeline

class UltraFluxModel:
    def __init__(self):
        hf_token = os.environ.get("HF_TOKEN")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Modified transformer config to remove unsupported attributes
        transformer_config = {
            "_class_name": "FluxTransformer2DModel",
            "_diffusers_version": "0.36.0.dev0",
            "_name_or_path": "transformer",
            "attention_head_dim": 128,
            "axes_dims_rope": [16, 56, 56],
            "guidance_embeds": True,
            "in_channels": 64,
            "joint_attention_dim": 4096,
            "num_attention_heads": 24,
            "num_layers": 19,
            "num_single_layers": 38,
            "out_channels": None,
            "patch_size": 1,
            "pooled_projection_dim": 768
        }

        config = {"transformer": transformer_config}

        self.pipe = DiffusionPipeline.from_pretrained(
            "Owen777/UltraFlux-v1",
            dtype=torch.float16,
            token=hf_token,
            config=config
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