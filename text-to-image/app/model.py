import os
import torch
from PIL import Image
import logging
from diffusers import FluxPipeline, AutoencoderKL
from diffusers.models import FluxTransformer2DModel
from diffusers import FlowMatchEulerDiscreteScheduler

logger = logging.getLogger(__name__)

class UltraFluxModel:
    def __init__(self):
        logger.info("Načítám UltraFlux model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        local_vae = AutoencoderKL.from_pretrained("Owen777/UltraFlux-v1", subfolder="vae", torch_dtype=torch.bfloat16)
        transformer = FluxTransformer2DModel.from_pretrained("Owen777/UltraFlux-v1", subfolder="transformer", torch_dtype=torch.bfloat16)
        # transformer = FluxTransformer2DModel.from_pretrained("Owen777/UltraFlux-v1-1-Transformer", torch_dtype=torch.bfloat16)  # NOTE: uncomment this line to use UltraFlux-v1.1
        self.pipe = FluxPipeline.from_pretrained("Owen777/UltraFlux-v1", vae=local_vae, torch_dtype=torch.bfloat16, transformer=transformer)
        self.pipe.scheduler.config.use_dynamic_shifting = False
        self.pipe.scheduler.config.time_shift = 4
        self.pipe = self.pipe.to(device)

        logger.info(f"Model načten na zařízení: {self.pipe.device}")
        print(f"Model loaded on device: {self.pipe.device}")

    def generate(self, prompt: str, height: int = 4096, width: int = 4096, guidance_scale: float = 4.0, num_inference_steps: int = 50, max_sequence_length: int = 512, seed: int = 0):
        logger.info(f"Spouštím generování obrázku s promptem: '{prompt}'")

        def progress_callback(step: int, timestep, latents):
            logger.info(f"Generování: krok {step + 1}/{num_inference_steps} pro prompt: '{prompt}'")

        generator = torch.Generator("cpu").manual_seed(seed)
        result = self.pipe(
            prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            max_sequence_length=max_sequence_length,
            generator=generator,
            callback_on_step_end=progress_callback
        ).images[0]
        logger.info(f"Generování obrázku dokončeno pro prompt: '{prompt}'")
        return result