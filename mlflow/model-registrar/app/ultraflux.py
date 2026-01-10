import torch
from diffusers import DiffusionPipeline
import mlflow
import mlflow.transformers  # Diffusers pipelines jsou kompatibilní s transformers integration

# Název modelu z Hugging Face
model_name = "Owen777/UltraFlux-v1"

# Detekce zařízení
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.float32

# Stáhnout pipeline
pipe = DiffusionPipeline.from_pretrained(
    model_name,
    torch_dtype=torch_dtype
).to(device)
pipe.enable_attention_slicing()  # Optimalizace pro paměť

# Příprava input_example (text prompt pro generování obrázku)
sample_prompt = ["A beautiful landscape with mountains and a lake"]

with mlflow.start_run() as run:
    # Log parametrů
    mlflow.log_params({"model_name": model_name, "torch_dtype": "float16"})

    # Log modelu a registrace
    mlflow.transformers.log_model(
        transformers_model=pipe,
        name="ultraflux-model",
        input_example=sample_prompt,
        pip_requirements=["diffusers", "torch", "transformers"],
        registered_model_name="ultraflux-model"
    )

    print(f"UltraFlux model zaregistrován: {run.info.run_id}")