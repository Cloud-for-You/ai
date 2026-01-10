from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import mlflow
import mlflow.transformers

# Vybraný HF model (změněno na veřejný model kvůli gated GPT2)
model_name = "gpt2"

# Stáhnout model a tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Přidat pad token pokud chybí
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Vytvořit pipeline pro text generation
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Připravit jednoduchý input_example
sample_text = ["Hello, how are you?"]

with mlflow.start_run() as run:
    # Log parametrů (volitelné)
    mlflow.log_params({"model_name": model_name, "num_parameters": model.num_parameters()})

    # Log modelu a registrace
    mlflow.transformers.log_model(
        transformers_model=pipe,
        name="gpt2-model",
        input_example=sample_text,
        pip_requirements=["transformers", "torch"],
        registered_model_name="gpt2-model"
    )

    print(f"Model zaregistrován: {run.info.run_id}")