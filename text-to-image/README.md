# UltraFlux Text-to-Image API

Toto API umožňuje generovat obrázky z textových promptů pomocí modelu UltraFlux z Hugging Face.

## Jak spustit

1. Nainstaluj závislosti:
   ```
   pip install -r requirements.txt
   ```

2. Nastav environment proměnnou `HF_TOKEN` s tvým Hugging Face tokenem (pokud je potřeba pro přístup k modelu).

3. Spusť server:
   ```
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

Server bude běžet na `http://localhost:8000`.

## Použití API

API má endpoint `/generate` pro POST požadavky.

### Parametry požadavku

- `prompt` (string, povinný): Textový popis obrázku, který chcete vygenerovat.
- `height` (int, volitelný, výchozí 4096): Výška generovaného obrázku v pixelech.
- `width` (int, volitelný, výchozí 4096): Šířka generovaného obrázku v pixelech.
- `guidance_scale` (float, volitelný, výchozí 4.0): Síla guidance.
- `num_inference_steps` (int, volitelný, výchozí 50): Počet kroků inference.
- `max_sequence_length` (int, volitelný, výchozí 512): Maximální délka sekvence pro prompt.
- `seed` (int, volitelný, výchozí 0): Seed pro generátor náhodných čísel pro reprodukovatelné výsledky.

### Příklad volání API pomocí curl

```
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Krásná krajina s horami a jezerem", "height": 4096, "width": 4096, "guidance_scale": 4.0, "num_inference_steps": 50, "max_sequence_length": 512, "seed": 0}' \
     | jq -r '.image_base64' | base64 -d > obrazek.png
```

Tento příkaz vygeneruje obrázek podle promptu "Krásná krajina s horami a jezerem" a uloží ho do souboru `obrazek.png`.

### Příklad volání API pomocí Python

```python
import requests
import base64

# Volání API
response = requests.post("http://localhost:8000/generate", json={
    "prompt": "Krásná krajina s horami a jezerem",
    "height": 4096,
    "width": 4096,
    "guidance_scale": 4.0,
    "num_inference_steps": 50,
    "max_sequence_length": 512,
    "seed": 0
})

# Získání dat
data = response.json()
image_data = base64.b64decode(data['image_base64'])

# Uložení obrázku
with open('obrazek.png', 'wb') as f:
    f.write(image_data)

print("Obrázek uložen jako obrazek.png")
```

## Poznámky

- Model používá UltraFlux pipeline s vlastními komponentami (VAE, transformer) z Hugging Face repository "Owen777/UltraFlux-v1".
- Scheduler je nastaven s use_dynamic_shifting=False a time_shift=4.
- Pro GPU podporu je potřeba PyTorch s CUDA, jinak se použije CPU (pomalejší).
- Obrázky jsou vráceny jako base64 kódovaný PNG.