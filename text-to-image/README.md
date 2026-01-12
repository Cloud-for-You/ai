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
- `steps` (int, volitelný, výchozí 30): Počet kroků inference.
- `guidance_scale` (float, volitelný, výchozí 7.5): Síla guidance.

### Příklad volání API pomocí curl

```
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Krásná krajina s horami a jezerem", "steps": 30, "guidance_scale": 7.5}' \
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
    "steps": 30,
    "guidance_scale": 7.5
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

- Model se načítá přímo z Hugging Face repository "Owen777/UltraFlux-v1".
- Pro GPU podporu je potřeba PyTorch s CUDA, jinak se použije CPU (pomalejší).
- Obrázky jsou vráceny jako base64 kódovaný PNG.