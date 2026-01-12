# UltraFlux Text-to-Image API

Tato aplikace poskytuje FastAPI server pro generování obrázků pomocí UltraFlux modelu.

## Instalace

1. Nainstalujte závislosti:
```bash
pip install -r requirements.txt
```

## Spuštění

### Lokálně
```bash
python app.py
```

Nebo pomocí Uvicorn:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

API bude dostupné na `http://localhost:8000`.

### Pomocí Docker
```bash
podman build -t ultraflux-api .
podman run -p 8000:8000 ultraflux-api
```

## Použití API

### Endpoint: POST /generate

Generuje obrázky na základě zadaných promptů.

#### Request
```json
{
  "prompts": ["Popis obrázku 1", "Popis obrázku 2"],
  "height": 4096,
  "width": 4096,
  "guidance_scale": 4.0,
  "num_inference_steps": 50,
  "max_sequence_length": 512,
  "seed": 0
}
```

#### Parametry
- `prompts` (povinné): Pole řetězců s popisy obrázků
- `height` (volitelné): Výška obrázku (výchozí: 4096)
- `width` (volitelné): Šířka obrázku (výchozí: 4096)
- `guidance_scale` (volitelné): Síla guidance (výchozí: 4.0)
- `num_inference_steps` (volitelné): Počet kroků inference (výchozí: 50)
- `max_sequence_length` (volitelné): Maximální délka sekvence (výchozí: 512)
- `seed` (volitelné): Seed pro generátor (výchozí: 0)

#### Response
```json
{
  "images": ["results/ultra_flux_01.jpeg", "results/ultra_flux_02.jpeg"]
}
```

Obrázky jsou uloženy v adresáři `results/` a response obsahuje cesty k nim.

## Příklad použití

```python
import requests

url = "http://localhost:8000/generate"
data = {
    "prompts": ["Krásná krajina s horami a jezerem"],
    "height": 2048,
    "width": 2048
}

response = requests.post(url, json=data)
result = response.json()
print(result["images"])
```

## Standalone skript

Pro jednorázové generování můžete použít `inf_ultraflux.py`:

```bash
python inf_ultraflux.py
```

Tento skript použije předdefinované prompty a uloží obrázky do `results/`.

## Poznámky

- Model vyžaduje CUDA pro akceleraci
- První spuštění stáhne váhy modelu (může trvat dlouho)
- Obrázky jsou generovány v vysokém rozlišení (4K)