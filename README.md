# GeoIdentifier

GeoIdentifier is an open-source proof-of-concept Python project for identifying the geographic location of an image using a hybrid approach:

- EXIF GPS coordinates (when embedded) for exact resolution.
- Vision models (+geospatial heuristics) when EXIF is missing:
  - GeoCLIP for candidate lat/lon predictions
  - CLIP zero-shot region classification for regional boosts
  - Clustering + density score + optional Mapillary visual similarity
  - Reverse geocoding via Nominatim (OpenStreetMap)

The repository includes a CLI tool and a Gradio web app.

---

## Features

- `GeoIdentifier` class in `src/identifier.py` with:
  - `identify(image_path, top_k=5)` returns candidate predictions and reasoning traces
  - EXIF short-circuit for guaranteed accuracy when GPS metadata exists
  - Ensemble model fallback using GeoCLIP+CLIP
- CLI interface: `cli.py`
- Web interface: `app.py` (Gradio)
- Mapillary plugin support via `MAPILLARY_ACCESS_TOKEN` (optional)
- Confidence scoring and duplicate suppression (100m dedupe)
- Human-readable reverse-geocoded addresses

---

## Quickstart

### 1. Clone repository

```bash
git clone https://github.com/<your-org>/geolocation.git
cd geolocation
```

### 2. Create environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. (Optional) Mapillary support

If you want Mapillary-based reference-image scoring, set your token:

```bash
export MAPILLARY_ACCESS_TOKEN="your_token"
# or create .env with:
# MAPILLARY_ACCESS_TOKEN=your_token
```

---

## Usage

### CLI mode

```bash
python cli.py /path/to/photo.jpg
python cli.py /path/to/photo.jpg --top-k 3 --verbose
```

Output includes:
- selected strategy (`exif` or `ensemble`)
- top candidates with confidence
- address (reverse-geocoded), coordinates, and Google Maps links

### Web mode (Gradio)

```bash
python app.py
```

Then open the local Gradio URL (typically `http://127.0.0.1:7860`) and upload a photo.

- Shows up to 6 predictions (EXIF or ensemble)
- Confidence, readable address, and embedded Google map
- Step-by-step trace of inference

---

## Module API

```python
from src.identifier import GeoIdentifier

geo = GeoIdentifier()
result = geo.identify("example.jpg", top_k=5)

# result is IdentificationResult:
# - result.strategy_used
# - result.predictions (LocationPrediction list)
# - result.process_trace

for p in result.predictions:
    print(p.source, p.lat, p.lon, p.confidence, p.address)
```

### Data types

- `LocationPrediction`: lat, lon, confidence, address, source (`exif`/`ensemble`)
- `IdentificationResult`: predictions list, strategy_used, process_trace, best property

---

## Internals

- `.src/identifier.py` contains the full pipeline.
- `extract_exif_gps(image_path)` reads GPS from EXIF.
- `EnsemblePredictor.predict(image_path, top_k, trace)` returns scored candidates.
- Region prompts and bounds in `EnsemblePredictor.REGION_PROMPTS` / `REGION_BOUNDS`.
- CLIP model uses `openai/clip-vit-base-patch32`.
- GeoCLIP is loaded via `geoclip.GeoCLIP()`.
- Reverse geocoding via `geopy.geocoders.Nominatim`.

---

## How the model narrows this down (core trace example)

This is the step-by-step process:

1. `Loaded image and started location identification pipeline.`
2. `No EXIF GPS metadata found; falling back to visual inference.`
3. `Requested top 5 final predictions; sampling 50 GeoCLIP candidates.`
4. `GeoCLIP returned 50 candidates.`
5. `CLIP region ranking: ex. North America (14.4%), Latin America (13.0%), Europe (13.0%).`
6. `Original/flip embedding agreement percent`
7. `Candidate summaries (first 5 shown)`
8. `Found x clusters: cluster x: size=50, points=[...]` (cluster points are detailed from GeoCLIP candidates.)
9. `Cluster candidate at (lat,long): ex. density=0.237, spread=0.29km, region=0.144, mapillary=0.733, strength=0.322.`
10. `Selected best cluster with x points, spread x km, score x, mapillary x.`
11. `Found x clusters; chose densest cluster with x points and ~x km spread.`
12. `Mapillary nearby visual support score: x%.`
13. `Confidence calibration applied: spread=x, tie=x, flip=x.`
14. `Ranked winning-cluster points by weighted geometric mean and returned centroid-first top 5.`
15. `Filtered to avoid nearby duplicates (<100m), selected 1 candidates from 5 results.`
16. `Reverse-geocoded top candidates into human-readable addresses.`
17. `Final best prediction: lat, long-122.39701 from <METHOD> at x% confidence.`

---

## Troubleshooting

- Slow initial run: model weights download from Hugging Face at first execution.
- Missing package errors: `pip install -r requirements.txt`.
- Mapillary errors: verify `MAPILLARY_ACCESS_TOKEN` and network connectivity.
- If EXIF is present but not recognized, confirm image contains JPEG EXIF GPS and not stripped metadata.

---

## Notes

- Accuracy depends on image content and available training data.
- Intended for exploration and research; not guaranteed for production use.
- Consider rate limits for Nominatim (OpenStreetMap) and Mapillary APIs.

---

## License

MIT