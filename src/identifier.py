"""
geo_identifier/src/identifier.py
---------------------------------
Location identification from images using:
  - EXIF GPS (if present)
  - Ensemble: GeoCLIP + CLIP zero-shot region boost

Usage:
    from src.identifier import GeoIdentifier
    gi = GeoIdentifier()
    result = gi.identify("photo.jpg")
"""

from __future__ import annotations

import io
import logging
import math
import os
import statistics
import types
from dataclasses import dataclass, field
from typing import Any, Optional

from PIL import Image, ExifTags
from geopy.geocoders import Nominatim
import requests

logger = logging.getLogger(__name__)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * r * math.asin(math.sqrt(max(0.0, min(1.0, a))))


def _weighted_geometric_mean(values: list[float], weights: list[float], eps: float = 1e-8) -> float:
    if not values or not weights or len(values) != len(weights):
        return 0.0
    weight_sum = sum(weights)
    if weight_sum <= 0:
        return 0.0
    log_sum = 0.0
    for value, weight in zip(values, weights):
        log_sum += weight * math.log(max(eps, float(value)))
    return math.exp(log_sum / weight_sum)


def _as_feature_tensor(output):
    """Normalize CLIP outputs across transformers versions to a tensor."""
    import torch

    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if isinstance(output, (tuple, list)) and output:
        if isinstance(output[0], torch.Tensor):
            return output[0]
    raise TypeError(f"Unsupported CLIP output type: {type(output).__name__}")


def _load_dotenv_dotenv(path: str = ".env") -> dict[str, str]:
    env: dict[str, str] = {}
    if not os.path.isfile(path):
        return env
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                env[key] = value
    except Exception as e:
        logger.warning("Failed to read .env file %s: %s", path, e)
    return env


@dataclass
class LocationPrediction:
    lat: float
    lon: float
    confidence: float
    address: str = ""
    source: str = ""           # "exif" | "ensemble"


@dataclass
class IdentificationResult:
    predictions: list[LocationPrediction] = field(default_factory=list)
    strategy_used: str = ""
    process_trace: list[str] = field(default_factory=list)

    @property
    def best(self) -> Optional[LocationPrediction]:
        return self.predictions[0] if self.predictions else None


def _dms_to_decimal(dms, ref: str) -> float:
    d, m, s = dms
    decimal = float(d) + float(m) / 60 + float(s) / 3600
    if ref in ("S", "W"):
        decimal = -decimal
    return decimal


def extract_exif_gps(image_path: str) -> Optional[tuple[float, float]]:
    try:
        img = Image.open(image_path)
        exif_reader = getattr(img, "_getexif", None)
        exif_raw = exif_reader() if callable(exif_reader) else None
        if not exif_raw:
            return None
        if not isinstance(exif_raw, dict):
            return None
        exif = {ExifTags.TAGS.get(k, k): v for k, v in exif_raw.items()}
        gps_info = exif.get("GPSInfo")
        if not gps_info:
            return None
        gps = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps_info.items()}
        lat = _dms_to_decimal(gps["GPSLatitude"], gps["GPSLatitudeRef"])
        lon = _dms_to_decimal(gps["GPSLongitude"], gps["GPSLongitudeRef"])
        return lat, lon
    except Exception as e:
        logger.debug("EXIF extraction failed: %s", e)
        return None


class EnsemblePredictor:
    """
    GeoCLIP provides raw lat/lon candidates.
    CLIP zero-shot classifies the image into broad world regions
    and boosts candidates that fall inside the matching region.

        Final score uses three signals combined by weighted geometric mean:
            1) GeoCLIP candidate score
            2) CLIP region alignment score
            3) Cluster density/support score
    """

    @staticmethod
    def _resolve_mapillary_token() -> str:
        token = os.getenv("MAPILLARY_ACCESS_TOKEN", "").strip()
        if token:
            return token

        dotenv_path = os.path.join(os.getcwd(), ".env")
        items = _load_dotenv_dotenv(dotenv_path)
        token = items.get("MAPILLARY_ACCESS_TOKEN", "").strip()
        return token

    REGION_PROMPTS = {
        "East Asia": [
            "a photo in China", "a photo in Japan", "a photo in South Korea",
            "a photo in Taiwan", "a photo in Mongolia", "a photo in North Korea",
            "Chinese architecture", "Japanese street", "Korean cityscape",
            "Beijing skyline", "Tokyo subway", "Seoul Hanok village",
            "temple in Kyoto", "Great Wall of China", "Mount Fuji",
        ],
        "South/Southeast Asia": [
            "a photo in India", "a photo in Thailand", "a photo in Vietnam",
            "a photo in Indonesia", "a photo in Malaysia", "a photo in Philippines",
            "tropical Asian street", "Hindu temple", "Buddhist pagoda",
            "Kerala backwaters", "Bangkok market", "Bali rice terraces",
            "Angkor Wat", "Ho Chi Minh City skyline", "Singapore skyline",
        ],
        "Europe": [
            "a photo in Europe", "European architecture", "cobblestone street Europe",
            "a photo in France", "a photo in Italy", "a photo in Germany",
            "a photo in Spain", "a photo in the UK", "a photo in Scandinavia",
            "Paris cafe", "Venice canal", "Berlin street art",
            "Swiss Alps", "Amsterdam canal", "Greek island village",
        ],
        "Middle East / North Africa": [
            "a photo in the Middle East", "Islamic architecture", "desert city",
            "a photo in Morocco", "a photo in Egypt", "a photo in UAE",
            "a photo in Turkey", "a photo in Saudi Arabia", "a photo in Israel",
            "Cairo bazaar", "Istanbul mosque", "Dubai skyline",
            "Sahara dunes", "Petra facade", "Jerusalem old city",
        ],
        "Sub-Saharan Africa": [
            "a photo in Africa", "savanna landscape", "African city",
            "a photo in Nigeria", "a photo in Kenya", "a photo in South Africa",
            "a photo in Ghana", "a photo in Tanzania", "a photo in Ethiopia",
            "Cape Town harbor", "Nairobi skyline", "Serengeti wildlife",
            "Victoria Falls", "Madagascar forest", "Kilimanjaro",
        ],
        "North America": [
            "a photo in the USA", "American cityscape", "a photo in Canada",
            "a photo in Mexico", "a photo in Central America", "a photo in the Caribbean",
            "North American suburb", "New York street", "Los Angeles skyline",
            "Toronto skyline", "Grand Canyon", "Rocky Mountains",
            "Mexico City plaza", "Havana colonial street",
        ],
        "Latin America": [
            "a photo in Brazil", "a photo in Mexico", "Latin American city",
            "a photo in Argentina", "a photo in Colombia", "a photo in Chile",
            "a photo in Peru", "colorful colonial architecture", "Amazon rainforest",
            "Rio de Janeiro beach", "Buenos Aires avenue", "Machu Picchu",
            "Cartagena old town", "Atacama desert",
        ],
        "Oceania": [
            "a photo in Australia", "a photo in New Zealand", "Australian outback",
            "a photo in Fiji", "a photo in Papua New Guinea", "a photo in Samoa",
            "Sydney Opera House", "Great Barrier Reef", "Milford Sound",
            "Auckland harbor", "Melbourne laneway", "Tasmanian wilderness",
        ],
    }

    REGION_BOUNDS = {
        "East Asia":                  (18,  55,  95, 145),
        "South/Southeast Asia":       (-10, 35,  60, 140),
        "Europe":                     (35,  72, -25,  45),
        "Middle East / North Africa": (15,  40, -18,  65),
        "Sub-Saharan Africa":         (-35, 15, -20,  55),
        "North America":              (15,  85, -170, -50),
        "Latin America":              (-56, 15, -120, -35),
        "Oceania":                    (-50,   0, 110, 180),
    }

    def __init__(self):
        self._geoclip = None
        self._clip_model: Any = None
        self._clip_processor: Any = None
        self._mapillary_token = self._resolve_mapillary_token()
        if self._mapillary_token:
            logger.info("Mapillary token loaded (length=%s).", len(self._mapillary_token))
        else:
            logger.info("Mapillary token is missing; Mapillary similarity will be skipped.")

    def _load_geoclip(self):
        if self._geoclip is None:
            from geoclip import GeoCLIP
            self._geoclip = GeoCLIP()
            # GeoCLIP expects get_image_features() to return a tensor.
            # In transformers>=5 this can be a BaseModelOutputWithPooling.
            clip_model = self._geoclip.image_encoder.CLIP
            original_get_image_features = clip_model.get_image_features

            def _wrapped_get_image_features(this, *args, **kwargs):
                output = original_get_image_features(*args, **kwargs)
                return _as_feature_tensor(output)

            clip_model.get_image_features = types.MethodType(_wrapped_get_image_features, clip_model)
        return self._geoclip

    def _load_clip(self):
        if self._clip_model is None:
            from transformers import CLIPProcessor, CLIPModel
            self._clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self._clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        return self._clip_model, self._clip_processor

    def _embed_image_clip(self, pil_image):
        import torch
        import torch.nn.functional as F
        model, processor = self._load_clip()
        if processor is None:
            raise RuntimeError("CLIP processor failed to initialize")
        image_inputs = processor(images=pil_image, return_tensors="pt")
        with torch.no_grad():
            image_features = _as_feature_tensor(model.get_image_features(**image_inputs))
            image_features = F.normalize(image_features, dim=-1)
        return image_features

    def _classify_region_clip(self, image_path: str) -> tuple[dict[str, float], float]:
        import torch
        import torch.nn.functional as F
        model, processor = self._load_clip()
        if processor is None:
            raise RuntimeError("CLIP processor failed to initialize")
        img = Image.open(image_path).convert("RGB")
        img_flipped = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        image_features = self._embed_image_clip(img)
        image_features_flipped = self._embed_image_clip(img_flipped)
        flip_agreement = float((image_features @ image_features_flipped.T).squeeze().item())

        # Score each region by its best-matching prompt for original and flipped images.
        region_scores: dict[str, float] = {}
        for region, prompts in self.REGION_PROMPTS.items():
            text_inputs = processor(
                text=prompts, return_tensors="pt", padding=True, truncation=True
            )
            with torch.no_grad():
                text_features = _as_feature_tensor(model.get_text_features(**text_inputs))
                text_features = F.normalize(text_features, dim=-1)
            sims = (image_features @ text_features.T).squeeze(0)
            sims_flip = (image_features_flipped @ text_features.T).squeeze(0)
            region_scores[region] = float((sims.max() + sims_flip.max()) / 2.0)

        total = sum(region_scores.values()) or 1.0
        return {k: v / total for k, v in region_scores.items()}, max(0.0, min(1.0, (flip_agreement + 1.0) / 2.0))

    def _build_clusters(
        self,
        candidates: list[dict],
        eps_km: float = 120.0,
        min_points: int = 3,
    ) -> list[list[int]]:
        if not candidates:
            return []

        neighbors: list[list[int]] = []
        for i, ci in enumerate(candidates):
            n = []
            for j, cj in enumerate(candidates):
                d = _haversine_km(ci["lat"], ci["lon"], cj["lat"], cj["lon"])
                if d <= eps_km:
                    n.append(j)
            neighbors.append(n)

        visited = set()
        clusters: list[list[int]] = []
        for i in range(len(candidates)):
            if i in visited:
                continue
            visited.add(i)
            if len(neighbors[i]) < min_points:
                continue
            cluster = set(neighbors[i])
            queue = list(neighbors[i])
            while queue:
                idx = queue.pop()
                if idx not in visited:
                    visited.add(idx)
                    if len(neighbors[idx]) >= min_points:
                        queue.extend(neighbors[idx])
                cluster.add(idx)
            clusters.append(sorted(cluster))

        if clusters:
            return clusters

        fallback = list(range(min(3, len(candidates))))
        return [fallback] if fallback else []

    def _cluster_centroid(self, candidates: list[dict], cluster_indices: list[int]) -> tuple[float, float, float]:
        cluster_points = [candidates[i] for i in cluster_indices]
        med_lat = statistics.median([p["lat"] for p in cluster_points])
        med_lon = statistics.median([p["lon"] for p in cluster_points])

        weighted_lat = 0.0
        weighted_lon = 0.0
        weight_sum = 0.0
        for point in cluster_points:
            d = _haversine_km(point["lat"], point["lon"], med_lat, med_lon)
            w = point["geo_score"] * (1.0 / (1.0 + d))
            weighted_lat += point["lat"] * w
            weighted_lon += point["lon"] * w
            weight_sum += w

        if weight_sum <= 0:
            return med_lat, med_lon, 9999.0

        cent_lat = weighted_lat / weight_sum
        cent_lon = weighted_lon / weight_sum
        spread = sum(_haversine_km(p["lat"], p["lon"], cent_lat, cent_lon) for p in cluster_points) / len(cluster_points)
        return cent_lat, cent_lon, spread

    @staticmethod
    def _meters_to_degree_deltas(lat: float, radius_m: float) -> tuple[float, float]:
        lat_delta = radius_m / 111_320.0
        lon_scale = max(math.cos(math.radians(lat)), 1e-6)
        lon_delta = radius_m / (111_320.0 * lon_scale)
        return lat_delta, lon_delta

    def _build_bbox(self, lat: float, lon: float, radius_m: float) -> str:
        lat_delta, lon_delta = self._meters_to_degree_deltas(lat, radius_m)
        left = lon - lon_delta
        right = lon + lon_delta
        bottom = lat - lat_delta
        top = lat + lat_delta
        return f"{left:.7f},{bottom:.7f},{right:.7f},{top:.7f}"

    def _query_mapillary_images(self, lat: float, lon: float, radius_m: float = 500.0) -> list[dict[str, Any]]:
        if not self._mapillary_token:
            return []

        params = {
            "fields": "id,computed_geometry,thumb_1024_url,captured_at,is_pano",
            "bbox": self._build_bbox(lat, lon, radius_m),
            "limit": 25,
            "is_pano": "false",
        }
        attempt_order = [
            ({**params, "access_token": self._mapillary_token}, {}),
            (params, {"Authorization": f"OAuth {self._mapillary_token}"}),
        ]

        for attempt_params, headers in attempt_order:
            try:
                response = requests.get(
                    "https://graph.mapillary.com/images",
                    params=attempt_params,
                    headers=headers,
                    timeout=8,
                )
                if response.status_code in (401, 403):
                    logger.warning("Mapillary auth failed (status=%s): %s", response.status_code, response.text)
                    continue
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, dict):
                    data = payload.get("data", [])
                    if isinstance(data, list):
                        return data
            except Exception as e:
                logger.warning("Mapillary /images query failed: %s", e)
                continue
        return []

    def _mapillary_similarity(self, image_path: str, lat: float, lon: float) -> Optional[float]:
        if not self._mapillary_token:
            logger.info("Mapillary token was empty at _mapillary_similarity time; skipping Mapillary check.")
            return None

        try:
            query_image = Image.open(image_path).convert("RGB")
            query_features = self._embed_image_clip(query_image)

            data = self._query_mapillary_images(lat=lat, lon=lon, radius_m=80.0)
            if not data:
                logger.info("Mapillary returned no nearby images around %s,%s", lat, lon)
                return None

            best_visual = -1.0
            best_distance_km = float("inf")
            for item in data:
                url = item.get("thumb_1024_url")
                if not url:
                    continue
                coords = (item.get("computed_geometry") or {}).get("coordinates")
                if not isinstance(coords, list) or len(coords) < 2:
                    continue
                img_lon, img_lat = coords[0], coords[1]
                if not isinstance(img_lat, (int, float)) or not isinstance(img_lon, (int, float)):
                    continue

                distance_km = _haversine_km(lat, lon, float(img_lat), float(img_lon))
                img_resp = requests.get(url, timeout=8)
                try:
                    img_resp.raise_for_status()
                except Exception as e:
                    logger.warning("Mapillary thumbnail download failed: %s", e)
                    continue
                candidate_img = Image.open(io.BytesIO(img_resp.content)).convert("RGB")
                candidate_features = self._embed_image_clip(candidate_img)
                sim = float((query_features @ candidate_features.T).squeeze().item())

                if sim > best_visual:
                    best_visual = sim
                    best_distance_km = distance_km

            if best_visual < -0.5:
                return None

            visual_score = max(0.0, min(1.0, (best_visual + 1.0) / 2.0))
            # Prefer images whose computed_geometry is very close to the target coordinate.
            distance_score = max(0.0, min(1.0, 1.0 - (best_distance_km / 0.08)))
            return max(0.0, min(1.0, 0.7 * visual_score + 0.3 * distance_score))
        except Exception as e:
            logger.info("Mapillary similarity check skipped/failed: %s", e)
            return None

    def _in_region(self, lat: float, lon: float, region: str) -> bool:
        bounds = self.REGION_BOUNDS.get(region)
        if not bounds:
            return False
        lat_min, lat_max, lon_min, lon_max = bounds
        return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max

    def predict(
        self, image_path: str, top_k: int = 5, trace: Optional[list[str]] = None
    ) -> list[tuple[tuple[float, float], float]]:
        trace = trace if trace is not None else []
        geoclip = self._load_geoclip()
        n_candidates = min(100, max(50, top_k * 10))
        trace.append(f"Requested top {top_k} final predictions; sampling {n_candidates} GeoCLIP candidates.")
        geo_preds, geo_probs = geoclip.predict(image_path, top_k=n_candidates)
        trace.append(f"GeoCLIP returned {len(geo_preds)} candidates.")

        try:
            region_probs, flip_agreement = self._classify_region_clip(image_path)
            logger.info("CLIP region scores: %s",
                        {k: f"{v:.3f}" for k, v in sorted(region_probs.items(), key=lambda x: -x[1])})
            top_regions = sorted(region_probs.items(), key=lambda x: -x[1])[:3]
            region_summary = ", ".join(f"{name} ({prob:.1%})" for name, prob in top_regions)
            trace.append(f"CLIP region ranking: {region_summary}.")
            trace.append(f"Original/flip embedding agreement: {flip_agreement:.1%}.")
        except Exception as e:
            logger.warning("CLIP region classification failed: %s", e)
            region_probs = {}
            flip_agreement = 0.5
            trace.append("CLIP region ranking unavailable (classification failed).")

        candidates = []
        for (lat, lon), gp in zip(geo_preds, geo_probs):
            lat = float(lat)
            lon = float(lon)
            gp = max(1e-8, float(gp))
            region_alignment = 1e-3
            matched_region = "none"
            if region_probs:
                for region, rp in sorted(region_probs.items(), key=lambda x: -x[1]):
                    if self._in_region(lat, lon, region):
                        region_alignment = max(region_alignment, float(rp))
                        matched_region = region
                        break
            candidates.append(
                {
                    "lat": lat,
                    "lon": lon,
                    "geo_score": gp,
                    "region_score": region_alignment,
                    "cluster_score": 1e-3,
                    "composite_score": gp,
                }
            )
            if len(candidates) <= 5:
                if matched_region:
                    trace.append(
                        f"Candidate {lat:.3f}, {lon:.3f}: geo={gp:.3f}, region={region_alignment:.3f} ({matched_region})."
                    )
                else:
                    trace.append(f"Candidate {lat:.3f}, {lon:.3f}: geo={gp:.3f}, region={region_alignment:.3f} (no region match).")

        clusters = self._build_clusters(candidates, eps_km=120.0, min_points=3)
        if not clusters:
            return []

        # Add logging and trace for the clusters that were generated.
        cluster_summary = []
        for i, cluster in enumerate(clusters, start=1):
            cluster_locs = [(candidates[idx]["lat"], candidates[idx]["lon"]) for idx in cluster]
            cluster_summary.append(f"cluster {i}: size={len(cluster)}, points={cluster_locs}")
        logger.info("Found %d clusters: %s", len(clusters), "; ".join(cluster_summary))
        trace.append(f"Found {len(clusters)} clusters: {', '.join(cluster_summary)}")

        best_cluster = clusters[0]
        best_cluster_strength = -1.0
        best_cluster_mapillary = None

        for cluster_indices in clusters:
            cent_lat, cent_lon, spread_km = self._cluster_centroid(candidates, cluster_indices)

            density_strength = 0.0
            region_scores = []
            for idx in cluster_indices:
                point = candidates[idx]
                d = _haversine_km(point["lat"], point["lon"], cent_lat, cent_lon)
                density_strength += point["geo_score"] * (1.0 / (1.0 + d))
                region_scores.append(point.get("region_score", 0.0))

            avg_region = statistics.mean(region_scores) if region_scores else 0.0
            tightness = 1.0 / (1.0 + spread_km)

            if self._mapillary_token:
                mapillary_score = self._mapillary_similarity(image_path, cent_lat, cent_lon)
                mapillary_score = max(0.01, mapillary_score) if mapillary_score is not None else 0.1
            else:
                mapillary_score = 0.75

            cluster_strength = _weighted_geometric_mean(
                [density_strength + 1e-8, avg_region + 1e-8, tightness + 1e-8, mapillary_score + 1e-8],
                [0.45, 0.20, 0.20, 0.15],
            )

            trace.append(
                f"Cluster candidate at {cent_lat:.4f},{cent_lon:.4f}: density={density_strength:.3f},"
                f" spread={spread_km:.2f}km, region={avg_region:.3f}, mapillary={mapillary_score:.3f},"
                f" strength={cluster_strength:.3f}."
            )

            if cluster_strength > best_cluster_strength:
                best_cluster = cluster_indices
                best_cluster_strength = cluster_strength
                best_cluster_mapillary = mapillary_score

        centroid_lat, centroid_lon, centroid_spread_km = self._cluster_centroid(candidates, best_cluster)
        trace.append(
            f"Selected best cluster with {len(best_cluster)} points, spread {centroid_spread_km:.1f}km,"
            f" score {best_cluster_strength:.3f}, mapillary {best_cluster_mapillary:.3f}."
        )

        centroid_lat, centroid_lon, centroid_spread_km = self._cluster_centroid(candidates, best_cluster)
        trace.append(
            f"Found {len(clusters)} clusters; chose densest cluster with {len(best_cluster)} points and ~{centroid_spread_km:.1f}km spread."
        )

        # Inverse-distance density signal inside winning cluster.
        cluster_points = [candidates[i] for i in best_cluster]
        for i in best_cluster:
            pi = candidates[i]
            density = 0.0
            for pj in cluster_points:
                d = _haversine_km(pi["lat"], pi["lon"], pj["lat"], pj["lon"])
                density += 1.0 / (1.0 + d)
            pi["cluster_score"] = density

        max_cluster_score = max(candidates[i]["cluster_score"] for i in best_cluster) if best_cluster else 1.0
        for i in best_cluster:
            c = candidates[i]
            c["cluster_score"] = max(1e-8, c["cluster_score"] / max_cluster_score)
            c["composite_score"] = _weighted_geometric_mean(
                [c["geo_score"], c["region_score"], c["cluster_score"]],
                [0.50, 0.20, 0.30],
            )

        ranked_cluster = sorted((candidates[i] for i in best_cluster), key=lambda x: -x["composite_score"])
        if not ranked_cluster:
            return []

        mapillary_support = self._mapillary_similarity(image_path, centroid_lat, centroid_lon)
        if mapillary_support is not None:
            trace.append(f"Mapillary nearby visual support score: {mapillary_support:.1%}.")
        else:
            trace.append("Mapillary visual support unavailable (no token, no nearby images, or request failed).")

        top_scores = [p["composite_score"] for p in ranked_cluster[:3]]
        score_gap = (max(top_scores) - min(top_scores)) if len(top_scores) > 1 else 0.0
        spread_penalty = max(0.35, min(1.0, 1.0 - (centroid_spread_km / 350.0)))
        tie_penalty = max(0.45, min(1.0, 1.0 - (score_gap * 1.8)))
        agreement_penalty = max(0.5, min(1.0, flip_agreement))
        confidence_scale = spread_penalty * tie_penalty * agreement_penalty
        if mapillary_support is not None:
            confidence_scale *= max(0.6, min(1.0, 0.6 + 0.4 * mapillary_support))
        trace.append(
            f"Confidence calibration applied: spread={spread_penalty:.2f}, tie={tie_penalty:.2f}, flip={agreement_penalty:.2f}."
        )

        centroid_region_score = max((p["region_score"] for p in ranked_cluster), default=1e-3)
        centroid_cluster_score = max((p["cluster_score"] for p in ranked_cluster), default=1e-3)
        centroid_geo_score = statistics.mean(p["geo_score"] for p in ranked_cluster[: min(5, len(ranked_cluster))])
        centroid_score = _weighted_geometric_mean(
            [centroid_geo_score, centroid_region_score, centroid_cluster_score],
            [0.50, 0.20, 0.30],
        ) * confidence_scale

        final = [((centroid_lat, centroid_lon), centroid_score)]
        for p in ranked_cluster[: max(1, top_k - 1)]:
            final.append(((p["lat"], p["lon"]), p["composite_score"] * confidence_scale))

        max_score = max(score for _, score in final) if final else 1.0
        normalized = [
            ((lat, lon), max(1e-4, min(1.0, score / (max_score or 1.0))))
            for (lat, lon), score in final
        ]
        normalized.sort(key=lambda x: -x[1])

        # Filter out points within 100 meters of a higher-confidence candidate,
        # but continue scanning normalized list until we have top_k final outputs.
        filtered = []
        for (lat, lon), score in normalized:
            if len(filtered) >= top_k:
                break
            keep = True
            for (k_lat, k_lon), _ in filtered:
                if _haversine_km(lat, lon, k_lat, k_lon) <= 0.1:
                    keep = False
                    break
            if keep:
                filtered.append(((lat, lon), score))

        trace.append(
            f"Ranked winning-cluster points by weighted geometric mean and returned centroid-first top {top_k}."
        )
        trace.append(
            f"Filtered to avoid nearby duplicates (<100m), selected {len(filtered)} candidates from {len(normalized)} results."
        )
        return filtered


class ReverseGeocoder:
    def __init__(self):
        self._geolocator: Any = Nominatim(user_agent="geo-identifier-app")

    def lookup(self, lat: float, lon: float) -> str:
        try:
            location = self._geolocator.reverse(f"{lat}, {lon}", language="en")
            if not location:
                return f"{lat:.4f}, {lon:.4f}"
            return str(getattr(location, "address", f"{lat:.4f}, {lon:.4f}"))
        except Exception as e:
            logger.warning("Reverse geocode failed: %s", e)
            return f"{lat:.4f}, {lon:.4f}"

    def lookup_region(self, lat: float, lon: float) -> str:
        try:
            location = self._geolocator.reverse(f"{lat}, {lon}", language="en")
            if not location:
                return f"{lat:.4f}, {lon:.4f}"
            raw = getattr(location, "raw", {}) or {}
            address = raw.get("address", {}) if isinstance(raw, dict) else {}
            region = (
                address.get("state")
                or address.get("region")
                or address.get("county")
                or address.get("country")
            )
            country = address.get("country")
            if region and country and region != country:
                return f"{region}, {country}"
            return region or country or str(getattr(location, "address", f"{lat:.4f}, {lon:.4f}"))
        except Exception as e:
            logger.warning("Reverse region geocode failed: %s", e)
            return f"{lat:.4f}, {lon:.4f}"


class GeoIdentifier:
    """
    Priority order:
      1. EXIF GPS  — exact, if present
      2. Ensemble  — GeoCLIP + CLIP region boost
    """

    def __init__(self):
        self.ensemble = EnsemblePredictor()
        self.geocoder = ReverseGeocoder()

    def identify(self, image_path: str, top_k: int = 5) -> IdentificationResult:
        result = IdentificationResult()
        result.process_trace.append("Loaded image and started location identification pipeline.")

        exif_gps = extract_exif_gps(image_path)
        if exif_gps:
            lat, lon = exif_gps
            result.predictions.append(LocationPrediction(
                lat=lat, lon=lon, confidence=1.0,
                address=self.geocoder.lookup(lat, lon),
                source="exif",
            ))
            result.strategy_used = "exif"
            logger.info("EXIF GPS found.")
            result.process_trace.append(
                f"EXIF GPS found exact coordinates at {lat:.5f}, {lon:.5f}; kept as highest-priority prediction."
            )
        else:
            result.process_trace.append("No EXIF GPS metadata found; falling back to visual inference.")

        try:
            for (lat, lon), score in self.ensemble.predict(
                image_path, top_k=top_k, trace=result.process_trace
            ):
                result.predictions.append(LocationPrediction(
                    lat=lat, lon=lon, confidence=round(score, 4),
                    address=self.geocoder.lookup(lat, lon),
                    source="ensemble",
                ))
            if not result.strategy_used:
                result.strategy_used = "ensemble"
            result.process_trace.append("Reverse-geocoded top candidates into human-readable addresses.")
        except Exception as e:
            logger.error("Ensemble prediction failed: %s", e)
            result.process_trace.append(f"Ensemble stage failed with error: {e}")

        result.predictions.sort(key=lambda p: (0 if p.source == "exif" else 1, -p.confidence))
        ensemble_top = [p for p in result.predictions if p.source == "ensemble"][:3]
        if len(ensemble_top) == 3:
            pair_dists = [
                _haversine_km(ensemble_top[i].lat, ensemble_top[i].lon, ensemble_top[j].lat, ensemble_top[j].lon)
                for i in range(3)
                for j in range(i + 1, 3)
            ]
            max_pair_dist = max(pair_dists)
            conf_gap = max(p.confidence for p in ensemble_top) - min(p.confidence for p in ensemble_top)
            if max_pair_dist <= 140 and conf_gap <= 0.12:
                region_label = self.geocoder.lookup_region(ensemble_top[0].lat, ensemble_top[0].lon)
                ensemble_top[0].address = f"{region_label} (region-level consensus from tightly clustered top-3)"
                result.process_trace.append(
                    f"Top-3 ensemble candidates are close (max {max_pair_dist:.1f}km, confidence gap {conf_gap:.3f}); presenting region-level result."
                )

        if result.predictions:
            best = result.predictions[0]
            result.process_trace.append(
                f"Final best prediction: {best.lat:.5f}, {best.lon:.5f} from {best.source.upper()} at {best.confidence:.1%} confidence."
            )
        else:
            result.process_trace.append("No valid prediction could be produced.")
        return result