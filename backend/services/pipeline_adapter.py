"""
pipeline_adapter.py
Reads the existing JSON/CSV outputs from the AgriDrone pipeline and translates
them into MongoDB-ready documents. Does NOT re-run or duplicate any pipeline logic.
"""
import json
import os
from pathlib import Path
from typing import Optional
from backend.config import settings

BASE = Path(settings.PIPELINE_BASE_DIR)


def _load_json(rel_path: str) -> Optional[dict]:
    p = BASE / rel_path
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return None


# ─── Health ──────────────────────────────────────────────────────────────────

def get_health_summary() -> Optional[dict]:
    report = _load_json("metadata/health_report.json")
    if not report:
        return None
    fs = report.get("field_summary", {})
    severity_dist = fs.get("severity_distribution", {})
    healthy = severity_dist.get("HEALTHY", 0)
    mild = severity_dist.get("MILD_STRESS", 0)
    moderate = severity_dist.get("MODERATE_STRESS", 0)
    severe = severity_dist.get("SEVERE_STRESS", 0)
    return {
        "zone_count": fs.get("zone_count", 9),
        "mean_field_score": fs.get("mean_field_score"),
        "mean_stress_pct": fs.get("mean_stress_pct"),
        "mean_vegetation_pct": fs.get("mean_vegetation_pct"),
        "severity_distribution": severity_dist,
        "healthy_zones": healthy,
        "mild_zones": mild,
        "moderate_zones": moderate,
        "severe_zones": severe,
        "gps_center": fs.get("gps_center"),
        "zones": report.get("zones", []),
    }


# ─── Pathogen ─────────────────────────────────────────────────────────────────

def get_pathogen_summary() -> Optional[dict]:
    report = _load_json("metadata/pathogen_report.json")
    if not report:
        return None
    fs = report.get("field_summary", {})
    return {
        "total_zones": fs.get("total_zones", 9),
        "total_detections": fs.get("total_detections"),
        "class_counts": fs.get("class_counts", {}),
        "severity_counts": fs.get("severity_counts", {}),
        "mean_zone_coverage_pct": fs.get("mean_zone_coverage_pct"),
        "max_zone_coverage_pct": fs.get("max_zone_coverage_pct"),
        "zones": report.get("zones", []),
    }


# ─── Zone Manifest ────────────────────────────────────────────────────────────

def get_zone_manifest() -> Optional[dict]:
    return _load_json("metadata/zone_manifest.json")


# ─── Epidemiology ─────────────────────────────────────────────────────────────

def get_epidemiology_summary() -> Optional[dict]:
    """Load epidemiology report from the artifacts index if available."""
    artifacts = _load_json("artifacts/artifact_index.json")
    if artifacts:
        epi = artifacts.get("epidemiology", {})
        if epi:
            return epi
    # Try direct file
    return _load_json("metadata/epidemiology_report.json")


# ─── Scenario Analysis ────────────────────────────────────────────────────────

def get_scenario_analysis() -> Optional[dict]:
    return _load_json("metadata/scenario_analysis_report.json")


# ─── Weather ──────────────────────────────────────────────────────────────────

def get_weather_summary() -> Optional[dict]:
    report = _load_json("metadata/weather_report.json")
    if not report:
        return None

    capture = report.get("capture_day_conditions", {})
    prior = report.get("prior_7day_accumulation", {})
    risks = report.get("agro_risk_flags", {})

    return {
        "capture_date": report.get("capture_date"),
        "gps_center": report.get("gps_center"),
        "temperature_min": capture.get("temperature_min_c"),
        "temperature_max": capture.get("temperature_max_c"),
        "humidity_mean": capture.get("relative_humidity_mean_pct"),
        "rainfall_mm": capture.get("precipitation_sum_mm"),
        "seven_day_rain_mm": prior.get("total_precipitation_mm"),
        "seven_day_avg_rh": prior.get("mean_rh_pct"),
        "seven_day_avg_tmax": prior.get("mean_tmax_c"),
        "high_humidity_risk": risks.get("high_leaf_wetness_risk"),
        "rust_favorable": risks.get("rust_favorable_conditions"),
        "spot_favorable": risks.get("leaf_spot_favorable_conditions"),
        "chlorosis_risk": risks.get("chlorosis_favorable_conditions"),
    }


# ─── Build FieldZone documents from pipeline outputs ─────────────────────────

def build_field_zone_docs(user_id: str = None, farm_id: str = None, crop_id: str = None) -> list:
    """
    Combine health_report + pathogen_report + zone_manifest + epidemiology
    into per-zone documents ready for MongoDB insertion.
    """
    health = _load_json("metadata/health_report.json") or {}
    pathogen = _load_json("metadata/pathogen_report.json") or {}
    manifest = _load_json("metadata/zone_manifest.json") or {}
    artifacts = _load_json("artifacts/artifact_index.json") or {}

    health_zones = {z["zone_id"]: z for z in health.get("zones", [])}
    pathogen_zones = {z["zone_id"]: z for z in pathogen.get("zones", [])}
    manifest_zones = {z["zone_id"]: z for z in manifest.get("micro_zones", [])}

    # Try to get epidemiology per-zone data from artifacts
    epi_zones = {}
    epi_data = artifacts.get("epidemiology", {})
    if isinstance(epi_data, dict):
        for z in epi_data.get("zones", []):
            epi_zones[z.get("zone_id", "")] = z

    docs = []
    zone_ids = sorted(set(list(health_zones.keys()) + list(pathogen_zones.keys()) + list(manifest_zones.keys())))

    for zone_id in zone_ids:
        h = health_zones.get(zone_id, {})
        p = pathogen_zones.get(zone_id, {})
        m = manifest_zones.get(zone_id, {})
        e = epi_zones.get(zone_id, {})

        zone_num = int(zone_id.replace("Z", "")) if zone_id.startswith("Z") else 0
        health_data = h.get("health", {})
        severity_label = health_data.get("severity_label", "unknown")

        # Map pipeline health labels to API labels
        label_map = {
            "HEALTHY": "healthy",
            "MILD_STRESS": "mild",
            "MODERATE_STRESS": "moderate",
            "SEVERE_STRESS": "severe",
        }
        health_status = label_map.get(severity_label, "unknown")

        pathogen_sev = p.get("pathogen_severity", "NONE")
        risk_map = {"HIGH": "high", "MEDIUM": "medium", "LOW": "low", "NONE": "low"}
        risk_level = risk_map.get(pathogen_sev, "unknown")

        gps_bounds = m.get("gps_bounds", {})
        centroid = gps_bounds.get("center", {})

        doc = {
            "zone_id": zone_id,
            "zone_name": f"Zone {zone_num}",
            "zone_number": zone_num,
            "health_status": health_status,
            "risk_level": risk_level,
            "composite_health_score": health_data.get("composite_score"),
            "severity_label": severity_label,
            "pathogen_severity": pathogen_sev,
            "detection_count": p.get("detection_count", 0),
            "zone_coverage_pct": p.get("zone_coverage_pct"),
            "gi_star": e.get("gi_star"),
            "hotspot_class": e.get("hotspot_class"),
            "spread_risk": e.get("spread_risk"),
            "gps_centroid": centroid if centroid else None,
            "geometry": {
                "type": "Polygon",
                "gps_bounds": gps_bounds,
                "pixel_bbox": m.get("pixel_bbox"),
                "grid_position": m.get("grid_position") or h.get("grid_position"),
            },
            "source_files": {
                "health_heatmap": health_data.get("files", {}).get("heatmap") if h.get("files") else h.get("health", {}).get("files", {}).get("heatmap"),
                "micro_zone_image": m.get("crop_image"),
                "pathogen_annotated": p.get("files", {}).get("final_annotated"),
                "pathogen_heatmap": p.get("files", {}).get("heatmap"),
            },
            "user_id": user_id,
            "farm_id": farm_id,
            "crop_id": crop_id,
        }
        docs.append(doc)
    return docs


# ─── Build DiseaseScan document from pipeline outputs ────────────────────────

def build_scan_document(user_id: str, farm_id: str = None, crop_id: str = None) -> dict:
    health = get_health_summary()
    pathogen = get_pathogen_summary()
    weather = get_weather_summary()
    scenario = get_scenario_analysis()

    # Determine dominant disease
    dominant_disease = "Unknown"
    confidence = 0.0
    if pathogen:
        class_counts = pathogen.get("class_counts", {})
        if class_counts:
            dominant_disease = max(class_counts, key=class_counts.get)
            # Approximate confidence from mean coverage
            confidence = round(min(0.99, (pathogen.get("mean_zone_coverage_pct") or 0) / 100 * 2), 2)

    # Severity
    coverage = pathogen.get("mean_zone_coverage_pct") if pathogen else None
    if coverage is None:
        sev_level = "unknown"
    elif coverage > 25:
        sev_level = "high"
    elif coverage > 10:
        sev_level = "medium"
    else:
        sev_level = "low"

    return {
        "user_id": user_id,
        "farm_id": farm_id,
        "crop_id": crop_id,
        "classification": {
            "crop": "paddy",
            "disease": dominant_disease,
            "confidence": confidence,
        },
        "severity": {
            "affected_area_percentage": coverage,
            "level": sev_level,
        },
        "field_summary": health,
        "pathogen_summary": pathogen,
        "epidemiology": get_epidemiology_summary(),
        "scenario_analysis": scenario,
        "pipeline_metadata": {
            "pipeline_version": "v3.0",
            "source": "agridrone_digital_twin",
            "engine": (scenario or {}).get("genai_status", {}).get("source", "rule_based"),
        },
        "image": {
            "original_url": "/pipeline/field_twin_grid.jpg",
            "processed_url": "/pipeline/field_health_overlay.jpg",
            "annotated_url": "/pipeline/field_pathogen_overlay.jpg",
        },
    }
