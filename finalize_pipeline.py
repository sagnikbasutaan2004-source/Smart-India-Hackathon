"""Finalization driver: runs the remaining pipeline steps (weather, spatial
epidemiology, GenAI scenario analysis) on top of already-generated artifacts,
then refreshes artifacts/artifact_index.json for the Streamlit dashboard."""
import os
import json
import numpy as np
from PIL import Image

import reconstruction as rc


def load_zones_from_manifest(manifest_path="./metadata/zone_manifest.json"):
    with open(manifest_path) as f:
        man = json.load(f)
    zones = []
    for zm in man["micro_zones"]:
        crop = np.array(Image.open(zm["crop_image"]).convert("RGB"))[:, :, ::-1].copy()
        zones.append({
            "zone_id": zm["zone_id"],
            "grid_row": zm["grid_position"]["row"],
            "grid_col": zm["grid_position"]["col"],
            "bbox": [zm["pixel_bbox"]["x1"], zm["pixel_bbox"]["y1"],
                     zm["pixel_bbox"]["x2"], zm["pixel_bbox"]["y2"]],
            "image_crop": crop,
        })
    return zones, man


def main():
    print("=" * 60)
    print("FINALIZATION: weather -> epidemiology -> GenAI scenario")
    print("=" * 60)

    zones, manifest = load_zones_from_manifest()
    gps_bounds = manifest["orthomosaic"]["gps_bounds"]
    source_metadatas = [
        {"file": os.path.join("./Kaggle image", s["filename"]),
         "lat": s["gps"]["lat"], "lon": s["gps"]["lon"],
         "height_m": s["flight_height_m"],
         "size": (s["original_size"]["width_px"], s["original_size"]["height_px"])}
        for s in manifest["source_images"]
    ]
    print(f"Loaded {len(zones)} zones | GPS center ({gps_bounds['lat_center']:.5f}, {gps_bounds['lon_center']:.5f})")

    with open("./metadata/health_report.json") as f:
        health_records = json.load(f)["zones"]
    with open("./metadata/pathogen_report.json") as f:
        pathogen_summaries = json.load(f)["zones"]

    print("\nSTEP A: Weather & Environmental Conditions (Open-Meteo)")
    weather_json_path, weather_summary = rc.fetch_historical_weather(
        gps_bounds, source_metadatas, out_dir="./metadata")

    print("\nSTEP B: Spatial Epidemiology (Getis-Ord Gi* hot/cold spots)")
    zone_manifest_path = "./metadata/zone_manifest.json"
    epi_json_path, epi_manifest, epi_overlay_path = rc.analyze_spatial_epidemiology(
        zones, health_records, pathogen_summaries, gps_bounds, zone_manifest_path,
        out_dir="./epidemiology")

    print("\nSTEP C: GenAI Scenario Analysis (full report)")
    scenario_json_path, scenario_report = rc.run_genai_scenario_analysis(
        weather_json_path, epi_json_path,
        "./metadata/health_report.json", "./metadata/pathogen_report.json",
        out_dir="./metadata")

    print("\nSTEP D: Refresh artifact index for dashboard")
    artifact_index = {
        "pipeline_version": "4.0-complete-weather-epi-genai",
        "stitching_stages": sorted(os.listdir(rc.ARTIFACTS_DIR)),
        "orthomosaic_files": {
            "raw_no_annotation": "./artifacts/03_final_stitched_canvas_NO_ANNOTATION_RAW.jpg",
            "annotated_grid": "./field_twin_grid.jpg",
            "health_overlay": "./field_health_overlay.jpg",
            "pathogen_overlay": "./field_pathogen_overlay.jpg",
            "epidemiology_overlay": epi_overlay_path,
        },
        "micro_zones_dir": "./micro_zones",
        "health_heatmaps_dir": "./health_heatmaps",
        "detections_dir": "./detections",
        "epidemiology_dir": "./epidemiology",
        "metadata_files": {
            "zone_manifest": zone_manifest_path,
            "health_report_json": "./metadata/health_report.json",
            "health_report_csv": "./metadata/health_report.csv",
            "pathogen_report_json": "./metadata/pathogen_report.json",
            "pathogen_detections_csv": "./metadata/pathogen_detections.csv",
            "pathogen_zone_summary_csv": "./metadata/pathogen_zone_summary.csv",
            "weather_report_json": weather_json_path,
            "weather_hourly_csv": "./metadata/weather_hourly.csv",
            "weather_daily_summary_csv": "./metadata/weather_daily_summary.csv",
            "epidemiology_report_json": epi_json_path,
            "epidemiology_zones_csv": "./epidemiology/epidemiology_zones.csv",
            "epidemiology_adjacency_csv": "./epidemiology/epidemiology_adjacency.csv",
            "scenario_analysis_report_json": scenario_json_path,
        },
        "zone_count": len(zones),
        "gps_center": {"lat": gps_bounds["lat_center"], "lon": gps_bounds["lon_center"]},
    }
    idx_path = os.path.join(rc.ARTIFACTS_DIR, "artifact_index.json")
    with open(idx_path, "w") as f:
        json.dump(artifact_index, f, indent=2)
    print(f"Artifact index refreshed -> {idx_path}")

    ag = weather_summary["inferred_agro_conditions"]
    print("\nSUMMARY")
    print(f"  Weather fetch_success : {weather_summary['fetch_success']}  capture_day={weather_summary['daily_summary']['capture_day'].get('date')}")
    print(f"  Hotspot counts        : {epi_manifest['field_summary']['hotspot_counts']}")
    print(f"  Scenario engine       : {scenario_report['genai_status']['source']} ({scenario_report['genai_status']['model']})")
    print(f"  Agro flags            : wetness={ag['high_leaf_wetness_risk']} rust={ag['favorable_rust_conditions']} spot={ag['favorable_leafspot_conditions']} chlorosis={ag['favorable_chlorosis_conditions']}")
    print("PIPELINE FULLY COMPLETE")


if __name__ == "__main__":
    main()
