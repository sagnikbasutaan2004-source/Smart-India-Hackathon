from fastapi import APIRouter, Depends
from backend.utils.auth import get_current_user
from backend.database.mongodb import get_db
from backend.services.pipeline_adapter import (
    get_health_summary,
    get_pathogen_summary,
    get_weather_summary,
    get_epidemiology_summary,
    get_scenario_analysis,
)

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


@router.get("/summary")
async def dashboard_summary(current_user: dict = Depends(get_current_user)):
    db = get_db()
    user_id = str(current_user["_id"])

    total_farms = await db["farms"].count_documents({"user_id": user_id})
    total_crops = await db["crops"].count_documents({"user_id": user_id})
    total_scans = await db["disease_scans"].count_documents({"user_id": user_id})

    health = get_health_summary()
    pathogen = get_pathogen_summary()
    weather = get_weather_summary()
    epidemiology = get_epidemiology_summary()

    # Build alerts
    alerts = []
    if pathogen:
        for zone in (pathogen.get("zones") or []):
            if zone.get("pathogen_severity") == "HIGH":
                alerts.append({
                    "type": "warning",
                    "zone": zone["zone_id"],
                    "message": f"High pathogen severity in {zone['zone_id']} — {zone.get('zone_coverage_pct', 0):.1f}% coverage",
                })
    if weather and weather.get("high_humidity_risk"):
        alerts.append({
            "type": "info",
            "message": "High leaf-wetness risk — conditions favorable for fungal infection",
        })

    return {
        "user": {
            "name": current_user.get("name"),
            "profile": current_user.get("profile"),
            "location": current_user.get("location"),
        },
        "total_farms": total_farms,
        "total_crops": total_crops,
        "total_scans": total_scans,
        "field_health": health,
        "pathogen_summary": pathogen,
        "weather_summary": weather,
        "epidemiology_summary": epidemiology,
        "recent_alerts": alerts[:5],
    }


@router.get("/recent")
async def recent_scans(current_user: dict = Depends(get_current_user)):
    db = get_db()
    cursor = db["disease_scans"].find(
        {"user_id": str(current_user["_id"])},
        sort=[("created_at", -1)]
    ).limit(5)
    scans = []
    async for s in cursor:
        s["id"] = str(s.pop("_id"))
        scans.append({
            "id": s["id"],
            "disease": s.get("classification", {}).get("disease"),
            "confidence": s.get("classification", {}).get("confidence"),
            "severity_level": s.get("severity", {}).get("level"),
            "affected_pct": s.get("severity", {}).get("affected_area_percentage"),
            "created_at": s.get("created_at"),
        })
    return scans


@router.get("/alerts")
async def dashboard_alerts(current_user: dict = Depends(get_current_user)):
    pathogen = get_pathogen_summary()
    weather = get_weather_summary()
    scenario = get_scenario_analysis()
    alerts = []

    if pathogen:
        for zone in (pathogen.get("zones") or []):
            if zone.get("pathogen_severity") == "HIGH":
                classes = zone.get("class_distribution", {})
                dominant = max(classes, key=classes.get) if classes else "Unknown"
                alerts.append({
                    "severity": "high",
                    "zone": zone["zone_id"],
                    "pathogen": dominant,
                    "coverage_pct": zone.get("zone_coverage_pct"),
                    "message": f"{zone['zone_id']}: {dominant} detected at {zone.get('zone_coverage_pct', 0):.1f}% coverage",
                })

    if weather:
        if weather.get("rust_favorable"):
            alerts.append({"severity": "warning", "message": "Rust-favorable conditions — temperature + humidity in danger zone"})
        if weather.get("spot_favorable"):
            alerts.append({"severity": "warning", "message": "Leaf Spot conditions active — high moisture risk"})
        if weather.get("chlorosis_risk"):
            alerts.append({"severity": "info", "message": "Chlorosis risk — excessive rainfall may cause nutrient leaching"})

    return alerts
