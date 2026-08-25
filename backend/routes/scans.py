import os
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from bson import ObjectId
from backend.database.mongodb import get_db
from backend.utils.auth import get_current_user
from backend.services.pipeline_adapter import build_scan_document, get_health_summary, get_pathogen_summary

router = APIRouter(prefix="/api/scans", tags=["scans"])


def _serialize(doc: dict) -> dict:
    doc["id"] = str(doc.pop("_id"))
    return doc


@router.post("/analyze", status_code=201)
async def analyze(
    farm_id: str = Form(None),
    crop_id: str = Form(None),
    file: UploadFile = File(None),
    current_user: dict = Depends(get_current_user),
):
    """
    Performs analysis using the existing pipeline outputs.
    If an image is uploaded it is saved for reference but analysis uses
    pre-computed pipeline results (drone survey analysis).
    """
    db = get_db()
    user_id = str(current_user["_id"])

    image_path = None
    if file and file.filename:
        from backend.config import settings
        import aiofiles, pathlib
        uploads_dir = pathlib.Path(settings.PIPELINE_BASE_DIR) / "uploads" / user_id
        uploads_dir.mkdir(parents=True, exist_ok=True)
        ext = os.path.splitext(file.filename)[1]
        image_path = uploads_dir / f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}{ext}"
        content = await file.read()
        async with aiofiles.open(image_path, "wb") as f:
            await f.write(content)

    # Build scan document from pipeline outputs
    scan_doc = build_scan_document(user_id=user_id, farm_id=farm_id, crop_id=crop_id)
    scan_doc["created_at"] = datetime.now(timezone.utc)

    if image_path:
        scan_doc["image"]["uploaded_url"] = str(image_path)

    # Build and save recommendation
    pathogen = get_pathogen_summary() or {}
    health = get_health_summary() or {}
    coverage = (pathogen.get("mean_zone_coverage_pct") or 0)

    if coverage > 25:
        action = "urgent_attention"
    elif coverage > 10:
        action = "targeted_intervention"
    elif coverage > 2:
        action = "inspect"
    else:
        action = "monitor"

    severe_zones = health.get("severe_zones", 0)
    moderate_zones = health.get("moderate_zones", 0)
    detected_pathogens = list((pathogen.get("class_counts") or {}).keys())

    rec_doc = {
        "user_id": user_id,
        "scan_id": None,  # filled after scan insert
        "severity": scan_doc.get("severity", {}).get("level"),
        "affected_area_percentage": coverage,
        "action": action,
        "message": (
            f"Field shows {coverage:.1f}% mean pathogen coverage across {pathogen.get('total_zones', 9)} zones. "
            f"{severe_zones} zones in severe stress, {moderate_zones} in moderate stress. "
            f"Detected: {', '.join(detected_pathogens) or 'None'}. "
            f"Recommended action: {action.replace('_', ' ').title()}."
        ),
        "zones_affected": [z["zone_id"] for z in pathogen.get("zones", []) if z.get("pathogen_severity") == "HIGH"],
        "pathogens_detected": detected_pathogens,
        "created_at": datetime.now(timezone.utc),
    }

    rec_result = await db["recommendations"].insert_one(rec_doc)
    scan_doc["recommendation_id"] = str(rec_result.inserted_id)

    scan_result = await db["disease_scans"].insert_one(scan_doc)
    scan_id = str(scan_result.inserted_id)

    # Update recommendation with scan_id
    await db["recommendations"].update_one(
        {"_id": rec_result.inserted_id},
        {"$set": {"scan_id": scan_id}}
    )

    # Return the full scan document
    inserted = await db["disease_scans"].find_one({"_id": scan_result.inserted_id})
    return _serialize(inserted)


@router.get("")
async def list_scans(current_user: dict = Depends(get_current_user)):
    db = get_db()
    cursor = db["disease_scans"].find(
        {"user_id": str(current_user["_id"])},
        sort=[("created_at", -1)],
    ).limit(50)
    scans = []
    async for s in cursor:
        scans.append(_serialize(s))
    return scans


@router.get("/{scan_id}")
async def get_scan(scan_id: str, current_user: dict = Depends(get_current_user)):
    db = get_db()
    scan = await db["disease_scans"].find_one(
        {"_id": ObjectId(scan_id), "user_id": str(current_user["_id"])}
    )
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    return _serialize(scan)
