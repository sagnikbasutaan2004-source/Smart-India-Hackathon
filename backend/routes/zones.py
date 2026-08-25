from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Depends
from bson import ObjectId
from backend.database.mongodb import get_db
from backend.utils.auth import get_current_user
from backend.services.pipeline_adapter import build_field_zone_docs

router = APIRouter(prefix="/api/zones", tags=["zones"])


def _serialize(doc: dict) -> dict:
    doc["id"] = str(doc.pop("_id"))
    return doc


@router.get("/seed")
async def seed_zones_from_pipeline(
    farm_id: str = None,
    crop_id: str = None,
    current_user: dict = Depends(get_current_user)
):
    """
    Reads existing pipeline outputs and upserts field zones into MongoDB.
    Call this once after running finalize_pipeline.py.
    """
    db = get_db()
    user_id = str(current_user["_id"])
    docs = build_field_zone_docs(user_id=user_id, farm_id=farm_id, crop_id=crop_id)
    upserted = 0
    for doc in docs:
        zone_id = doc["zone_id"]
        doc["updated_at"] = datetime.now(timezone.utc)
        existing = await db["field_zones"].find_one({"zone_id": zone_id, "user_id": user_id})
        if existing:
            await db["field_zones"].update_one({"_id": existing["_id"]}, {"$set": doc})
        else:
            doc["created_at"] = datetime.now(timezone.utc)
            await db["field_zones"].insert_one(doc)
            upserted += 1
    return {"message": f"Seeded {len(docs)} zones ({upserted} new)", "zones": len(docs)}


@router.get("")
async def list_zones(current_user: dict = Depends(get_current_user)):
    db = get_db()
    cursor = db["field_zones"].find({"user_id": str(current_user["_id"])})
    zones = []
    async for z in cursor:
        zones.append(_serialize(z))

    # If no zones in DB yet, serve directly from pipeline adapter
    if not zones:
        from backend.services.pipeline_adapter import build_field_zone_docs
        docs = build_field_zone_docs()
        for i, d in enumerate(docs):
            d["id"] = f"pipeline_{d['zone_id']}"
        return docs
    return zones


@router.get("/{zone_id}")
async def get_zone(zone_id: str, current_user: dict = Depends(get_current_user)):
    db = get_db()
    # Try ObjectId first, then zone_id string
    zone = None
    try:
        zone = await db["field_zones"].find_one({"_id": ObjectId(zone_id), "user_id": str(current_user["_id"])})
    except Exception:
        pass
    if not zone:
        zone = await db["field_zones"].find_one({"zone_id": zone_id, "user_id": str(current_user["_id"])})
    if not zone:
        raise HTTPException(status_code=404, detail="Zone not found")
    return _serialize(zone)
