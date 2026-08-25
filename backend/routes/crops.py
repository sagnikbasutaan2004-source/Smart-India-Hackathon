from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Depends
from bson import ObjectId
from backend.database.mongodb import get_db
from backend.schemas.schemas import CreateCropRequest
from backend.utils.auth import get_current_user

router = APIRouter(prefix="/api/crops", tags=["crops"])


def _serialize(doc: dict) -> dict:
    doc["id"] = str(doc.pop("_id"))
    return doc


@router.post("", status_code=201)
async def create_crop(body: CreateCropRequest, current_user: dict = Depends(get_current_user)):
    db = get_db()
    # Verify farm ownership
    farm = await db["farms"].find_one({"_id": ObjectId(body.farm_id), "user_id": str(current_user["_id"])})
    if not farm:
        raise HTTPException(status_code=404, detail="Farm not found")
    doc = {
        "farm_id": body.farm_id,
        "user_id": str(current_user["_id"]),
        "crop_name": body.crop_name,
        "variety": body.variety,
        "season": body.season,
        "sowing_date": body.sowing_date,
        "expected_harvest_date": body.expected_harvest_date,
        "area": body.area,
        "area_unit": body.area_unit,
        "status": "active",
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }
    result = await db["crops"].insert_one(doc)
    doc["id"] = str(result.inserted_id)
    doc.pop("_id", None)
    return doc


@router.get("")
async def list_crops(current_user: dict = Depends(get_current_user)):
    db = get_db()
    cursor = db["crops"].find({"user_id": str(current_user["_id"])})
    crops = []
    async for c in cursor:
        crops.append(_serialize(c))
    return crops


@router.get("/{crop_id}")
async def get_crop(crop_id: str, current_user: dict = Depends(get_current_user)):
    db = get_db()
    crop = await db["crops"].find_one({"_id": ObjectId(crop_id), "user_id": str(current_user["_id"])})
    if not crop:
        raise HTTPException(status_code=404, detail="Crop not found")
    return _serialize(crop)


@router.put("/{crop_id}")
async def update_crop(crop_id: str, body: CreateCropRequest, current_user: dict = Depends(get_current_user)):
    db = get_db()
    update = {
        "crop_name": body.crop_name,
        "variety": body.variety,
        "season": body.season,
        "area": body.area,
        "area_unit": body.area_unit,
        "updated_at": datetime.now(timezone.utc),
    }
    result = await db["crops"].update_one(
        {"_id": ObjectId(crop_id), "user_id": str(current_user["_id"])},
        {"$set": update}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Crop not found")
    return await get_crop(crop_id, current_user)


@router.delete("/{crop_id}")
async def delete_crop(crop_id: str, current_user: dict = Depends(get_current_user)):
    db = get_db()
    result = await db["crops"].delete_one({"_id": ObjectId(crop_id), "user_id": str(current_user["_id"])})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Crop not found")
    return {"message": "Crop deleted"}
