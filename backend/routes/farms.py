from datetime import datetime, timezone
from typing import List
from fastapi import APIRouter, HTTPException, Depends
from bson import ObjectId
from backend.database.mongodb import get_db
from backend.schemas.schemas import CreateFarmRequest
from backend.utils.auth import get_current_user

router = APIRouter(prefix="/api/farms", tags=["farms"])


def _serialize(doc: dict) -> dict:
    doc["id"] = str(doc.pop("_id"))
    return doc


@router.post("", status_code=201)
async def create_farm(body: CreateFarmRequest, current_user: dict = Depends(get_current_user)):
    db = get_db()
    doc = {
        "user_id": str(current_user["_id"]),
        "farm_name": body.farm_name,
        "location": body.location.model_dump() if body.location else {},
        "total_land_size": body.total_land_size,
        "land_unit": body.land_unit,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }
    result = await db["farms"].insert_one(doc)
    doc["id"] = str(result.inserted_id)
    doc.pop("_id", None)
    return doc


@router.get("")
async def list_farms(current_user: dict = Depends(get_current_user)):
    db = get_db()
    cursor = db["farms"].find({"user_id": str(current_user["_id"])})
    farms = []
    async for f in cursor:
        farms.append(_serialize(f))
    return farms


@router.get("/{farm_id}")
async def get_farm(farm_id: str, current_user: dict = Depends(get_current_user)):
    db = get_db()
    farm = await db["farms"].find_one({"_id": ObjectId(farm_id), "user_id": str(current_user["_id"])})
    if not farm:
        raise HTTPException(status_code=404, detail="Farm not found")
    return _serialize(farm)


@router.put("/{farm_id}")
async def update_farm(farm_id: str, body: CreateFarmRequest, current_user: dict = Depends(get_current_user)):
    db = get_db()
    update = {
        "farm_name": body.farm_name,
        "location": body.location.model_dump() if body.location else {},
        "total_land_size": body.total_land_size,
        "land_unit": body.land_unit,
        "updated_at": datetime.now(timezone.utc),
    }
    result = await db["farms"].update_one(
        {"_id": ObjectId(farm_id), "user_id": str(current_user["_id"])},
        {"$set": update}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Farm not found")
    return await get_farm(farm_id, current_user)


@router.delete("/{farm_id}")
async def delete_farm(farm_id: str, current_user: dict = Depends(get_current_user)):
    db = get_db()
    result = await db["farms"].delete_one({"_id": ObjectId(farm_id), "user_id": str(current_user["_id"])})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Farm not found")
    return {"message": "Farm deleted"}
