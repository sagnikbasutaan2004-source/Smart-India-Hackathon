from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Depends
from bson import ObjectId
from backend.database.mongodb import get_db
from backend.schemas.schemas import UpdateUserRequest
from backend.utils.auth import get_current_user

router = APIRouter(prefix="/api/users", tags=["users"])


@router.get("/me")
async def get_profile(current_user: dict = Depends(get_current_user)):
    current_user["id"] = str(current_user.pop("_id"))
    current_user.pop("hashed_password", None)
    return current_user


@router.put("/me")
async def update_profile(body: UpdateUserRequest, current_user: dict = Depends(get_current_user)):
    db = get_db()
    update_data = {"updated_at": datetime.now(timezone.utc)}
    if body.name is not None:
        update_data["name"] = body.name
    if body.phone is not None:
        update_data["phone"] = body.phone
    if body.location is not None:
        update_data["location"] = body.location.model_dump()
    if body.profile is not None:
        update_data["profile"] = body.profile.model_dump()

    await db["users"].update_one(
        {"_id": ObjectId(str(current_user["_id"]))},
        {"$set": update_data},
    )
    updated = await db["users"].find_one({"_id": ObjectId(str(current_user["_id"]))})
    updated["id"] = str(updated.pop("_id"))
    updated.pop("hashed_password", None)
    return updated
