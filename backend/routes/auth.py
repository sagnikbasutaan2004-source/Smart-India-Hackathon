from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, status
from bson import ObjectId
from backend.database.mongodb import get_db
from backend.schemas.schemas import RegisterRequest, LoginRequest, TokenResponse
from backend.utils.auth import hash_password, verify_password, create_access_token

router = APIRouter(prefix="/api/auth", tags=["auth"])


def _serialize_user(u: dict) -> dict:
    u["id"] = str(u.pop("_id"))
    u.pop("hashed_password", None)
    return u


@router.post("/register", status_code=201)
async def register(body: RegisterRequest):
    db = get_db()
    if db is None:
        raise HTTPException(status_code=503, detail="Database not connected")
    existing = await db["users"].find_one({"email": body.email.lower()})
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")
    doc = {
        "name": body.name,
        "email": body.email.lower(),
        "phone": body.phone,
        "hashed_password": hash_password(body.password),
        "location": {"state": None, "district": None, "village": None},
        "profile": {"land_size": None, "land_unit": "acres", "primary_crop": "paddy"},
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }
    result = await db["users"].insert_one(doc)
    token = create_access_token({"sub": str(result.inserted_id)})
    return {"access_token": token, "token_type": "bearer", "user_id": str(result.inserted_id)}


@router.post("/login")
async def login(body: LoginRequest):
    db = get_db()
    if db is None:
        raise HTTPException(status_code=503, detail="Database not connected")
    user = await db["users"].find_one({"email": body.email.lower()})
    if not user or not verify_password(body.password, user.get("hashed_password", "")):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_access_token({"sub": str(user["_id"])})
    return {"access_token": token, "token_type": "bearer", "user_id": str(user["_id"])}


@router.get("/me")
async def me(current_user: dict = __import__("fastapi").Depends(__import__("backend.utils.auth", fromlist=["get_current_user"]).get_current_user)):
    current_user["id"] = str(current_user.pop("_id"))
    current_user.pop("hashed_password", None)
    return current_user
