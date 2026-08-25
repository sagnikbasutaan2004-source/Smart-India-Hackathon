import sys
import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Add project root to path so existing pipeline is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.database.mongodb import connect_to_mongo, close_mongo_connection
from backend.routes import auth, users, farms, crops, zones, scans, dashboard, weather

app = FastAPI(
    title="TargetedCrop API",
    description="Full-stack agricultural intelligence platform backed by the AgriDrone Digital Twin pipeline.",
    version="1.0.0",
)

# ─── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Startup / Shutdown ───────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    await connect_to_mongo()

@app.on_event("shutdown")
async def shutdown():
    await close_mongo_connection()

# ─── Static files: serve pipeline overlays directly ──────────────────────────
BASE = Path(__file__).parent.parent
pipeline_dir = BASE
app.mount("/pipeline", StaticFiles(directory=str(pipeline_dir)), name="pipeline")

# ─── Routers ──────────────────────────────────────────────────────────────────
app.include_router(auth.router)
app.include_router(users.router)
app.include_router(farms.router)
app.include_router(crops.router)
app.include_router(zones.router)
app.include_router(scans.router)
app.include_router(dashboard.router)
app.include_router(weather.router)


@app.get("/")
async def root():
    return {
        "name": "TargetedCrop API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    return {"status": "ok"}
