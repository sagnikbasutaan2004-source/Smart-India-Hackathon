from fastapi import APIRouter, Depends
from backend.utils.auth import get_current_user
from backend.services.pipeline_adapter import get_weather_summary
import json
from pathlib import Path
from backend.config import settings

router = APIRouter(prefix="/api/weather", tags=["weather"])

BASE = Path(settings.PIPELINE_BASE_DIR)


@router.get("")
async def get_weather(current_user: dict = Depends(get_current_user)):
    summary = get_weather_summary()
    # Also return hourly data if available
    hourly_path = BASE / "metadata/weather_hourly.csv"
    daily_path = BASE / "metadata/weather_daily_summary.csv"

    import csv
    hourly = []
    if hourly_path.exists():
        with open(hourly_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                hourly.append(row)

    daily = []
    if daily_path.exists():
        with open(daily_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                daily.append(row)

    return {
        "summary": summary,
        "hourly": hourly[:48],   # last 48 hours
        "daily": daily,
    }
