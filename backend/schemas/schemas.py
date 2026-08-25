from datetime import datetime
from typing import Optional, Any
from pydantic import BaseModel, EmailStr


# ─── Auth ─────────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str
    phone: Optional[str] = None


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


# ─── User ─────────────────────────────────────────────────────────────────────

class UserLocation(BaseModel):
    state: Optional[str] = None
    district: Optional[str] = None
    village: Optional[str] = None


class UserProfile(BaseModel):
    land_size: Optional[float] = None
    land_unit: Optional[str] = "acres"
    primary_crop: Optional[str] = "paddy"


class UserResponse(BaseModel):
    id: str
    name: str
    email: str
    phone: Optional[str] = None
    location: UserLocation = UserLocation()
    profile: UserProfile = UserProfile()
    created_at: Optional[datetime] = None


class UpdateUserRequest(BaseModel):
    name: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[UserLocation] = None
    profile: Optional[UserProfile] = None


# ─── Farm ─────────────────────────────────────────────────────────────────────

class FarmLocation(BaseModel):
    state: Optional[str] = None
    district: Optional[str] = None
    village: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class CreateFarmRequest(BaseModel):
    farm_name: str
    location: Optional[FarmLocation] = FarmLocation()
    total_land_size: Optional[float] = None
    land_unit: Optional[str] = "acres"


class FarmResponse(BaseModel):
    id: str
    user_id: str
    farm_name: str
    location: FarmLocation
    total_land_size: Optional[float] = None
    land_unit: Optional[str] = None
    created_at: Optional[datetime] = None


# ─── Crop ─────────────────────────────────────────────────────────────────────

class CreateCropRequest(BaseModel):
    farm_id: str
    crop_name: str = "paddy"
    variety: Optional[str] = None
    season: Optional[str] = None
    sowing_date: Optional[datetime] = None
    expected_harvest_date: Optional[datetime] = None
    area: Optional[float] = None
    area_unit: Optional[str] = "acres"


class CropResponse(BaseModel):
    id: str
    farm_id: str
    user_id: str
    crop_name: str
    variety: Optional[str] = None
    season: Optional[str] = None
    area: Optional[float] = None
    area_unit: Optional[str] = None
    status: str
    created_at: Optional[datetime] = None


# ─── Zone ─────────────────────────────────────────────────────────────────────

class ZoneResponse(BaseModel):
    id: str
    zone_id: str
    zone_name: str
    zone_number: int
    health_status: Optional[str] = None
    risk_level: Optional[str] = None
    gi_star: Optional[float] = None
    hotspot_class: Optional[str] = None
    spread_risk: Optional[float] = None
    composite_health_score: Optional[float] = None
    severity_label: Optional[str] = None
    pathogen_severity: Optional[str] = None
    detection_count: Optional[int] = None
    zone_coverage_pct: Optional[float] = None
    gps_centroid: Optional[dict] = None
    source_files: Optional[dict] = None


# ─── Scan ─────────────────────────────────────────────────────────────────────

class ScanAnalyzeRequest(BaseModel):
    farm_id: Optional[str] = None
    crop_id: Optional[str] = None


class ScanResponse(BaseModel):
    id: str
    user_id: str
    farm_id: Optional[str] = None
    crop_id: Optional[str] = None
    classification: Optional[dict] = None
    severity: Optional[dict] = None
    field_summary: Optional[dict] = None
    epidemiology: Optional[dict] = None
    scenario_analysis: Optional[dict] = None
    pipeline_metadata: Optional[dict] = None
    created_at: Optional[datetime] = None


# ─── Dashboard ────────────────────────────────────────────────────────────────

class DashboardSummary(BaseModel):
    total_farms: int = 0
    total_crops: int = 0
    total_scans: int = 0
    field_health: Optional[dict] = None
    pathogen_summary: Optional[dict] = None
    epidemiology_summary: Optional[dict] = None
    weather_summary: Optional[dict] = None
    recent_alerts: Optional[list] = []
