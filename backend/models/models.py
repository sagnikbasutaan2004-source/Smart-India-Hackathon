from datetime import datetime, timezone
from typing import Optional, List
from pydantic import BaseModel, Field, EmailStr


class PyObjectId(str):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v, handler=None):
        return str(v)


# ─── User ────────────────────────────────────────────────────────────────────

class UserLocation(BaseModel):
    state: Optional[str] = None
    district: Optional[str] = None
    village: Optional[str] = None


class UserProfile(BaseModel):
    land_size: Optional[float] = None
    land_unit: Optional[str] = "acres"
    primary_crop: Optional[str] = "paddy"


class UserInDB(BaseModel):
    name: str
    email: str
    phone: Optional[str] = None
    hashed_password: str
    location: UserLocation = UserLocation()
    profile: UserProfile = UserProfile()
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ─── Farm ────────────────────────────────────────────────────────────────────

class FarmLocation(BaseModel):
    state: Optional[str] = None
    district: Optional[str] = None
    village: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class FarmInDB(BaseModel):
    user_id: str
    farm_name: str
    location: FarmLocation = FarmLocation()
    total_land_size: Optional[float] = None
    land_unit: Optional[str] = "acres"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ─── Crop ────────────────────────────────────────────────────────────────────

class CropInDB(BaseModel):
    farm_id: str
    user_id: str
    crop_name: str = "paddy"
    variety: Optional[str] = None
    season: Optional[str] = None
    sowing_date: Optional[datetime] = None
    expected_harvest_date: Optional[datetime] = None
    area: Optional[float] = None
    area_unit: Optional[str] = "acres"
    status: str = "active"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ─── Field Zone ──────────────────────────────────────────────────────────────

class FieldZoneInDB(BaseModel):
    farm_id: Optional[str] = None
    crop_id: Optional[str] = None
    user_id: Optional[str] = None
    zone_name: str
    zone_number: int
    zone_id: str  # e.g. "Z01"
    area: Optional[float] = None
    area_unit: Optional[str] = "acres"
    health_status: Optional[str] = "unknown"
    risk_level: Optional[str] = "unknown"
    gi_star: Optional[float] = None
    hotspot_class: Optional[str] = None
    spread_risk: Optional[float] = None
    composite_health_score: Optional[float] = None
    severity_label: Optional[str] = None
    pathogen_severity: Optional[str] = None
    detection_count: Optional[int] = 0
    zone_coverage_pct: Optional[float] = None
    gps_centroid: Optional[dict] = None
    geometry: Optional[dict] = None
    source_files: Optional[dict] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ─── Disease Scan ─────────────────────────────────────────────────────────────

class DiseaseScanInDB(BaseModel):
    user_id: str
    farm_id: Optional[str] = None
    crop_id: Optional[str] = None
    zone_id: Optional[str] = None

    image: Optional[dict] = {}  # {original_url, processed_url, annotated_url}

    classification: Optional[dict] = {}   # {crop, disease, confidence}
    localization: Optional[dict] = {}     # {model, regions}
    segmentation: Optional[dict] = {}     # {enabled, model, mask_url}

    severity: Optional[dict] = {}         # {affected_area_percentage, level}
    recommendation_id: Optional[str] = None

    pipeline_metadata: Optional[dict] = {}
    field_summary: Optional[dict] = {}    # full health/pathogen summary
    epidemiology: Optional[dict] = {}     # hotspot/coldspot summary
    scenario_analysis: Optional[dict] = {}

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ─── Recommendation ──────────────────────────────────────────────────────────

class RecommendationInDB(BaseModel):
    scan_id: Optional[str] = None
    user_id: str
    severity: Optional[str] = None
    affected_area_percentage: Optional[float] = None
    action: Optional[str] = None  # monitor / inspect / targeted_intervention / urgent_attention
    message: Optional[str] = None
    zones_affected: Optional[List[str]] = []
    pathogens_detected: Optional[List[str]] = []
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ─── Weather Record ──────────────────────────────────────────────────────────

class WeatherRecordInDB(BaseModel):
    farm_id: Optional[str] = None
    user_id: Optional[str] = None
    capture_date: Optional[str] = None
    temperature_min: Optional[float] = None
    temperature_max: Optional[float] = None
    humidity_mean: Optional[float] = None
    rainfall_mm: Optional[float] = None
    seven_day_rain_mm: Optional[float] = None
    high_humidity_risk: Optional[bool] = None
    rust_favorable: Optional[bool] = None
    spot_favorable: Optional[bool] = None
    chlorosis_risk: Optional[bool] = None
    raw_report: Optional[dict] = {}
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
