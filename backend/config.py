from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    MONGODB_URI: str = ""
    MONGODB_DATABASE: str = "targetedcrop"
    JWT_SECRET_KEY: str = "change-this-secret"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 10080
    GROQ_API_KEY: str = ""
    PIPELINE_BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()
