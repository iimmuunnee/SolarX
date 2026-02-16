"""Backend configuration settings."""
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """Application settings."""

    # API Settings
    api_title: str = "SolarX API"
    api_version: str = "1.0.0"
    api_description: str = "Battery optimization API for solar-powered robot charging stations"

    # CORS Settings
    cors_origins: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8000",
    ]

    # Cache Settings
    cache_maxsize: int = 100
    cache_ttl_seconds: int = 3600  # 1 hour

    # SolarX Paths (relative to backend directory)
    solarx_base_dir: str = "../SolarX"
    solarx_data_dir: str = "../SolarX/data"
    solarx_model_path: str = "../SolarX/src/lstm_solar_model.pth"

    # Pre-computed Results
    precomputed_dir: str = "precomputed"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
