"""Application configuration.

All settings are centralized here so that nothing is hardcoded across the
codebase. Paths are derived dynamically from this file's location, which keeps
the project portable across machines and operating systems.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed application settings.

    Values can be overridden via environment variables or a ``.env`` file
    placed in the ``backend/`` directory.
    """

    # --- General -----------------------------------------------------------
    APP_NAME: str = "AutoPrep AI"
    APP_VERSION: str = "0.1.0"
    API_V1_PREFIX: str = "/api/v1"

    # --- Filesystem paths --------------------------------------------------
    # config.py -> core -> app -> backend  (parents[2] == backend/)
    BACKEND_DIR: Path = Path(__file__).resolve().parents[2]

    # --- Uploads -----------------------------------------------------------
    # Maximum accepted upload size (bytes). Default: 50 MB.
    MAX_UPLOAD_SIZE: int = 50 * 1024 * 1024
    ALLOWED_EXTENSIONS: tuple[str, ...] = (".csv",)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def upload_dir(self) -> Path:
        """Absolute path to the directory where uploaded files are stored."""
        return self.BACKEND_DIR / "uploads"

    @property
    def processed_dir(self) -> Path:
        """Absolute path where cleaned dataset copies are stored."""
        return self.BACKEND_DIR / "processed"

    @property
    def database_url(self) -> str:
        """SQLAlchemy connection string for the local SQLite database."""
        db_path = self.BACKEND_DIR / "autoprep.db"
        return f"sqlite:///{db_path.as_posix()}"


# A single shared settings instance, imported wherever configuration is needed.
settings = Settings()

# Ensure storage directories always exist at startup.
settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.processed_dir.mkdir(parents=True, exist_ok=True)
