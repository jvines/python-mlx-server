from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    host: str = "127.0.0.1"
    port: int = 1234
    log_level: str = "INFO"

    # Registry and model storage
    model_registry_path: Path = Path.home() / ".config" / "mlx-server" / "registry.json"
    model_dir: Path = Path.home() / ".cache" / "mlx-server" / "models"

    # Idle model eviction — 0 disables TTL
    model_ttl_seconds: int = 1800

    model_config = SettingsConfigDict(env_prefix="MLX_")


settings = Settings()
