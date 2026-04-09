"""Application configuration via environment variables."""

from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    rate_limit_per_minute: int = 60
    default_target_score: float = 0.90
    max_iterations: int = 50
    simulation_timeout: int = 120
    pybullet_timestep: float = 1.0 / 240.0
    output_fps: int = 30

    model_config = ConfigDict(env_prefix="OYSTERWORLD_")
