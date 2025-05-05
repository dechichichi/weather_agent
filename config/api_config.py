import os
from pydantic import BaseSettings

class Settings(BaseSettings):
    # 模型配置
    d_model: int = 512
    nhead: int = 8
    max_seq_len: int = 512

    # 缓存配置
    USE_REDIS: bool = os.getenv("USE_REDIS", "False").lower() == "true"
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))
    CACHE_TTL: int = 300  # 5分钟

    class Config:
        env_file = ".env"

def load_config():
    return Settings()