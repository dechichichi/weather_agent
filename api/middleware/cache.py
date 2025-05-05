from aiocache import Cache, cached
from aiocache.serializers import JsonSerializer
import config.api_config as api_config  # 导入配置

# 初始化缓存（支持Redis/Memory）
cache = Cache(
    Cache.REDIS if api_config.load_config().USE_REDIS else Cache.MEMORY,
    endpoint=api_config.load_config().REDIS_HOST,
    port=api_config.load_config().REDIS_PORT,
    serializer=JsonSerializer()
)

def cache_decorator(ttl: int = api_config.load_config().CACHE_TTL):
    """带TTL的异步缓存装饰器"""
    return cached(
        ttl=ttl,
        cache=cache,
        key_builder=lambda f, *args, **kwargs: f"cache:{f.__name__}:{args}:{kwargs}"
    )
    