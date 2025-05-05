from fastapi import APIRouter
import importlib.metadata
from data.database.queries import get_db  # 假设检查数据库连接通过获取数据库会话来实现
import config.api_config as api_config  # 导入配置

router = APIRouter()

# 假设模型加载状态标记
model_loaded = True  

def check_db_connection():
    try:
        next(get_db())
        return "OK"
    except Exception:
        return "Error"

@router.get("/health")
async def health_check():
    return {
        "status": "OK",
        "version": importlib.metadata.version("your-package"),  # 动态获取版本
        "dependencies": {
            "database": check_db_connection(),
            "model": model_loaded  # 模型状态标记
        }
    }