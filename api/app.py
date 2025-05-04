# api/app.py
from fastapi import FastAPI
from config import load_config
from api.routes.forecast import router as forecast_router

# 创建 FastAPI 应用实例
app = FastAPI(title="Weather Agent API")
# 加载配置
config = load_config()

# 包含天气预报路由
app.include_router(forecast_router, prefix="/api")

# 定义健康检查接口
@app.get("/health")
async def health_check():
    """
    健康检查接口，返回服务的状态和版本信息
    """
    return {"status": "OK", "version": "0.1.0"}