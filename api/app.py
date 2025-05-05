from fastapi import FastAPI, Depends
from aiocache import cached  # 异步缓存库
from api.routers import health, forecast  # 导入路由模块
from pydantic import constr

app = FastAPI()

# 模拟 get_current_user 函数
def get_current_user():
    # 这里可以添加实际的用户验证逻辑
    return {"username": "test_user"}

# 包含路由
app.include_router(health.router)
app.include_router(forecast.router)

# 使用支持异步的缓存装饰器
@app.get("/protected")
@cached(ttl=60)  # 使用 aiocache
async def protected_route(user: dict = Depends(get_current_user)):
    return {"message": "This is a protected route", "user": user}
