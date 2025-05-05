import logging
from fastapi import APIRouter, HTTPException, status
from pydantic import StringConstraints
from typing import Annotated  # 导入 Annotated
from model.transformer import predict as predict_weather  # 假设模型预测函数

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/forecast")
async def get_forecast(
    location: Annotated[str, StringConstraints(min_length=1, max_length=50)]
):
    try:
        # 异步调用模型
        prediction, confidence = await predict_weather(location)
        return {"prediction": prediction, "confidence": confidence}
    except ValueError as e:  # 业务异常
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # 其他异常
        logger.error(f"Forecast error: {e}")  # 记录日志而非返回详情
        raise HTTPException(status_code=500, detail="预测服务暂不可用")
    