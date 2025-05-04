# api/routes/forecast.py
from fastapi import APIRouter
from model.weather_transformer import predict_weather  # 假设的模型预测函数

# 创建路由实例
router = APIRouter()

# 定义天气预报接口
@router.get("/forecast")
async def get_forecast(location: str):
    """
    根据地点获取天气预报
    :param location: 要查询的地点
    :return: 包含天气预报结果和置信度的 JSON 对象
    """
    try:
        # 调用模型进行预测
        prediction, confidence = predict_weather(location)
        return {"prediction": prediction, "confidence": confidence}
    except Exception as e:
        # 处理异常情况
        return {"error": f"预测失败: {str(e)}"}