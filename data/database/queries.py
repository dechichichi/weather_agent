# data/database/queries.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .models import Base, WeatherRecord

# 假设数据库连接字符串，可根据实际情况修改
DATABASE_URL = "sqlite:///./weather.db"
engine = create_engine(DATABASE_URL)
Base.metadata.create_all(bind=engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_weather_record(timestamp, temperature, humidity, location):
    """
    创建一条新的天气记录
    :param timestamp: 记录时间
    :param temperature: 温度
    :param humidity: 湿度
    :param location: 地点
    :return: 创建的天气记录
    """
    db = next(get_db())
    record = WeatherRecord(timestamp=timestamp, temperature=temperature, humidity=humidity, location=location)
    db.add(record)
    db.commit()
    db.refresh(record)
    return record

def get_weather_record_by_id(record_id):
    """
    根据记录 ID 获取天气记录
    :param record_id: 记录 ID
    :return: 对应的天气记录
    """
    db = next(get_db())
    return db.query(WeatherRecord).filter(WeatherRecord.id == record_id).first()

def get_all_weather_records():
    """
    获取所有的天气记录
    :return: 所有天气记录的列表
    """
    db = next(get_db())
    return db.query(WeatherRecord).all()

def get_weather_records_by_location(location):
    """
    根据地点获取天气记录
    :param location: 地点
    :return: 对应地点的天气记录列表
    """
    db = next(get_db())
    return db.query(WeatherRecord).filter(WeatherRecord.location == location).all()

def delete_weather_record(record_id):
    """
    根据记录 ID 删除天气记录
    :param record_id: 记录 ID
    :return: 删除成功返回 True，否则返回 False
    """
    db = next(get_db())
    record = db.query(WeatherRecord).filter(WeatherRecord.id == record_id).first()
    if record:
        db.delete(record)
        db.commit()
        return True
    return False