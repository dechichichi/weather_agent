# data/database/models.py
from sqlalchemy import Column, Integer, String, DateTime, Float 
from sqlalchemy.ext.declarative import declarative_base

# 创建一个基类
#Base是一个基类，所有的表映射类都需要继承自这个基类。这个基类包含了 SQLAlchemy 用于将类映射到数据库表的所有功能。
Base = declarative_base()

#定义表映射类
# 这个类定义了一个名为 weather_records 的表，包含 id、timestamp、temperature、humidity 和 location 字段。
class WeatherRecord(Base):
    # 指定表名
    __tablename__ = "weather_records"
    
    id = Column(Integer, primary_key=True) #：用于存储记录的唯一标识符，数据类型为整数，是表的主键。
    timestamp = Column(DateTime) # 用于存储记录的时间戳，数据类型为日期时间。
    temperature = Column(Float) # 用于存储记录的温度值，数据类型为浮点数
    humidity = Column(Float) # 用于存储记录的湿度值，数据类型为浮点数
    location = Column(String(50)) #于存储记录的位置信息，数据类型为长度不超过50的字符串。