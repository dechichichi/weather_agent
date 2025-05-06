![deepseek_mermaid_20250503_3e65dd](C:\Users\Lyan\Downloads\deepseek_mermaid_20250503_3e65dd.png)

$$

$$

```
weather_agent/
├── .gitignore  # 忽略特定文件
├── README.md  # 展示项目图片
├── requirements.txt  # 列出依赖库
├── config/
│   ├── api.yaml  # API服务配置
│   ├── api_config.py  # 加载API配置
│   ├── load_config.py  # 加载YAML配置
│   └── model_params.yaml  # 模型参数配置
├── data/
│   ├── database/
│   │   ├── models.py  # 定义数据库表
│   │   └── queries.py  # 数据库操作
│   └── processed/
│       └── processed.py  # 数据预处理
├── features/
│   ├── normalization.py  # 数据标准化
│   └── extraction.py  # 特征提取
├── api/
│   ├── middleware/
│   │   └── cache.py  # 缓存配置
│   ├── routers/
│   │   ├── forecast.py  # 天气预测接口
│   │   └── health.py  # 健康检查接口
│   └── app.py  # FastAPI应用
├── model/
│   ├── core/
│   │   ├── attention/
│   │   │   └── multi_head.py  # 多头注意力机制
│   │   ├── embeddings/
│   │   │   ├── positional_encoding.py  # 位置编码
│   │   │   └── token_embedding.py  # 词嵌入
│   │   ├── norms/
│   │   │   └── add_norm.py  # 残差归一化
│   │   └── feed_forward.py  # 前馈网络
│   ├── encoder/
│   │   └── encoder.py  # 编码器
│   ├── decoder/
│   │   └── decoder.py  # 解码器
│   ├── config.py  # 模型配置
│   ├── train.py  # 模型训练
│   └── transformer.py  # Transformer模型
```

