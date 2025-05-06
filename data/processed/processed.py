import torch
from data.database.queries import get_all_weather_records
from model.config import Config
from model.core.embeddings.token_embedding import Embedding

# 加载配置
config = Config()

def load_and_preprocess_data():
    # 从数据库获取所有天气记录
    all_records = get_all_weather_records()

    # 假设我们将每条记录的文本信息作为输入，这里简单模拟文本信息
    input_texts = []
    output_labels = []

    for record in all_records:
        # 这里需要根据实际情况将记录转换为文本和标签
        # 假设记录有一个 text 属性作为输入文本，一个 label 属性作为输出标签
        input_text = str(record)  # 简单示例，实际需要根据记录结构修改
        output_label = 0  # 简单示例，实际需要根据记录结构修改
        input_texts.append(input_text)
        output_labels.append(output_label)

    # 对输入文本进行分词和转换为索引
    tokenized_inputs = []
    for text in input_texts:
        # 这里简单假设将文本拆分为字符作为分词
        tokens = [ord(c) for c in text]
        tokenized_inputs.append(tokens)

    # 使用 TokenEmbedding 进行嵌入
    embedding = Embedding(vocab_size=config.vocab_size)
    input_embeddings = embedding(tokenized_inputs)

    # 将输出标签转换为张量
    output_tensor = torch.tensor(output_labels, dtype=torch.long)

    return input_embeddings, output_tensor