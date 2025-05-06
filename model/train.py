import torch
import torch.nn as nn
import torch.optim as optim
from model.transformer import Transformer
from config.load_config import load_model_config
from data.database.queries import get_all_weather_records
from model.core.embeddings.token_embedding import Embedding
import config

# 加载模型配置
model_config = load_model_config()['transformer']
d_model = model_config['d_model']
nhead = model_config['nhead']
num_layers = model_config['num_layers']
dropout = model_config['dropout']

# 假设的词汇表大小和输出维度
vocab_size = config.Config().vocab_size
output_dim = 3

# 初始化模型
model = Transformer(N=num_layers, vocab_size=vocab_size, output_dim=output_dim)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 数据处理函数
def process_data(records):
    input_texts = []
    output_labels = []
    for record in records:
        # 这里可以根据实际情况处理每条记录，将其转换为输入文本和输出标签
        # 示例：将地点作为输入文本，简单将温度情况分类作为输出标签
        input_text = record.location
        if record.temperature > 25:
            output_label = 0  # 高温
        elif record.temperature < 15:
            output_label = 1  # 低温
        else:
            output_label = 2  # 常温
        input_texts.append([ord(c) for c in input_text])
        output_labels.append(output_label)
    
    embedding = Embedding(vocab_size=vocab_size)
    input_embeddings = embedding(input_texts)
    output_tensor = torch.tensor(output_labels, dtype=torch.long)
    return input_embeddings, output_tensor

# 训练函数
def train_model(model, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        all_records = get_all_weather_records()
        input_data, output_data = process_data(all_records)

        # 前向传播
        output = model((input_data, output_data.unsqueeze(1).expand(-1, input_data.size(1))))
        if output is None:
            continue

        # 计算损失
        loss = criterion(output.view(-1, output_dim), output_data.view(-1))

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

# 训练模型
num_epochs = 10
train_model(model, criterion, optimizer, num_epochs)