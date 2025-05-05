from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import config
from model.encoder.encoder import Encoder
from model.decoder.decoder import Decoder
from model.core.embeddings.token_embedding import Embedding
import torch
import torch.nn as nn


class TransformerLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x_input, x_output = x
        encoder_output = self.encoder(x_input)
        decoder_output = self.decoder(x_output, encoder_output)
        return encoder_output, decoder_output


class Transformer(nn.Module):
    def __init__(self, N, vocab_size, output_dim):
        super().__init__()
        self.embedding_input = Embedding(vocab_size=vocab_size)
        self.embedding_output = Embedding(vocab_size=vocab_size)
        self.output_dim = output_dim
        self.linear = nn.Linear(config.d_model, output_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.layers = nn.ModuleList([TransformerLayer() for _ in range(N)])

    def forward(self, x):
        try:
            x_input, x_output = x
            x_input = self.embedding_input(x_input)
            x_output = self.embedding_output(x_output)

            encoder_output = None
            for layer in self.layers:
                encoder_output, decoder_output = layer((x_input, x_output))
                x_input = encoder_output
                x_output = decoder_output

            output = self.linear(decoder_output)
            output = self.softmax(output)

            return output
        except Exception as e:
            print(f"An error occurred during the forward pass: {e}")
            return None


def predict(model, input_data, output_data):
    """
    使用训练好的模型进行预测。
    :param model: 训练好的 Transformer 模型
    :param input_data: 输入数据
    :param output_data: 输出数据
    :return: 预测结果
    """
    # 设置模型为评估模式
    model.eval()
    # 关闭梯度计算
    with torch.no_grad():
        # 将输入数据转换为张量
        input_tensor = torch.tensor(input_data, dtype=torch.long)
        output_tensor = torch.tensor(output_data, dtype=torch.long)
        # 进行预测
        output = model((input_tensor, output_tensor))
        # 获取预测结果的索引
        predicted_indices = torch.argmax(output, dim=-1)
        return predicted_indices