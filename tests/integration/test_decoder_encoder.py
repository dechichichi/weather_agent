## 运行全部测试
#pytest tests/ -v
import torch
from model.encoder.encoder import Encoder
from model.decoder.decoder import Decoder
from model.config import TransformerConfig

def test_decoder_with_encoder():
    config = TransformerConfig(d_model=64, nhead=4)
    
    # 初始化模块
    encoder = Encoder(config)
    decoder = Decoder(config)
    
    # 模拟Encoder输入 (batch_size=2, seq_len=10)
    src = torch.randn(2, 10, config.d_model)
    memory = encoder(src)
    
    # 模拟Decoder输入 (seq_len=5)
    tgt = torch.randn(2, 5, config.d_model)
    
    # 正向传播
    output = decoder(tgt, memory)
    
    # 验证输出形状 (应保持目标序列长度)
    assert output.shape == (2, 5, config.d_model), \
        f"Expected shape (2,5,{config.d_model}), got {output.shape}"