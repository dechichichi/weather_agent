## 仅运行单元测试
#pytest tests/unit/ -v
import torch
import pytest
from model.encoder.encoder import Encoder
from model.config import TransformerConfig

@pytest.fixture
def sample_config():
    return TransformerConfig(
        d_model=64, 
        nhead=4, 
        dim_feedforward=256
    )

def test_encoder_forward_shape(sample_config):
    # 初始化Encoder
    encoder = Encoder(sample_config)
    
    # 模拟输入 (batch_size=2, seq_len=10)
    x = torch.randn(2, 10, sample_config.d_model)
    
    # 正向传播
    output = encoder(x)
    
    # 断言输出形状与输入一致
    assert output.shape == x.shape, \
        f"Expected shape {x.shape}, got {output.shape}"