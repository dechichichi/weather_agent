import torch
from model.core.attention.multi_head import MultiHeadAttention
from model.config import TransformerConfig

def test_causal_mask_effect():
    config = TransformerConfig(d_model=64, nhead=4)
    attn = MultiHeadAttention(config.d_model, config.nhead)
    
    # 生成测试输入 (seq_len=3)
    query = key = value = torch.randn(1, 3, config.d_model)
    
    # 不带掩码的输出
    output_no_mask = attn(query, key, value)
    
    # 带因果掩码的输出
    output_with_mask = attn(query, key, value, mask='causal')
    
    # 验证最后一个token的输出差异（应受掩码影响）
    assert not torch.allclose(output_no_mask[:, -1, :], output_with_mask[:, -1, :]), \
        "Causal mask未生效，未来信息未被正确屏蔽"