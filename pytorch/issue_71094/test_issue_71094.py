import pytest
import torch
import torch.nn.functional as F

def test_f():
    # 71094
    input_tensor = torch.randint(-128, -1, [3, 6], dtype=torch.int64)
    weight = torch.rand([30522, 384], dtype=torch.float32)
    padding_idx = 0
    
    with pytest.raises(RuntimeError) as excinfo:
        F.embedding_bag(
            input_tensor,
            weight,
            padding_idx=padding_idx,
            max_norm=None,
            norm_type=2.0,
            scale_grad_by_freq=False,
            sparse=False,
        )

    assert "index out of range" in str(excinfo.value) or "Index must be non-negative" in str(excinfo.value)