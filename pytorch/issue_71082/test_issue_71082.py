import pytest
import torch
import torch.nn.functional as F

def test_f():
    # 71082
    input_tensor = torch.randint(-1, 1, [3], dtype=torch.int64)
    r = 100
    
    with pytest.raises(RuntimeError) as excinfo:
        torch.combinations(input_tensor, r=r)
    
    assert "Expected r to be smaller than" in str(excinfo.value) or "size is invalid" in str(excinfo.value)