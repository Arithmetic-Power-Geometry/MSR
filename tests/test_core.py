import torch
from src.msr.experiment import MLP, flat_params, set_from_flat, projection_repair

def test_flat_roundtrip():
    m = MLP(3, 2, 4)
    v = flat_params(m).clone()
    set_from_flat(m, v)
    assert torch.allclose(flat_params(m), v)

def test_projection_satisfies_simple_target():
    torch.manual_seed(1)
    m = MLP(2,2,4)
    x = torch.tensor([[1.0,-0.5]])
    with torch.no_grad():
        target = 1 - int(m(x).argmax(1)[0])
    yt = torch.tensor([target])
    metric = torch.ones_like(flat_params(m))
    m, _, _ = projection_repair(m, x, yt, metric, margin=.2, rounds=12)
    assert int(m(x).argmax(1)[0]) == target
