import torch

from src.utils.repro import set_seed, get_device


def test_set_seed_runs():
    set_seed(42)
    x = torch.rand(3)
    assert x.shape == (3,)


def test_set_seed_reproducible():
    set_seed(42)
    x1 = torch.rand(3)

    set_seed(42)
    x2 = torch.rand(3)

    assert torch.equal(x1, x2)


def test_get_device_returns_valid_device():
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in {"cuda", "mps", "cpu"}

