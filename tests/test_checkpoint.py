"""Unit tests for src/utils/checkpoint.py."""

import torch
import torch.nn as nn

from src.utils.checkpoint import load_checkpoint, save_checkpoint


def _make_model():
    """Create a small linear model for testing."""
    return nn.Linear(4, 2)


def _make_optimizer(model):
    """Create an SGD optimizer for the given model."""
    return torch.optim.SGD(model.parameters(), lr=0.01)


def _sample_config():
    return {"seed": 42, "lr": 0.001, "experiment": "test"}


def _sample_metrics():
    return {"macro_auroc": 0.85, "per_class_auroc": {"A": 0.9, "B": 0.8}}


def test_save_load_roundtrip_model_state(tmp_path):
    """Model state dict is preserved through save/load round-trip."""
    model = _make_model()
    optimizer = _make_optimizer(model)
    path = str(tmp_path / "ckpt.pt")

    save_checkpoint(path, model, optimizer, epoch=5, config=_sample_config(), val_metrics=_sample_metrics())

    # Create a fresh model with different weights
    model2 = _make_model()
    # Weights should differ before loading
    assert not _state_dicts_equal(model.state_dict(), model2.state_dict())

    load_checkpoint(path, model2, device="cpu")

    # After loading, weights should match
    assert _state_dicts_equal(model.state_dict(), model2.state_dict())


def test_save_load_roundtrip_optimizer_state(tmp_path):
    """Optimizer state dict is preserved through save/load round-trip."""
    model = _make_model()
    optimizer = _make_optimizer(model)

    # Run a training step so optimizer has non-trivial state
    x = torch.randn(3, 4)
    loss = model(x).sum()
    loss.backward()
    optimizer.step()

    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(path, model, optimizer, epoch=1, config=_sample_config(), val_metrics=_sample_metrics())

    model2 = _make_model()
    optimizer2 = _make_optimizer(model2)
    load_checkpoint(path, model2, optimizer=optimizer2, device="cpu")

    # Compare optimizer state dicts
    orig = optimizer.state_dict()
    loaded = optimizer2.state_dict()
    assert orig["param_groups"] == loaded["param_groups"]
    for key in orig["state"]:
        for k, v in orig["state"][key].items():
            if isinstance(v, torch.Tensor):
                assert torch.equal(v, loaded["state"][key][k])
            else:
                assert v == loaded["state"][key][k]


def test_metadata_preserved(tmp_path):
    """Epoch and val_metrics are preserved in the returned metadata."""
    model = _make_model()
    optimizer = _make_optimizer(model)
    config = _sample_config()
    metrics = _sample_metrics()
    path = str(tmp_path / "ckpt.pt")

    save_checkpoint(path, model, optimizer, epoch=10, config=config, val_metrics=metrics)

    model2 = _make_model()
    meta = load_checkpoint(path, model2, device="cpu")

    assert meta["epoch"] == 10
    assert meta["val_metrics"] == metrics
    assert meta["config"] == config
    assert meta["seed"] == 42


def test_load_without_optimizer(tmp_path):
    """load_checkpoint works when optimizer is None."""
    model = _make_model()
    optimizer = _make_optimizer(model)
    path = str(tmp_path / "ckpt.pt")

    save_checkpoint(path, model, optimizer, epoch=3, config=_sample_config(), val_metrics=_sample_metrics())

    model2 = _make_model()
    meta = load_checkpoint(path, model2, optimizer=None, device="cpu")

    assert _state_dicts_equal(model.state_dict(), model2.state_dict())
    assert meta["epoch"] == 3


def _state_dicts_equal(sd1, sd2):
    """Check if two state dicts have identical keys and tensor values."""
    if sd1.keys() != sd2.keys():
        return False
    return all(torch.equal(sd1[k], sd2[k]) for k in sd1)
