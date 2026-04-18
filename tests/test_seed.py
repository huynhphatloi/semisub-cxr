"""Unit tests for src/utils/seed.py."""

import random

import numpy as np
import torch

from src.utils.seed import set_all_seeds


def test_set_all_seeds_deterministic_random():
    """Python random produces identical sequences after seeding twice."""
    set_all_seeds(123)
    a = [random.random() for _ in range(5)]
    set_all_seeds(123)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_set_all_seeds_deterministic_numpy():
    """NumPy random produces identical sequences after seeding twice."""
    set_all_seeds(456)
    a = np.random.rand(5).tolist()
    set_all_seeds(456)
    b = np.random.rand(5).tolist()
    assert a == b


def test_set_all_seeds_deterministic_torch():
    """PyTorch random produces identical tensors after seeding twice."""
    set_all_seeds(789)
    a = torch.rand(5)
    set_all_seeds(789)
    b = torch.rand(5)
    assert torch.equal(a, b)


def test_different_seeds_produce_different_values():
    """Different seeds should produce different random sequences."""
    set_all_seeds(1)
    a = random.random()
    set_all_seeds(2)
    b = random.random()
    assert a != b
