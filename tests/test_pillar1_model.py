"""Test Pillar 1 model forward pass."""

import torch
from fedstable.pillar1_depeg.model import DepegPredictor


def test_depeg_predictor_shape():
    model = DepegPredictor(input_size=8, hidden_size=64, num_layers=2, num_horizons=3)
    x = torch.randn(4, 24, 8)
    out = model(x)
    assert out.shape == (4, 3)
    probs = model.predict_proba(x)
    assert probs.shape == (4, 3)
    assert (probs >= 0).all() and (probs <= 1).all()
