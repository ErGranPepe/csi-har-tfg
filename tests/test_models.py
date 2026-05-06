"""
tests/test_models.py
====================
Unit tests for all HAR model architectures.
Run: py -m pytest tests/ -v
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest

from model.models_zoo import build_model, MODEL_CONFIGS, count_parameters
from model.data_loader import TOTAL_FEATURES, ACTIVITY_NAMES
NUM_CLASSES = len(ACTIVITY_NAMES)


# ── Fixtures ──────────────────────────────────────────────────────────────────

BATCH   = 4
SEQ_LEN = 128
N_CLS   = len([
    "Standing", "Walking", "Get Down",
    "Sitting",  "Get Up",  "Lying",   "No Person"
])  # 7

@pytest.fixture(params=list(MODEL_CONFIGS.keys()))
def model_name(request):
    return request.param

@pytest.fixture
def random_input():
    return torch.randn(BATCH, SEQ_LEN, TOTAL_FEATURES)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestModelOutput:
    def test_output_shape(self, model_name, random_input):
        """Model output must be (B, 7) logits."""
        m = build_model(model_name)
        m.eval()
        with torch.no_grad():
            out = m(random_input)
        assert out.shape == (BATCH, N_CLS), \
            f"{model_name}: expected ({BATCH},{N_CLS}), got {out.shape}"

    def test_output_finite(self, model_name, random_input):
        """No NaN or Inf in output."""
        m = build_model(model_name)
        m.eval()
        with torch.no_grad():
            out = m(random_input)
        assert torch.isfinite(out).all(), f"{model_name}: output contains NaN/Inf"

    def test_softmax_sums_to_one(self, model_name, random_input):
        """Softmax probabilities must sum to 1.0 per sample."""
        import torch.nn.functional as F
        m = build_model(model_name)
        m.eval()
        with torch.no_grad():
            probs = F.softmax(m(random_input), dim=1)
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones(BATCH), atol=1e-5), \
            f"{model_name}: softmax does not sum to 1"

    def test_parameter_count_positive(self, model_name):
        """Model must have at least 100K trainable parameters."""
        m = build_model(model_name)
        n = count_parameters(m)
        assert n > 100_000, f"{model_name}: only {n} parameters"

    def test_single_sample_inference(self, model_name):
        """Single-sample batch must work (batch size 1)."""
        m = build_model(model_name)
        m.eval()
        x = torch.randn(1, SEQ_LEN, TOTAL_FEATURES)
        with torch.no_grad():
            out = m(x)
        assert out.shape == (1, N_CLS)

    def test_gradient_flows(self, model_name, random_input):
        """Backward pass must produce finite gradients."""
        m = build_model(model_name)
        m.train()
        out = m(random_input)
        loss = out.mean()
        loss.backward()
        for name, p in m.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), \
                    f"{model_name}: NaN gradient in {name}"


class TestModelRegistry:
    def test_all_models_in_registry(self):
        expected = {"SimpleLSTM", "BiLSTM", "FCN", "Transformer"}
        assert set(MODEL_CONFIGS.keys()) == expected

    def test_each_model_has_description(self):
        for name, cfg in MODEL_CONFIGS.items():
            assert "description" in cfg, f"{name} missing 'description'"
            assert "paper" in cfg, f"{name} missing 'paper'"

    def test_build_model_unknown_raises(self):
        with pytest.raises((KeyError, ValueError)):
            build_model("NonExistentModel")
