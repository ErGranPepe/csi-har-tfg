"""
tests/test_zone_classifier.py
==============================
Tests for the ZoneClassifier and zone feature extraction.
Run: py -m pytest tests/ -v
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from model.position_estimator import (
    ZoneClassifier, ZoneDataset, ZONE_NAMES, NUM_ZONES,
    extract_zone_features,
)
from model.data_loader import RAW_FEATURES, AMP_MAX

SEQ_LEN = 128


class TestExtractZoneFeatures:
    def test_output_shape(self):
        window = np.random.rand(SEQ_LEN, RAW_FEATURES).astype(np.float32) * AMP_MAX
        feat = extract_zone_features(window)
        assert feat.shape == (16,), f"expected (16,), got {feat.shape}"

    def test_output_finite(self):
        window = np.random.rand(SEQ_LEN, RAW_FEATURES).astype(np.float32) * AMP_MAX
        feat = extract_zone_features(window)
        assert np.isfinite(feat).all(), "extract_zone_features: NaN/Inf in output"

    def test_zero_window(self):
        """Zero amplitude window should not crash."""
        window = np.zeros((SEQ_LEN, RAW_FEATURES), dtype=np.float32)
        feat = extract_zone_features(window)
        assert feat.shape == (16,)
        assert np.isfinite(feat).all()

    def test_proximity_higher_raw_amplitude_than_far(self):
        """
        Raw (unnormalized) proximity features should have higher mean amplitude
        than Far features — physical constraint from ITU-R P.1238 path loss.
        Uses extract_zone_features directly on simulated windows.
        """
        from model.data_loader import simulate_activity, AMP_MAX
        rng = np.random.default_rng(0)
        # Simulate proximity: high mean amplitude (strong signal, short path)
        prox_amplitudes = rng.uniform(300, 500, (20, 128, 456)).astype(np.float32)
        far_amplitudes  = rng.uniform(15,   50, (20, 128, 456)).astype(np.float32)
        prox_means = np.array([extract_zone_features(w)[0] for w in prox_amplitudes])
        far_means  = np.array([extract_zone_features(w)[0] for w in far_amplitudes])
        assert prox_means.mean() > far_means.mean(), \
            f"Proximity raw mean {prox_means.mean():.2f} not > Far {far_means.mean():.2f}"


class TestZoneClassifier:
    def test_output_shape(self):
        clf = ZoneClassifier()
        clf.eval()
        x = torch.randn(8, 16)
        with torch.no_grad():
            out = clf(x)
        assert out.shape == (8, NUM_ZONES), \
            f"expected (8,{NUM_ZONES}), got {out.shape}"

    def test_output_finite(self):
        clf = ZoneClassifier()
        clf.eval()
        x = torch.randn(8, 16)
        with torch.no_grad():
            out = clf(x)
        assert torch.isfinite(out).all()

    def test_predict_method(self):
        clf = ZoneClassifier()
        clf.eval()
        window = np.random.rand(SEQ_LEN, RAW_FEATURES).astype(np.float32) * AMP_MAX
        feat = extract_zone_features(window)
        mean = np.zeros(16)
        std  = np.ones(16)
        zone_idx, zone_probs = clf.predict(feat, mean, std)
        assert 0 <= zone_idx < NUM_ZONES
        assert zone_probs.shape == (NUM_ZONES,)
        assert abs(zone_probs.sum() - 1.0) < 1e-5

    def test_zone_names_count(self):
        assert len(ZONE_NAMES) == NUM_ZONES == 4


class TestZoneDataset:
    def test_length(self):
        ds = ZoneDataset(n_per_zone=50)
        assert len(ds) == 50 * NUM_ZONES

    def test_item_shapes(self):
        ds = ZoneDataset(n_per_zone=20)
        x, y = ds[0]
        assert x.shape == (16,), f"expected (16,), got {x.shape}"
        assert y.shape == (), f"expected scalar, got {y.shape}"

    def test_label_range(self):
        ds = ZoneDataset(n_per_zone=20)
        for i in range(len(ds)):
            _, y = ds[i]
            assert 0 <= int(y) < NUM_ZONES

    def test_mean_std_shapes(self):
        ds = ZoneDataset(n_per_zone=20)
        assert ds.mean.shape == (16,)
        assert ds.std.shape  == (16,)
        assert np.isfinite(ds.mean).all()
        assert np.isfinite(ds.std).all()
        assert (ds.std >= 0).all()
