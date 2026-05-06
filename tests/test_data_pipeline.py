"""
tests/test_data_pipeline.py
============================
Tests for the CSI feature extraction and simulation pipeline.
Run: py -m pytest tests/ -v
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from model.data_loader import (
    SUBCARRIERS, ANTENNA_PAIRS, RAW_FEATURES, TOTAL_FEATURES,
    AMP_MAX, AMP_MIN,
    simulate_activity, compute_pca, full_preprocess,
    SimulatedCSIDataset,
)

NUM_CLASSES = 7
SEQ_LEN = 128


class TestSimulateActivity:
    @pytest.mark.parametrize("cls_idx", range(NUM_CLASSES))
    def test_output_shape(self, cls_idx):
        out = simulate_activity(cls_idx, SEQ_LEN, seed=42)
        assert out.shape == (SEQ_LEN, RAW_FEATURES), \
            f"class {cls_idx}: expected ({SEQ_LEN},{RAW_FEATURES}), got {out.shape}"

    @pytest.mark.parametrize("cls_idx", range(NUM_CLASSES))
    def test_output_dtype(self, cls_idx):
        out = simulate_activity(cls_idx, SEQ_LEN, seed=0)
        assert out.dtype in (np.float32, np.float64), \
            f"class {cls_idx}: unexpected dtype {out.dtype}"

    @pytest.mark.parametrize("cls_idx", range(NUM_CLASSES))
    def test_amplitude_range(self, cls_idx):
        """Simulated amplitudes should be in a physically plausible range."""
        out = simulate_activity(cls_idx, SEQ_LEN, seed=7)
        assert out.min() >= -AMP_MAX * 0.5, \
            f"class {cls_idx}: amplitude too negative ({out.min():.1f})"
        assert out.max() <= AMP_MAX * 2.0, \
            f"class {cls_idx}: amplitude too large ({out.max():.1f})"

    def test_reproducible_with_seed(self):
        a = simulate_activity(1, SEQ_LEN, seed=123)
        b = simulate_activity(1, SEQ_LEN, seed=123)
        assert np.allclose(a, b), "simulate_activity not reproducible with same seed"

    def test_different_seeds_differ(self):
        a = simulate_activity(1, SEQ_LEN, seed=1)
        b = simulate_activity(1, SEQ_LEN, seed=2)
        assert not np.allclose(a, b), "Different seeds produced identical output"

    def test_no_nan_inf(self):
        for cls_idx in range(NUM_CLASSES):
            out = simulate_activity(cls_idx, SEQ_LEN, seed=99)
            assert np.isfinite(out).all(), f"class {cls_idx}: NaN/Inf in simulate_activity"


class TestComputePCA:
    def test_output_shape(self):
        amp = np.random.rand(SEQ_LEN, RAW_FEATURES).astype(np.float32) * AMP_MAX
        out = compute_pca(amp)
        assert out.shape == (SEQ_LEN, 12), \
            f"compute_pca: expected ({SEQ_LEN},12), got {out.shape}"

    def test_output_finite(self):
        amp = np.random.rand(SEQ_LEN, RAW_FEATURES).astype(np.float32) * AMP_MAX
        out = compute_pca(amp)
        assert np.isfinite(out).all(), "compute_pca: NaN/Inf in output"

    def test_small_batch(self):
        """PCA should handle batches smaller than n_components gracefully."""
        amp = np.random.rand(2, RAW_FEATURES).astype(np.float32) * AMP_MAX
        out = compute_pca(amp)
        assert out.shape[1] == 12

    def test_all_zeros_no_crash(self):
        """All-zero input (zero variance) must not crash — returns zeros."""
        amp = np.zeros((SEQ_LEN, RAW_FEATURES), dtype=np.float32)
        out = compute_pca(amp)
        assert out.shape == (SEQ_LEN, 12)
        assert np.isfinite(out).all()


class TestFullPreprocess:
    def test_output_shape(self):
        amp = np.clip(
            np.random.rand(SEQ_LEN, RAW_FEATURES).astype(np.float32) * AMP_MAX,
            0, AMP_MAX
        )
        out = full_preprocess(amp)
        assert out.shape == (SEQ_LEN, TOTAL_FEATURES), \
            f"full_preprocess: expected ({SEQ_LEN},{TOTAL_FEATURES}), got {out.shape}"

    def test_output_normalised(self):
        amp = np.random.rand(SEQ_LEN, RAW_FEATURES).astype(np.float32) * AMP_MAX
        out = full_preprocess(amp)
        assert out.min() >= -1e-5, f"full_preprocess: min {out.min():.4f} below 0"
        assert out.max() <= 1.0 + 1e-5, f"full_preprocess: max {out.max():.4f} above 1"

    def test_output_dtype(self):
        amp = np.random.rand(SEQ_LEN, RAW_FEATURES).astype(np.float32) * AMP_MAX
        out = full_preprocess(amp)
        assert out.dtype == np.float32

    def test_all_zeros_no_crash(self):
        """Zero-amplitude window (empty room with no signal variance) must not crash."""
        amp = np.zeros((SEQ_LEN, RAW_FEATURES), dtype=np.float32)
        out = full_preprocess(amp)
        assert out.shape == (SEQ_LEN, TOTAL_FEATURES)
        assert np.isfinite(out).all()

    def test_nan_input_sanitised(self):
        """NaN values in raw CSI must be replaced, not propagate to output."""
        amp = np.random.rand(SEQ_LEN, RAW_FEATURES).astype(np.float32) * AMP_MAX
        amp[10, 50] = float('nan')
        amp[64, 200] = float('inf')
        out = full_preprocess(amp)
        assert np.isfinite(out).all(), "NaN/Inf survived full_preprocess"


class TestSimulatedDataset:
    def test_length(self):
        ds = SimulatedCSIDataset(num_samples=70, seq_len=SEQ_LEN)
        assert len(ds) == 70

    def test_item_shapes(self):
        ds = SimulatedCSIDataset(num_samples=70, seq_len=SEQ_LEN)
        x, y = ds[0]
        assert x.shape == (SEQ_LEN, TOTAL_FEATURES), \
            f"Dataset item x: expected ({SEQ_LEN},{TOTAL_FEATURES}), got {x.shape}"
        assert y.shape == (), f"Dataset item y: expected scalar, got {y.shape}"

    def test_label_range(self):
        ds = SimulatedCSIDataset(num_samples=70, seq_len=SEQ_LEN)
        for i in range(len(ds)):
            _, y = ds[i]
            assert 0 <= int(y) < NUM_CLASSES, \
                f"Dataset item {i}: label {y} out of range [0,{NUM_CLASSES})"

    def test_balanced_classes(self):
        """Dataset should have equal or near-equal samples per class."""
        n = 70   # 10 per class
        ds = SimulatedCSIDataset(num_samples=n, seq_len=SEQ_LEN)
        counts = np.zeros(NUM_CLASSES, dtype=int)
        for i in range(len(ds)):
            _, y = ds[i]
            counts[int(y)] += 1
        assert counts.min() >= 1, "Some class has 0 samples"
        ratio = counts.max() / (counts.min() + 1e-8)
        assert ratio < 3.0, f"Dataset imbalance ratio {ratio:.1f} too large"
