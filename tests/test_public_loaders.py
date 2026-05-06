"""
Tests for public-dataset loaders in model/data_loader.py.
All tests use small in-memory mock files so no real datasets are needed.
"""
import os
import shutil
import tempfile

import numpy as np
import pytest
import scipy.io as sio
import torch

from model.data_loader import (
    DATASET_CATALOG,
    RAW_FEATURES,
    TOTAL_FEATURES,
    ARILDataset,
    MMFiDataset,
    NTUFiDataset,
    SignFiDataset,
    UTHARDataset,
    WiARDataset,
    autodetect_and_load,
    universal_csi_interpolate,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def tmpdir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ── universal_csi_interpolate ─────────────────────────────────────────────────

class TestUniversalInterpolate:
    def test_output_shape_upscale(self):
        raw = np.random.rand(50, 90).astype(np.float32)
        out = universal_csi_interpolate(raw, target_seq_len=128, target_features=RAW_FEATURES)
        assert out.shape == (128, RAW_FEATURES)

    def test_output_shape_already_correct(self):
        raw = np.random.rand(128, RAW_FEATURES).astype(np.float32)
        out = universal_csi_interpolate(raw)
        assert out.shape == (128, RAW_FEATURES)

    def test_output_shape_downscale(self):
        raw = np.random.rand(500, 342).astype(np.float32)
        out = universal_csi_interpolate(raw, target_seq_len=128, target_features=RAW_FEATURES)
        assert out.shape == (128, RAW_FEATURES)

    def test_output_dtype(self):
        raw = np.random.rand(100, 90).astype(np.float64)
        out = universal_csi_interpolate(raw)
        assert out.dtype == np.float32

    def test_no_negative_values(self):
        raw = np.abs(np.random.rand(100, 90)).astype(np.float32)
        out = universal_csi_interpolate(raw)
        assert out.min() >= 0.0

    def test_empty_input_returns_zeros(self):
        raw = np.zeros((0, 90), dtype=np.float32)
        out = universal_csi_interpolate(raw, target_seq_len=128, target_features=RAW_FEATURES)
        assert out.shape == (128, RAW_FEATURES)
        assert np.all(out == 0.0)


# ── DATASET_CATALOG ───────────────────────────────────────────────────────────

class TestDatasetCatalog:
    REQUIRED_KEYS = {"description", "activities", "n_classes", "hardware",
                     "features", "format", "url", "loader"}

    def test_catalog_has_expected_entries(self):
        assert "UT-HAR" in DATASET_CATALOG
        assert "NTU-Fi-HAR" in DATASET_CATALOG
        assert "MM-Fi" in DATASET_CATALOG
        assert "SignFi" in DATASET_CATALOG
        assert "WiAR" in DATASET_CATALOG
        assert "ARIL" in DATASET_CATALOG

    def test_each_entry_has_required_keys(self):
        for name, info in DATASET_CATALOG.items():
            for key in self.REQUIRED_KEYS:
                assert key in info, f"DATASET_CATALOG['{name}'] missing key '{key}'"

    def test_n_classes_positive(self):
        for name, info in DATASET_CATALOG.items():
            assert info["n_classes"] > 0, f"n_classes must be >0 for {name}"

    def test_features_positive(self):
        for name, info in DATASET_CATALOG.items():
            assert info["features"] > 0, f"features must be >0 for {name}"


# ── UTHARDataset ──────────────────────────────────────────────────────────────

class TestUTHARDataset:
    def test_csv_layout_loads_one_sample(self, tmpdir):
        act_dir = os.path.join(tmpdir, "walk")
        os.makedirs(act_dir)
        # 181 cols: 1 time + 90 amp + 90 phase
        data = np.random.rand(200, 181).astype(np.float32)
        np.savetxt(os.path.join(act_dir, "rec1.csv"), data, delimiter=",")

        ds = UTHARDataset(tmpdir, seq_len=128)
        assert len(ds) >= 1

    def test_output_tensor_shape(self, tmpdir):
        act_dir = os.path.join(tmpdir, "walk")
        os.makedirs(act_dir)
        data = np.random.rand(200, 181).astype(np.float32)
        np.savetxt(os.path.join(act_dir, "rec1.csv"), data, delimiter=",")

        ds = UTHARDataset(tmpdir, seq_len=128)
        x, y = ds[0]
        assert x.shape == (128, TOTAL_FEATURES)
        assert x.dtype == torch.float32
        assert isinstance(y.item(), int)

    def test_npy_layout_loads_samples(self, tmpdir):
        arr = np.random.rand(5, 250, 90).astype(np.float32)
        np.save(os.path.join(tmpdir, "walk.npy"), arr)

        ds = UTHARDataset(tmpdir, seq_len=128)
        assert len(ds) == 5
        x, y = ds[0]
        assert x.shape == (128, TOTAL_FEATURES)

    def test_empty_dir_returns_empty_dataset(self, tmpdir):
        ds = UTHARDataset(tmpdir, seq_len=128)
        assert len(ds) == 0

    def test_label_in_valid_range(self, tmpdir):
        act_dir = os.path.join(tmpdir, "walk")
        os.makedirs(act_dir)
        data = np.random.rand(200, 90).astype(np.float32)
        np.savetxt(os.path.join(act_dir, "rec1.csv"), data, delimiter=",")

        ds = UTHARDataset(tmpdir, seq_len=128)
        for _, y in ds:
            assert 0 <= y.item() <= 6


# ── NTUFiDataset ──────────────────────────────────────────────────────────────

class TestNTUFiDataset:
    def _make_mat_3_114_T(self, path, T=297):
        data = {"csi_data": np.random.rand(3, 114, T).astype(np.float32)}
        sio.savemat(path, data)

    def test_loads_mat_3_114_T(self, tmpdir):
        self._make_mat_3_114_T(os.path.join(tmpdir, "walk_s1.mat"))
        ds = NTUFiDataset(tmpdir, seq_len=128)
        assert len(ds) >= 1

    def test_output_shape(self, tmpdir):
        self._make_mat_3_114_T(os.path.join(tmpdir, "walk_s1.mat"))
        ds = NTUFiDataset(tmpdir, seq_len=128)
        x, y = ds[0]
        assert x.shape == (128, TOTAL_FEATURES)

    def test_loads_mat_T_342(self, tmpdir):
        data = {"csi_data": np.random.rand(300, 342).astype(np.float32)}
        sio.savemat(os.path.join(tmpdir, "run_s1.mat"), data)
        ds = NTUFiDataset(tmpdir, seq_len=128)
        assert len(ds) >= 1
        x, _ = ds[0]
        assert x.shape == (128, TOTAL_FEATURES)

    def test_subfolder_label_inference(self, tmpdir):
        subdir = os.path.join(tmpdir, "walk", "subject01")
        os.makedirs(subdir)
        self._make_mat_3_114_T(os.path.join(subdir, "sample.mat"))
        ds = NTUFiDataset(tmpdir, seq_len=128)
        assert len(ds) >= 1
        _, y = ds[0]
        assert y.item() == NTUFiDataset.ACTIVITY_MAP["walk"]

    def test_dtype(self, tmpdir):
        self._make_mat_3_114_T(os.path.join(tmpdir, "fall_s1.mat"))
        ds = NTUFiDataset(tmpdir, seq_len=128)
        x, _ = ds[0]
        assert x.dtype == torch.float32


# ── MMFiDataset ───────────────────────────────────────────────────────────────

class TestMMFiDataset:
    def test_loads_T_3_114(self, tmpdir):
        data = {"csi": np.random.rand(297, 3, 114).astype(np.float32)}
        sio.savemat(os.path.join(tmpdir, "mock_action.mat"), data)

        ds = MMFiDataset(tmpdir, seq_len=128)
        assert len(ds) >= 1

    def test_output_shape(self, tmpdir):
        data = {"csi": np.random.rand(297, 3, 114).astype(np.float32)}
        sio.savemat(os.path.join(tmpdir, "mock_action.mat"), data)

        ds = MMFiDataset(tmpdir, seq_len=128)
        x, y = ds[0]
        assert x.shape == (128, TOTAL_FEATURES)
        assert x.dtype == torch.float32

    def test_loads_3_114_T(self, tmpdir):
        data = {"csi": np.random.rand(3, 114, 297).astype(np.float32)}
        sio.savemat(os.path.join(tmpdir, "mock_action.mat"), data)
        ds = MMFiDataset(tmpdir, seq_len=128)
        assert len(ds) >= 1
        x, _ = ds[0]
        assert x.shape == (128, TOTAL_FEATURES)

    def test_activity_label_auto_mapping(self, tmpdir):
        for act in ("E01", "E02"):
            act_dir = os.path.join(tmpdir, act)
            os.makedirs(act_dir)
            data = {"csi": np.random.rand(297, 3, 114).astype(np.float32)}
            sio.savemat(os.path.join(act_dir, "wifi_csi.mat"), data)

        ds = MMFiDataset(tmpdir, seq_len=128)
        labels = {y.item() for _, y in ds}
        assert len(labels) == 2  # two distinct labels for two activities


# ── SignFiDataset ─────────────────────────────────────────────────────────────

class TestSignFiDataset:
    def test_full_mat_loads(self, tmpdir):
        N = 10
        data = {
            "CSI_data_all": np.random.rand(200, 30, 3, N).astype(np.float32),
            "label_all":    np.arange(N, dtype=np.int32),
        }
        sio.savemat(os.path.join(tmpdir, "SignFi_dataset.mat"), data)

        ds = SignFiDataset(tmpdir, seq_len=128)
        assert len(ds) == N

    def test_output_shape(self, tmpdir):
        N = 5
        data = {
            "CSI_data_all": np.random.rand(200, 30, 3, N).astype(np.float32),
            "label_all":    np.zeros(N, dtype=np.int32),
        }
        sio.savemat(os.path.join(tmpdir, "SignFi_dataset.mat"), data)

        ds = SignFiDataset(tmpdir, seq_len=128)
        x, y = ds[0]
        assert x.shape == (128, TOTAL_FEATURES)
        assert x.dtype == torch.float32

    def test_per_gesture_mat(self, tmpdir):
        data = {"csi": np.random.rand(200, 30, 3).astype(np.float32)}
        sio.savemat(os.path.join(tmpdir, "gesture_001.mat"), data)

        ds = SignFiDataset(tmpdir, seq_len=128)
        assert len(ds) >= 1
        x, _ = ds[0]
        assert x.shape == (128, TOTAL_FEATURES)


# ── WiARDataset ───────────────────────────────────────────────────────────────

class TestWiARDataset:
    def test_loads_csv_from_subfolder(self, tmpdir):
        act_dir = os.path.join(tmpdir, "walk")
        os.makedirs(act_dir)
        data = np.random.rand(300, 90).astype(np.float32)
        np.savetxt(os.path.join(act_dir, "rec1.csv"), data, delimiter=",")

        ds = WiARDataset(tmpdir, seq_len=128)
        assert len(ds) >= 1

    def test_output_shape(self, tmpdir):
        act_dir = os.path.join(tmpdir, "run")
        os.makedirs(act_dir)
        data = np.random.rand(300, 90).astype(np.float32)
        np.savetxt(os.path.join(act_dir, "rec1.csv"), data, delimiter=",")

        ds = WiARDataset(tmpdir, seq_len=128)
        x, y = ds[0]
        assert x.shape == (128, TOTAL_FEATURES)
        assert x.dtype == torch.float32

    def test_label_from_path(self, tmpdir):
        act_dir = os.path.join(tmpdir, "fall")
        os.makedirs(act_dir)
        data = np.random.rand(300, 90).astype(np.float32)
        np.savetxt(os.path.join(act_dir, "rec1.csv"), data, delimiter=",")

        ds = WiARDataset(tmpdir, seq_len=128)
        _, y = ds[0]
        assert y.item() == WiARDataset.ACTIVITY_MAP["fall"]


# ── ARILDataset ───────────────────────────────────────────────────────────────

class TestARILDataset:
    def test_loads_csv_from_subfolder(self, tmpdir):
        act_dir = os.path.join(tmpdir, "walk")
        os.makedirs(act_dir)
        data = np.random.rand(300, 90).astype(np.float32)
        np.savetxt(os.path.join(act_dir, "rec1.csv"), data, delimiter=",")

        ds = ARILDataset(tmpdir, seq_len=128)
        assert len(ds) >= 1

    def test_output_shape(self, tmpdir):
        act_dir = os.path.join(tmpdir, "fall")
        os.makedirs(act_dir)
        data = np.random.rand(300, 90).astype(np.float32)
        np.savetxt(os.path.join(act_dir, "rec1.csv"), data, delimiter=",")

        ds = ARILDataset(tmpdir, seq_len=128)
        x, y = ds[0]
        assert x.shape == (128, TOTAL_FEATURES)
        assert x.dtype == torch.float32

    def test_label_from_path(self, tmpdir):
        act_dir = os.path.join(tmpdir, "boxing")
        os.makedirs(act_dir)
        data = np.random.rand(300, 90).astype(np.float32)
        np.savetxt(os.path.join(act_dir, "rec1.csv"), data, delimiter=",")

        ds = ARILDataset(tmpdir, seq_len=128)
        _, y = ds[0]
        assert y.item() == ARILDataset.ACTIVITY_MAP["boxing"]

    def test_extra_cols_truncated(self, tmpdir):
        act_dir = os.path.join(tmpdir, "run")
        os.makedirs(act_dir)
        data = np.random.rand(300, 95).astype(np.float32)  # 95 cols, only first 90 used
        np.savetxt(os.path.join(act_dir, "rec1.csv"), data, delimiter=",")

        ds = ARILDataset(tmpdir, seq_len=128)
        assert len(ds) >= 1
        x, _ = ds[0]
        assert x.shape == (128, TOTAL_FEATURES)


# ── autodetect_and_load ───────────────────────────────────────────────────────

class TestAutodetect:
    def test_detects_mmfi_from_mat(self, tmpdir):
        data = {"csi": np.random.rand(297, 3, 114).astype(np.float32)}
        sio.savemat(os.path.join(tmpdir, "sample.mat"), data)

        ds, name = autodetect_and_load(tmpdir, seq_len=128)
        assert name in ("MM-Fi", "NTU-Fi-HAR", "SignFi")  # any MAT-based

    def test_detects_uthar_from_npy(self, tmpdir):
        arr = np.random.rand(3, 250, 90).astype(np.float32)
        np.save(os.path.join(tmpdir, "walk.npy"), arr)
        os.makedirs(os.path.join(tmpdir, "walk"), exist_ok=True)

        ds, name = autodetect_and_load(tmpdir, seq_len=128)
        assert name == "UT-HAR"

    def test_detects_aril_from_csv(self, tmpdir):
        act_dir = os.path.join(tmpdir, "clean")
        os.makedirs(act_dir)
        data = np.random.rand(300, 90).astype(np.float32)
        np.savetxt(os.path.join(act_dir, "rec1.csv"), data, delimiter=",")

        ds, name = autodetect_and_load(tmpdir, seq_len=128)
        assert name == "ARIL"

    def test_raises_on_empty_dir(self, tmpdir):
        with pytest.raises(ValueError):
            autodetect_and_load(tmpdir, seq_len=128)

    def test_returns_dataset_with_correct_item_shape(self, tmpdir):
        data = {"csi": np.random.rand(297, 3, 114).astype(np.float32)}
        sio.savemat(os.path.join(tmpdir, "sample.mat"), data)

        ds, _ = autodetect_and_load(tmpdir, seq_len=128)
        assert len(ds) >= 1
        x, _ = ds[0]
        assert x.shape == (128, TOTAL_FEATURES)
