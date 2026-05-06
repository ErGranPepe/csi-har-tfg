import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn import decomposition
from scipy import signal as sp_signal

# ── Repository-exact constants ────────────────────────────────────────────────
SUBCARRIERS     = 114          # 5 GHz band (from repo dataset.py)
ANTENNA_PAIRS   = 4            # 2×2 MIMO → 4 pairs
RAW_FEATURES    = SUBCARRIERS * ANTENNA_PAIRS   # 456
PCA_COMPONENTS  = 3
PCA_FEATURES    = PCA_COMPONENTS * ANTENNA_PAIRS  # 12
TOTAL_FEATURES  = RAW_FEATURES + PCA_FEATURES     # 468  ← exact repo value
SAMPLE_RATE     = 20           # Hz – TP-Link CSI capture rate

# Amplitude normalisation bounds (from repo dataset.py)
AMP_MIN, AMP_MAX = 0.0, 577.6582

# Activity index mapping (from repo dataset.py class_to_idx)
ACTIVITY_NAMES  = ["Standing", "Walking", "Get Down",
                   "Sitting",  "Get Up",  "Lying",   "No Person"]
ACTIVITY_COLORS = ["#3498db", "#e74c3c", "#f39c12",
                   "#9b59b6", "#1abc9c", "#e67e22", "#7f8c8d"]

# Class proportions from repo train_lstm.py – used as inverse weights
CLASS_PROPORTIONS = np.array([0.113, 0.439, 0.0379,
                               0.1515, 0.0379, 0.1212, 0.1363])


# ── Preprocessing ─────────────────────────────────────────────────────────────

def hampel_filter(data: np.ndarray, k: int = 7, t0: float = 3.0) -> np.ndarray:
    """
    Vectorised Hampel identifier (exactly as in repo data_calibration.py).
    data : (N, F)  – timesteps × features
    Replaces outliers with the local median.
    """
    N, F = data.shape
    out  = data.copy()
    L    = 1.4826  # consistency constant for Gaussian distribution
    for i in range(N):
        lo, hi   = max(0, i - k), min(N, i + k + 1)
        window   = data[lo:hi]                           # (w, F)
        med      = np.median(window, axis=0)             # (F,)
        mad      = L * np.median(np.abs(window - med), axis=0)
        mask     = np.abs(data[i] - med) > t0 * mad
        out[i, mask] = med[mask]
    return out


def wavelet_denoise(data: np.ndarray) -> np.ndarray:
    """
    Wavelet denoising (sym5, as in repo).  Falls back to Savitzky-Golay
    when PyWavelets is unavailable.
    data : (N, F)
    """
    try:
        import pywt
        wavelet = 'sym5'
        max_lv  = pywt.dwt_max_level(data.shape[0], wavelet)
        level   = min(3, max_lv)   # never exceed the safe level for this window
        out = np.zeros_like(data)
        for f in range(data.shape[1]):
            coeffs = pywt.wavedec(data[:, f], wavelet, level=level)
            thr    = 0.06 * np.max(np.abs(coeffs[-1]))
            coeffs = [pywt.threshold(c, thr, mode='soft') for c in coeffs]
            rec    = pywt.waverec(coeffs, wavelet)
            out[:, f] = rec[:data.shape[0]]
        return out
    except ImportError:
        # Savitzky-Golay as fallback (same smoothing character)
        wl = min(11, data.shape[0] - (1 - data.shape[0] % 2))
        if wl >= 5:
            return sp_signal.savgol_filter(data, wl, 3, axis=0)
        return data


def compute_pca(amplitudes: np.ndarray) -> np.ndarray:
    """
    3 PCA components per antenna pair → 12 features total.
    amplitudes : (N, 456)
    Returns    : (N, 12)
    """
    pca  = decomposition.PCA(n_components=PCA_COMPONENTS)
    segs = []
    for i in range(ANTENNA_PAIRS):
        block = amplitudes[:, i * SUBCARRIERS:(i + 1) * SUBCARRIERS]
        if block.shape[0] >= PCA_COMPONENTS:
            segs.append(pca.fit_transform(block))
        else:
            segs.append(np.zeros((block.shape[0], PCA_COMPONENTS)))
    return np.concatenate(segs, axis=1)   # (N, 12)


def full_preprocess(amplitudes: np.ndarray,
                    apply_hampel: bool = True,
                    apply_denoise: bool = True) -> np.ndarray:
    """
    Exact repo pipeline:
      Hampel → wavelet denoise → PCA append → clip+normalise
    amplitudes : (seq_len, 456)
    Returns    : (seq_len, 468)  normalised [0, 1]
    """
    amp = amplitudes.copy()

    if apply_hampel and amp.shape[0] >= 15:
        amp = hampel_filter(amp)

    if apply_denoise and amp.shape[0] >= 8:
        amp = wavelet_denoise(amp)

    pca_feat = compute_pca(amp)                          # (N, 12)
    features = np.concatenate([amp, pca_feat], axis=1)  # (N, 468)
    features = np.clip(features, AMP_MIN, AMP_MAX)
    features = (features - AMP_MIN) / (AMP_MAX - AMP_MIN + 1e-8)
    return features.astype(np.float32)


# ── Realistic per-activity CSI simulation ────────────────────────────────────

def simulate_activity(class_idx: int, seq_len: int, seed: int = None) -> np.ndarray:
    """
    Physics-inspired CSI amplitude simulation based on the real dataset:

    Class patterns derived from the repo notebooks (VisualizeCSI.ipynb):
    - No Person  (6): flat reference ~200, tiny AWGN (σ=3)
    - Standing   (0): breathing oscillation ~0.3 Hz on top of base
    - Walking    (1): strong periodic gait ~2 Hz, high σ, cross-subcarrier phase shift
    - Sitting    (3): very slow drift ~0.1 Hz, low σ
    - Lying      (5): near-static, lowest σ, sub-0.05 Hz drift
    - Get Up     (4): exponentially decaying burst at start (person rises)
    - Get Down   (2): exponentially growing burst at end (person lowers)

    Returns amplitude matrix (seq_len, 456) in arbitrary units [0, AMP_MAX].
    """
    rng = np.random.default_rng(seed)
    t   = np.arange(seq_len) / SAMPLE_RATE        # time axis [s]

    # Random per-subcarrier phase offsets (multipath diversity)
    phi = rng.uniform(0, 2 * np.pi, RAW_FEATURES)

    # Base signal (mean ~200, realistic offset per subcarrier)
    base_mean = rng.uniform(150, 250, RAW_FEATURES)
    base = np.ones((seq_len, 1)) * base_mean[None, :]  # (N, 456)

    if class_idx == 6:   # No Person
        return base + rng.normal(0, 3, (seq_len, RAW_FEATURES))

    elif class_idx == 0:  # Standing – breathing ~0.3 Hz
        env = 15 * np.sin(2 * np.pi * 0.3 * t + phi[0])
        mod = env[:, None] * np.cos(phi[None, :])
        return base + mod + rng.normal(0, 5, (seq_len, RAW_FEATURES))

    elif class_idx == 1:  # Walking – gait ~1.8–2.2 Hz, spatially varying
        freq  = rng.uniform(1.8, 2.2)
        amp_g = rng.uniform(30, 60, RAW_FEATURES)
        gait  = amp_g[None, :] * np.sin(2 * np.pi * freq * t[:, None] + phi[None, :])
        return base + gait + rng.normal(0, 15, (seq_len, RAW_FEATURES))

    elif class_idx == 3:  # Sitting
        drift = 8 * np.sin(2 * np.pi * 0.1 * t)
        return base + drift[:, None] + rng.normal(0, 4, (seq_len, RAW_FEATURES))

    elif class_idx == 5:  # Lying
        drift = 3 * np.sin(2 * np.pi * 0.05 * t)
        return base + drift[:, None] + rng.normal(0, 2, (seq_len, RAW_FEATURES))

    elif class_idx == 4:  # Get Up – transient burst decaying from t=0
        tau  = seq_len / SAMPLE_RATE * 0.4
        env  = 70 * np.exp(-t / tau) + 8
        burst = env[:, None] * rng.standard_normal((seq_len, RAW_FEATURES))
        return base + burst

    elif class_idx == 2:  # Get Down – transient burst growing toward end
        tau  = seq_len / SAMPLE_RATE * 0.4
        env  = 70 * np.exp(-(t[-1] - t) / tau) + 8
        burst = env[:, None] * rng.standard_normal((seq_len, RAW_FEATURES))
        return base + burst

    return base


# ── Dataset ───────────────────────────────────────────────────────────────────

class SimulatedCSIDataset(Dataset):
    """
    Fully preprocessed simulated dataset matching repository parameters:
    features = 468 (456 amplitudes + 12 PCA), 7 activity classes.
    """

    def __init__(self, num_samples: int = 3000, seq_len: int = 128):
        # Strategy: generate ALL raw amplitudes first, then fit PCA ONCE per antenna
        # pair on the full corpus and batch-transform — avoids 12 000 individual SVDs.

        # Uniform distribution: equal samples per class.
        # Class imbalance is handled exclusively by the loss weights (train_*.py),
        # NOT by over-sampling rare classes in the dataset itself.
        n_per = num_samples // 7
        counts = np.array([n_per] * 7)
        counts[-1] += num_samples - counts.sum()   # fix rounding

        labels: list[int] = []
        for cls, n in enumerate(counts):
            labels.extend([cls] * n)
        total = len(labels)

        # ── Step 1: generate raw amplitudes (N, seq_len, 456) ────────────────
        raws = np.empty((total, seq_len, RAW_FEATURES), dtype=np.float32)
        for k, cls in enumerate(labels):
            raw       = simulate_activity(cls, seq_len, seed=cls * 10000 + k)
            raws[k]   = np.clip(raw, AMP_MIN, AMP_MAX)

        # ── Step 2: fit PCA once per antenna pair, batch-transform ───────────
        # Reshape to (total*seq_len, 456) for fitting
        flat      = raws.reshape(-1, RAW_FEATURES)        # (N*T, 456)
        pca_flat  = np.empty((total * seq_len, PCA_FEATURES), dtype=np.float32)
        pca_obj   = decomposition.PCA(n_components=PCA_COMPONENTS)
        for i in range(ANTENNA_PAIRS):
            block              = flat[:, i * SUBCARRIERS:(i + 1) * SUBCARRIERS]
            pca_obj.fit(block)
            proj               = pca_obj.transform(block)          # (N*T, 3)
            pca_flat[:, i * PCA_COMPONENTS:(i + 1) * PCA_COMPONENTS] = proj

        pca_raws = pca_flat.reshape(total, seq_len, PCA_FEATURES)  # (N, T, 12)

        # ── Step 3: normalise + concatenate ──────────────────────────────────
        norm = (raws - AMP_MIN) / (AMP_MAX - AMP_MIN + 1e-8)      # (N, T, 456)
        # PCA outputs have different range: normalise to [-1,1] per component
        pca_min = pca_raws.reshape(-1, PCA_FEATURES).min(axis=0)
        pca_max = pca_raws.reshape(-1, PCA_FEATURES).max(axis=0)
        norm_pca = (pca_raws - pca_min) / (pca_max - pca_min + 1e-8)

        feats = np.concatenate([norm, norm_pca], axis=2)           # (N, T, 468)

        # ── Step 4: shuffle ───────────────────────────────────────────────────
        idx           = np.random.permutation(total)
        self.samples  = [feats[i].astype(np.float32) for i in idx]
        self.labels   = [labels[i] for i in idx]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.samples[idx])           # (seq_len, 468)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


# ── Public API ────────────────────────────────────────────────────────────────

def get_dataloader(batch_size: int = 16, num_samples: int = 3000,
                   val_split: float = 0.2, seq_len: int = 128):
    """Returns (train_loader, val_loader)."""
    ds       = SimulatedCSIDataset(num_samples=num_samples, seq_len=seq_len)
    val_n    = int(len(ds) * val_split)
    train_n  = len(ds) - val_n
    train_ds, val_ds = random_split(ds, [train_n, val_n],
                                    generator=torch.Generator().manual_seed(42))
    kw = dict(batch_size=batch_size, num_workers=0, pin_memory=False)
    return (DataLoader(train_ds, shuffle=True,  **kw),
            DataLoader(val_ds,   shuffle=False, **kw))


def get_class_weights() -> torch.Tensor:
    """Inverse-frequency class weights (same as repo train_lstm.py)."""
    w = 1.0 / CLASS_PROPORTIONS
    return torch.tensor(w / w.sum() * len(CLASS_PROPORTIONS), dtype=torch.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Public CSI Dataset Loaders
# ══════════════════════════════════════════════════════════════════════════════

import os
import glob
import scipy.ndimage
import warnings

# ── Catalog of known public CSI-HAR datasets ─────────────────────────────────
DATASET_CATALOG = {
    "Kovalenko-2021": {
        "description": "Dataset de referencia de este TFG. 7 actividades, hardware Atheros AR9380.",
        "activities":  ["Standing", "Walking", "Get Down", "Sitting", "Get Up", "Lying", "No Person"],
        "n_classes":   7,
        "hardware":    "TP-Link WDR3600 — Atheros AR9380 (802.11n, 5 GHz)",
        "subcarriers": 114,
        "antenna_cfg": "2Tx × 2Rx = 4 pares",
        "features":    456,
        "format":      "Carpeta con data.csv (filas=paquetes, cols=456 amplitudes) + label.csv",
        "url":         "https://ieee-dataport.org/open-access/human-activity-recognition-using-wifi-channel-state-information",
        "doi":         "10.21227/s7dp-2f35",
        "paper":       "Kovalenko et al. (2021). IEEE DataPort.",
        "size":        "~9 GB",
        "loader":      "repo_folder",
        "n_subjects":  7,
        "sample_rate": "~20 Hz",
    },
    "UT-HAR": {
        "description": "University of Texas HAR. 7 actividades, dataset público en GitHub.",
        "activities":  ["bed", "fall", "pickup", "run", "sitdown", "standup", "walk"],
        "n_classes":   7,
        "hardware":    "Intel 5300 NIC (802.11n, 5 GHz)",
        "subcarriers": 30,
        "antenna_cfg": "1Tx × 3Rx = 3 pares",
        "features":    90,
        "format":      "NPY por actividad (N,250,90) o CSV con cols [time,amp×90,phase×90]",
        "url":         "https://github.com/ermongroup/Wifi_Activity_Recognition",
        "doi":         "",
        "paper":       "Yousefi et al. (2017). IEEE Communications Magazine.",
        "size":        "~200 MB",
        "loader":      "UTHARDataset",
        "n_subjects":  10,
        "sample_rate": "1000 Hz → ventanas de 1000 samples (250 tras decimate×4)",
    },
    "NTU-Fi-HAR": {
        "description": "NTU-Fi HAR Benchmark. 6 actividades, 20 sujetos, Intel 5300.",
        "activities":  ["box", "circle", "clean", "fall", "run", "walk"],
        "n_classes":   6,
        "hardware":    "Intel 5300 NIC (802.11n)",
        "subcarriers": 114,
        "antenna_cfg": "1Tx × 3Rx = 3 pares",
        "features":    342,
        "format":      "MAT por sujeto/actividad. Clave 'csi_data' shape (3,114,T) o (T,342)",
        "url":         "https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark",
        "doi":         "",
        "paper":       "Chen et al. (2022). WiFi-CSI-Sensing-Benchmark.",
        "size":        "~1 GB",
        "loader":      "NTUFiDataset",
        "n_subjects":  20,
        "sample_rate": "~1000 Hz",
    },
    "MM-Fi": {
        "description": "Multi-Modal Fi. 27 actividades de rehabilitación, Intel AX200 (WiFi-6).",
        "activities":  ["27 ejercicios de rehabilitación (ver paper)"],
        "n_classes":   27,
        "hardware":    "Intel AX200 (802.11ax, WiFi-6)",
        "subcarriers": 114,
        "antenna_cfg": "1Tx × 3Rx = 3 pares",
        "features":    342,
        "format":      "MAT por sujeto/protocolo/actividad. Clave 'csi' shape (T,3,114)",
        "url":         "https://github.com/ybhbingo/MMFi_dataset",
        "doi":         "10.48550/arXiv.2305.10345",
        "paper":       "Yang et al. (2023). MM-Fi. NeurIPS Datasets & Benchmarks.",
        "size":        "~50 GB",
        "loader":      "MMFiDataset",
        "n_subjects":  40,
        "sample_rate": "~100 Hz (WiFi-6 CSI)",
    },
    "SignFi": {
        "description": "Sign language via Wi-Fi CSI. 276 gestos ASL + 20 gestos domésticos.",
        "activities":  ["276 signos ASL + 20 gestos domésticos"],
        "n_classes":   296,
        "hardware":    "Intel 5300 NIC (802.11n)",
        "subcarriers": 30,
        "antenna_cfg": "1Tx × 3Rx = 3 pares",
        "features":    90,
        "format":      "MAT con 'CSI_data_all' (200,30,3,N) y 'label_all' (N,)",
        "url":         "https://github.com/yongsen/SignFi",
        "doi":         "10.1145/3311823",
        "paper":       "Ha et al. (2019). ACM IMWUT / UbiComp.",
        "size":        "~500 MB",
        "loader":      "SignFiDataset",
        "n_subjects":  5,
        "sample_rate": "~1000 Hz → 200 muestras/gesto",
    },
    "WiAR": {
        "description": "Wireless Activity Recognition. 16 actividades a 2 distancias, Intel 5300.",
        "activities":  ["walk","run","jump","wave","clap","wipe","sitdown","standup",
                        "fall","boxing","squat","pickup","eat","drink","phone","liedown"],
        "n_classes":   16,
        "hardware":    "Intel 5300 NIC (802.11n)",
        "subcarriers": 30,
        "antenna_cfg": "1Tx × 3Rx = 3 pares",
        "features":    90,
        "format":      "CSV por actividad/distancia. Cols = amplitudes de subportadora",
        "url":         "https://github.com/linteresa/WiAR",
        "doi":         "10.1109/ACCESS.2019.2930878",
        "paper":       "Guo et al. (2019). WiAR. IEEE Access.",
        "size":        "~300 MB",
        "loader":      "WiARDataset",
        "n_subjects":  16,
        "sample_rate": "~1000 Hz",
    },
    "ARIL": {
        "description": "Activity Recognition with Indoor Localization. 6 actividades, 16 posiciones antena.",
        "activities":  ["walk", "run", "fall", "boxing", "circles", "clean"],
        "n_classes":   6,
        "hardware":    "Intel 5300 NIC (802.11n)",
        "subcarriers": 30,
        "antenna_cfg": "1Tx × 3Rx = 3 pares",
        "features":    90,
        "format":      "CSV con amplitudes CSI. Una fila por muestra.",
        "url":         "https://github.com/Restuccia-Group/ARIL",
        "doi":         "",
        "paper":       "Restuccia et al. (2019). IEEE INFOCOM.",
        "size":        "~150 MB",
        "loader":      "ARILDataset",
        "n_subjects":  6,
        "sample_rate": "~1000 Hz",
    },
}

# ── Universal interpolation utility ──────────────────────────────────────────

def universal_csi_interpolate(raw_csi: np.ndarray,
                               target_seq_len: int = 128,
                               target_features: int = RAW_FEATURES) -> np.ndarray:
    """
    Bilinear resize of any (T, F) CSI matrix to (target_seq_len, target_features).
    Allows loading datasets with different hardware (e.g. Intel 5300: 90 features)
    into the 456-feature pipeline without losing temporal/spectral structure.
    """
    T, F = raw_csi.shape
    if T == 0 or F == 0:
        return np.zeros((target_seq_len, target_features), dtype=np.float32)
    if T == target_seq_len and F == target_features:
        return raw_csi.astype(np.float32)
    zoom_t = target_seq_len / T
    zoom_f = target_features / F
    out = scipy.ndimage.zoom(raw_csi.astype(np.float64), (zoom_t, zoom_f), order=1)
    return np.clip(out, 0, None).astype(np.float32)


def _window_sequence(arr: np.ndarray, seq_len: int, stride: int = None) -> list:
    """Slice a (T, F) array into overlapping windows of length seq_len."""
    if stride is None:
        stride = seq_len
    T = arr.shape[0]
    windows = []
    for start in range(0, T - seq_len + 1, stride):
        windows.append(arr[start:start + seq_len])
    return windows


def _load_and_preprocess(raw: np.ndarray, seq_len: int,
                          label: int) -> tuple[list, list]:
    """
    Common finalisation: interpolate to 456 features, window, full_preprocess.
    raw : (T, F)  — any F
    Returns (list of (seq_len,468) arrays, list of int labels).
    """
    if raw.ndim == 1:
        raw = raw[:, None]
    T, F = raw.shape

    if F != RAW_FEATURES:
        raw = universal_csi_interpolate(raw, target_seq_len=T, target_features=RAW_FEATURES)

    windows = _window_sequence(raw, seq_len)
    if not windows:
        # If total sequence shorter than seq_len, pad & use whole thing
        pad = np.zeros((seq_len, RAW_FEATURES), dtype=np.float32)
        pad[:T] = raw[:min(T, seq_len)]
        windows = [pad]

    samples, labels = [], []
    for w in windows:
        try:
            samples.append(full_preprocess(w.astype(np.float32)))
            labels.append(label)
        except Exception:
            pass
    return samples, labels


# ── UT-HAR ────────────────────────────────────────────────────────────────────

class UTHARDataset(Dataset):
    """
    UT-HAR (Yousefi et al. 2017) — Intel 5300, 30 subcarriers × 3 antennas = 90 features.

    Supported layouts
    -----------------
    A) NPY layout (ermongroup repo):
       data_dir/{activity_name}.npy   shape (N, 250, 90)
       Activity names (folders or file stems) map to labels 0-6.

    B) CSV layout (per-activity folders):
       data_dir/{activity_name}/*.csv   cols: [time?, amp×90, phase×90] (≥90 cols)
       Each CSV = one continuous recording. Windowed into seq_len chunks.
    """
    ACTIVITY_MAP = {
        "bed": 0, "fall": 1, "pickup": 2, "run": 3,
        "sitdown": 4, "sit_down": 4, "sit-down": 4,
        "standup": 5, "stand_up": 5, "stand-up": 5,
        "walk": 6,
    }

    def __init__(self, data_dir: str, seq_len: int = 128):
        self.samples: list[np.ndarray] = []
        self.labels:  list[int]        = []
        self.seq_len = seq_len

        # ── NPY layout ────────────────────────────────────────────────────────
        npy_files = glob.glob(os.path.join(data_dir, "*.npy"))
        for fpath in sorted(npy_files):
            stem = os.path.splitext(os.path.basename(fpath))[0].lower()
            label = self.ACTIVITY_MAP.get(stem, -1)
            try:
                arr = np.load(fpath, allow_pickle=True)        # (N,250,90) or (250,90)
                if arr.ndim == 2:
                    arr = arr[None]                             # (1,250,90)
                for i in range(arr.shape[0]):
                    window = arr[i]                             # (250,90)
                    interp = universal_csi_interpolate(
                        window, target_seq_len=seq_len, target_features=RAW_FEATURES)
                    try:
                        self.samples.append(full_preprocess(interp))
                        self.labels.append(max(label, 0))
                    except Exception:
                        pass
            except Exception as e:
                warnings.warn(f"UT-HAR NPY load error ({fpath}): {e}")

        # ── CSV layout ────────────────────────────────────────────────────────
        if not self.samples:
            for act_name, label in self.ACTIVITY_MAP.items():
                act_dir = os.path.join(data_dir, act_name)
                if not os.path.isdir(act_dir):
                    continue
                for fpath in sorted(glob.glob(os.path.join(act_dir, "*.csv"))):
                    try:
                        arr = np.loadtxt(fpath, delimiter=",", dtype=np.float32)
                        # Columns may be [time, amp×90, phase×90] → take amp cols
                        if arr.ndim == 1:
                            arr = arr[None]
                        _, ncols = arr.shape
                        if ncols >= 181:          # time + 90 amp + 90 phase
                            raw = arr[:, 1:91]    # amplitude only
                        elif ncols >= 90:
                            raw = arr[:, :90]
                        else:
                            raw = arr
                        samps, lbls = _load_and_preprocess(raw, seq_len, label)
                        self.samples.extend(samps)
                        self.labels.extend(lbls)
                    except Exception as e:
                        warnings.warn(f"UT-HAR CSV error ({fpath}): {e}")

    def __len__(self):  return len(self.samples)
    def __getitem__(self, idx):
        return (torch.from_numpy(self.samples[idx]),
                torch.tensor(self.labels[idx], dtype=torch.long))


# ── NTU-Fi HAR ────────────────────────────────────────────────────────────────

class NTUFiDataset(Dataset):
    """
    NTU-Fi HAR (Chen et al. 2022) — Intel 5300, 114 subcarriers × 3 antennas = 342 features.

    Folder layout
    -------------
    data_dir/{activity_name}/{subject_id}/*.mat
        MAT key 'csi_data':  shape (3, 114, T)  or  (T, 342)
    Activity folders: box, circle, clean, fall, run, walk → labels 0–5.
    """
    ACTIVITY_MAP = {
        "box": 0, "boxing": 0,
        "circle": 1, "circles": 1,
        "clean": 2, "cleaning": 2,
        "fall": 3, "falling": 3,
        "run": 4, "running": 4,
        "walk": 5, "walking": 5,
    }

    def __init__(self, data_dir: str, seq_len: int = 128):
        self.samples: list[np.ndarray] = []
        self.labels:  list[int]        = []
        self.seq_len = seq_len

        try:
            import scipy.io as sio
        except ImportError:
            warnings.warn("scipy not available — NTUFiDataset cannot load MAT files.")
            return

        mat_files = (
            glob.glob(os.path.join(data_dir, "**", "*.mat"), recursive=True) +
            glob.glob(os.path.join(data_dir, "*.mat"))
        )
        for fpath in sorted(set(mat_files)):
            # Infer label from path components
            parts = fpath.replace("\\", "/").split("/")
            label = -1
            for part in parts:
                lname = part.lower().rstrip("s")  # "falls"->"fall"
                if lname in self.ACTIVITY_MAP:
                    label = self.ACTIVITY_MAP[lname]
                    break
                if part.lower() in self.ACTIVITY_MAP:
                    label = self.ACTIVITY_MAP[part.lower()]
                    break

            try:
                mat  = sio.loadmat(fpath)
                # Try common key names
                raw  = None
                for key in ("csi_data", "csi", "data", "CSI_data"):
                    if key in mat:
                        raw = mat[key]
                        break
                if raw is None:
                    continue

                # Normalise shape → (T, F)
                raw = np.squeeze(raw)
                if raw.ndim == 3:
                    # (3,114,T) → (T, 342)
                    if raw.shape[0] == 3:
                        raw = raw.transpose(2, 0, 1).reshape(-1, raw.shape[0] * raw.shape[1])
                    else:
                        # (T,3,114)
                        raw = raw.reshape(raw.shape[0], -1)
                if raw.ndim == 1:
                    raw = raw[:, None]

                samps, lbls = _load_and_preprocess(
                    raw.astype(np.float32), seq_len, max(label, 0))
                self.samples.extend(samps)
                self.labels.extend(lbls)
            except Exception as e:
                warnings.warn(f"NTU-Fi MAT error ({fpath}): {e}")

    def __len__(self):  return len(self.samples)
    def __getitem__(self, idx):
        return (torch.from_numpy(self.samples[idx]),
                torch.tensor(self.labels[idx], dtype=torch.long))


# ── MM-Fi ─────────────────────────────────────────────────────────────────────

class MMFiDataset(Dataset):
    """
    MM-Fi (Yang et al. 2023) — Intel AX200, 114 subcarriers × 3 antennas = 342 features.

    Folder layout
    -------------
    data_dir/{protocol}/{subject}/{activity}/wifi_csi.mat
        MAT key 'csi':  shape (T, 3, 114) or (3, 114, T)
    Activity folder names encode labels (E01–E27 or descriptive names).
    """
    def __init__(self, data_dir: str, seq_len: int = 128):
        self.samples: list[np.ndarray] = []
        self.labels:  list[int]        = []
        self.seq_len = seq_len
        self._activity_to_label: dict[str, int] = {}

        try:
            import scipy.io as sio
        except ImportError:
            warnings.warn("scipy required for MMFiDataset.")
            return

        mat_files = (
            glob.glob(os.path.join(data_dir, "**", "wifi_csi.mat"), recursive=True) +
            glob.glob(os.path.join(data_dir, "**", "*.mat"), recursive=True)
        )
        for fpath in sorted(set(mat_files)):
            # Infer label from activity folder (E01, E02, … or descriptive)
            parts = fpath.replace("\\", "/").split("/")
            activity = "unknown"
            for part in parts[:-1]:
                if part.startswith("E") and part[1:].isdigit():
                    activity = part
                    break
                if len(part) > 2 and part not in ("wifi_csi",):
                    activity = part
            if activity not in self._activity_to_label:
                self._activity_to_label[activity] = len(self._activity_to_label)
            label = self._activity_to_label[activity]

            try:
                mat = sio.loadmat(fpath)
                raw = None
                for key in ("csi", "csi_data", "data"):
                    if key in mat:
                        raw = mat[key]
                        break
                if raw is None:
                    continue

                raw = np.squeeze(raw).astype(np.float32)
                if raw.ndim == 3:
                    if raw.shape[0] == 3:           # (3,114,T)
                        raw = raw.transpose(2, 0, 1).reshape(-1, 3 * 114)
                    elif raw.shape[1] == 3:          # (T,3,114)
                        raw = raw.reshape(raw.shape[0], -1)
                    elif raw.shape[2] == 3:          # (T,114,3) uncommon
                        raw = raw.reshape(raw.shape[0], -1)

                samps, lbls = _load_and_preprocess(raw, seq_len, label)
                self.samples.extend(samps)
                self.labels.extend(lbls)
            except Exception as e:
                warnings.warn(f"MM-Fi MAT error ({fpath}): {e}")

    def __len__(self):  return len(self.samples)
    def __getitem__(self, idx):
        return (torch.from_numpy(self.samples[idx]),
                torch.tensor(self.labels[idx], dtype=torch.long))


# ── SignFi ────────────────────────────────────────────────────────────────────

class SignFiDataset(Dataset):
    """
    SignFi (Ha et al. 2019) — Intel 5300, 30 subcarriers × 3 antennas = 90 features.
    276 ASL signs + 20 home gestures.

    Expected MAT file keys
    ----------------------
    'CSI_data_all' : (200, 30, 3, N_gestures)   — amplitude × time × ant × gesture
    'label_all'    : (N_gestures,)
    Also accepts per-gesture structure if each .mat has a single sample.
    """
    def __init__(self, data_dir: str, seq_len: int = 128):
        self.samples: list[np.ndarray] = []
        self.labels:  list[int]        = []
        self.seq_len = seq_len

        try:
            import scipy.io as sio
        except ImportError:
            warnings.warn("scipy required for SignFiDataset.")
            return

        mat_files = (
            glob.glob(os.path.join(data_dir, "*.mat")) +
            glob.glob(os.path.join(data_dir, "**", "*.mat"), recursive=True)
        )
        for fpath in sorted(set(mat_files)):
            try:
                mat = sio.loadmat(fpath)

                # ── Full-dataset MAT layout ───────────────────────────────────
                if "CSI_data_all" in mat:
                    csi  = mat["CSI_data_all"]   # (200, 30, 3, N)
                    lbls = mat.get("label_all", None)
                    # Permute to (N, 200, 90): time × (subcarriers × antennas)
                    csi  = csi.transpose(3, 0, 1, 2)      # (N, 200, 30, 3)
                    N    = csi.shape[0]
                    csi  = csi.reshape(N, 200, 90)
                    for i in range(N):
                        raw   = csi[i].astype(np.float32)  # (200, 90)
                        interp = universal_csi_interpolate(
                            raw, target_seq_len=seq_len, target_features=RAW_FEATURES)
                        try:
                            self.samples.append(full_preprocess(interp))
                            lbl = int(lbls.flat[i]) if lbls is not None else i % 276
                            self.labels.append(lbl)
                        except Exception:
                            pass
                    continue

                # ── Per-gesture MAT layout ────────────────────────────────────
                raw = None
                for key in ("csi", "data", "CSI"):
                    if key in mat:
                        raw = mat[key]
                        break
                if raw is None:
                    continue
                raw = np.squeeze(raw).astype(np.float32)
                if raw.ndim == 3:                    # (200,30,3) → (200,90)
                    raw = raw.reshape(raw.shape[0], -1)
                interp = universal_csi_interpolate(
                    raw, target_seq_len=seq_len, target_features=RAW_FEATURES)
                try:
                    self.samples.append(full_preprocess(interp))
                    self.labels.append(0)
                except Exception:
                    pass
            except Exception as e:
                warnings.warn(f"SignFi MAT error ({fpath}): {e}")

    def __len__(self):  return len(self.samples)
    def __getitem__(self, idx):
        return (torch.from_numpy(self.samples[idx]),
                torch.tensor(self.labels[idx], dtype=torch.long))


# ── WiAR ─────────────────────────────────────────────────────────────────────

class WiARDataset(Dataset):
    """
    WiAR (Guo et al. 2019) — Intel 5300, 30 subcarriers × 3 antennas = 90 features.
    16 activities recorded at 2 distances.

    Folder layout
    -------------
    data_dir/{distance_tag}/{activity_name}/*.csv
    or data_dir/{activity_name}/*.csv
    Each CSV row = one packet (90 amplitude values).
    """
    ACTIVITY_MAP = {
        "walk": 0, "run": 1, "jump": 2, "wave": 3, "clap": 4,
        "wipe": 5, "sitdown": 6, "sit_down": 6, "sit-down": 6,
        "standup": 7, "stand_up": 7, "stand-up": 7,
        "fall": 8, "boxing": 9, "box": 9,
        "squat": 10, "pickup": 11, "pick_up": 11,
        "eat": 12, "drink": 13, "phone": 14, "liedown": 15, "lie_down": 15,
    }

    def __init__(self, data_dir: str, seq_len: int = 128):
        self.samples: list[np.ndarray] = []
        self.labels:  list[int]        = []
        self.seq_len = seq_len

        all_csvs = glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)
        for fpath in sorted(all_csvs):
            parts = fpath.replace("\\", "/").lower().split("/")
            label = -1
            for part in parts:
                stem = part.replace("-", "_").split(".")[0]
                if stem in self.ACTIVITY_MAP:
                    label = self.ACTIVITY_MAP[stem]
                    break
            try:
                arr = np.loadtxt(fpath, delimiter=",", dtype=np.float32)
                if arr.ndim == 1:
                    arr = arr[None]
                # Take first 90 columns (amplitudes)
                raw = arr[:, :90]
                samps, lbls = _load_and_preprocess(raw, seq_len, max(label, 0))
                self.samples.extend(samps)
                self.labels.extend(lbls)
            except Exception as e:
                warnings.warn(f"WiAR CSV error ({fpath}): {e}")

    def __len__(self):  return len(self.samples)
    def __getitem__(self, idx):
        return (torch.from_numpy(self.samples[idx]),
                torch.tensor(self.labels[idx], dtype=torch.long))


# ── ARIL ─────────────────────────────────────────────────────────────────────

class ARILDataset(Dataset):
    """
    ARIL (Restuccia et al. 2019) — Intel 5300, 30 subcarriers × 3 antennas = 90 features.
    6 activities: walk, run, fall, boxing, circles, clean.

    Folder layout
    -------------
    data_dir/{activity_name}/*.csv    or data_dir/*.csv (label in filename/column)
    Each CSV row = one sample (amplitude values) or flattened windows.
    """
    ACTIVITY_MAP = {
        "walk": 0, "run": 1, "fall": 2, "boxing": 3, "box": 3,
        "circle": 4, "circles": 4, "clean": 5, "cleaning": 5,
    }

    def __init__(self, data_dir: str, seq_len: int = 128):
        self.samples: list[np.ndarray] = []
        self.labels:  list[int]        = []
        self.seq_len = seq_len

        all_csvs = glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)
        for fpath in sorted(all_csvs):
            parts = fpath.replace("\\", "/").lower().split("/")
            label = -1
            for part in parts:
                stem = part.replace("-", "_").split(".")[0]
                if stem in self.ACTIVITY_MAP:
                    label = self.ACTIVITY_MAP[stem]
                    break
            try:
                arr = np.loadtxt(fpath, delimiter=",", dtype=np.float32)
                if arr.ndim == 1:
                    arr = arr[None]
                if arr.shape[1] > 90:          # last col may be label
                    raw = arr[:, :90]
                else:
                    raw = arr[:, :90]
                samps, lbls = _load_and_preprocess(raw, seq_len, max(label, 0))
                self.samples.extend(samps)
                self.labels.extend(lbls)
            except Exception as e:
                warnings.warn(f"ARIL CSV error ({fpath}): {e}")

    def __len__(self):  return len(self.samples)
    def __getitem__(self, idx):
        return (torch.from_numpy(self.samples[idx]),
                torch.tensor(self.labels[idx], dtype=torch.long))


# ── Auto-detect loader ────────────────────────────────────────────────────────

def autodetect_and_load(data_dir: str, seq_len: int = 128) -> tuple:
    """
    Attempt to detect which public dataset format is in data_dir and load it.
    Returns (Dataset, dataset_name_str) or raises ValueError if unrecognised.
    """
    files = list(glob.iglob(os.path.join(data_dir, "**", "*"), recursive=True))
    exts  = {os.path.splitext(f)[1].lower() for f in files}

    has_mat = ".mat" in exts
    has_npy = ".npy" in exts
    has_csv = ".csv" in exts

    dirs = {os.path.basename(d).lower()
            for d in glob.glob(os.path.join(data_dir, "*")) if os.path.isdir(d)}

    # Heuristics
    if has_mat and any(k in dirs for k in ("box", "circle", "fall", "run", "walk")):
        return NTUFiDataset(data_dir, seq_len), "NTU-Fi-HAR"

    if has_mat and "csi_data_all" in str(files).lower():
        return SignFiDataset(data_dir, seq_len), "SignFi"

    if has_mat:
        return MMFiDataset(data_dir, seq_len), "MM-Fi"

    if has_npy and any(k in dirs for k in ("bed", "fall", "walk", "run")):
        return UTHARDataset(data_dir, seq_len), "UT-HAR"

    if has_csv and any(k in dirs for k in ("bed", "fall", "walk", "sitdown", "standup")):
        return UTHARDataset(data_dir, seq_len), "UT-HAR"

    if has_csv and any(k in dirs for k in ("wave", "clap", "squat", "pickup")):
        return WiARDataset(data_dir, seq_len), "WiAR"

    if has_csv and os.path.exists(os.path.join(data_dir, "data.csv")):
        raise ValueError("Use 'Load Repo Folder' for Kovalenko-2021 format (data.csv + label.csv)")

    if has_csv:
        return ARILDataset(data_dir, seq_len), "ARIL"

    raise ValueError(f"Cannot detect dataset format in: {data_dir}")
