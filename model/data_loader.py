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
