"""
position_estimator.py — CSI-Based Zone Classifier
====================================================

SCIENTIFIC HONESTY STATEMENT
------------------------------
This module implements ZONE CLASSIFICATION (coarse localization), NOT exact
position estimation. The distinction is critical:

  * Exact localization (e.g., x,y coordinates in cm) requires a prior
    measurement campaign with ground-truth positions and typically
    achieves 0.5–2m accuracy in the best-case (FILA, SpotFi, etc.).

  * Zone classification divides the environment into discrete regions
    and assigns the person to the most likely region based on signal
    amplitude statistics. This is simpler, more robust, and honest
    about what CSI amplitude alone can tell us.

PHYSICAL BASIS
--------------
The received signal amplitude follows the indoor path loss model
(ITU-R P.1238):

    PL(d) [dB] = 20·log10(f) + N·log10(d) + L_f(n) - 28

Equivalently, the linear amplitude (|H| in CSI) decays as:

    |H(d)| ≈ A_0 · (d_0 / d)^(n/2)

where:
  d    = transmitter–receiver distance [m]
  A_0  = reference amplitude at d_0 = 1 m  (≈ 350 for TP-Link WDR3600)
  n    = path loss exponent (indoor: typically 2.5–3.5)
  L_f  = floor attenuation factor (not applicable, single floor)

With the TP-Link WDR3600 and Atheros CSI Tool (source repo hardware),
the amplitude range is [0, 577.66] (AMP_MAX from dataset.py).

ZONES (calibrated to path loss model with n=2.8, A_0=380)
----------------------------------------------------------
  Zone 0 — PROXIMITY  (0.0–1.5 m):  |H| mean ≈ 290–500
  Zone 1 — NEAR       (1.5–3.0 m):  |H| mean ≈ 120–290
  Zone 2 — MID-RANGE  (3.0–5.0 m):  |H| mean ≈ 45–120
  Zone 3 — FAR        (5.0–8.0 m):  |H| mean ≈ 15–45

LIMITATIONS (mandatory disclosure for scientific work)
------------------------------------------------------
  1. Without a real measurement campaign, zone boundaries are
     derived from the theoretical path loss model only.
  2. Multipath, NLOS conditions, and furniture cause deviations
     of 1–3 dB from the free-space model.
  3. CSI amplitude alone cannot disambiguate positions with the
     same distance but different angles (angular ambiguity).
  4. The training data is SIMULATED. Real deployment requires
     calibration measurements at known positions.
  5. Temporal averaging is essential; instantaneous estimates
     are unreliable due to small-scale fading.

FEATURE VECTOR (16 dimensions, one set per antenna pair)
---------------------------------------------------------
  For each of the 4 antenna pairs (subcarriers: 0:114, 114:228, ...):
    mean     — average amplitude across 114 subcarriers and T timesteps
    std      — standard deviation (variance proxy)
    p90-p10  — 80th-percentile range (robust spread)
    energy   — sum of squared amplitudes (signal power)

  Concatenated → 4 × 4 = 16 features per window.

REFERENCES
----------
  [1] Chapre et al. (2014). CSI-MIMO for indoor localisation.
      Proc. 39th IEEE LCN, pp. 1–7.
  [2] Wu et al. (2012). FILA: Fine-grained indoor localisation.
      IEEE INFOCOM, pp. 2210–2218.
  [3] Wang et al. (2017). E-eyes: Device-free location-oriented
      activity recognition. MobiCom, pp. 617–628.
  [4] ITU-R P.1238-10 (2019). Propagation data and prediction methods
      for the planning of indoor radiocommunication systems.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.data_loader import SUBCARRIERS, ANTENNA_PAIRS, SAMPLE_RATE

# ── Zone definitions ──────────────────────────────────────────────────────────
ZONE_NAMES  = ["Proximity", "Near", "Mid-range", "Far"]
ZONE_COLORS = ["#00c853", "#ffab00", "#ff6d00", "#b71c1c"]
ZONE_RANGES = [(0.0, 1.5), (1.5, 3.0), (3.0, 5.0), (5.0, 8.0)]   # metres
NUM_ZONES   = len(ZONE_NAMES)

# Path loss model parameters (ITU-R P.1238, indoor office/residential)
_A0 = 380.0    # reference amplitude at d0 = 1 m  (calibrated to AMP_MAX=577)
_N  = 2.8      # path loss exponent (indoor, 5 GHz)
_D0 = 1.0      # reference distance [m]


def _amplitude_at_distance(d: float) -> float:
    """Mean expected CSI amplitude at distance d [m] (path loss model)."""
    return _A0 * (_D0 / max(d, 0.01)) ** (_N / 2.0)


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_zone_features(amplitudes: np.ndarray) -> np.ndarray:
    """
    Compute the 16-dim feature vector used by the zone classifier.

    Parameters
    ----------
    amplitudes : (T, 456)  raw amplitude window (not normalised)

    Returns
    -------
    features : (16,)  float32
    """
    feats = []
    for i in range(ANTENNA_PAIRS):
        block = amplitudes[:, i * SUBCARRIERS:(i + 1) * SUBCARRIERS]  # (T, 114)
        flat  = block.ravel()
        feats.append(float(flat.mean()))
        feats.append(float(flat.std()))
        p10, p90 = np.percentile(flat, [10, 90])
        feats.append(float(p90 - p10))
        feats.append(float(np.sum(flat ** 2) / flat.size))   # mean energy
    return np.array(feats, dtype=np.float32)


# ── Simulator for zone training data ─────────────────────────────────────────

def simulate_zone(zone_idx: int, seq_len: int = 128,
                  seed: int = None) -> np.ndarray:
    """
    Generate a synthetic amplitude window for training zone classifier.

    The signal model follows the ITU-R P.1238 path loss equation.
    Each zone maps to a distance range; we sample d uniformly within
    the zone and compute expected amplitude + shadowing noise.

    Parameters
    ----------
    zone_idx : int      zone label (0–3)
    seq_len  : int      window length [samples]
    seed     : int      RNG seed for reproducibility

    Returns
    -------
    amplitudes : (seq_len, 456)  synthetic amplitude window
    """
    rng = np.random.default_rng(seed)
    d_lo, d_hi = ZONE_RANGES[zone_idx]
    d = rng.uniform(d_lo if d_lo > 0 else 0.3, d_hi)

    # Expected mean amplitude from path loss model
    mu = _amplitude_at_distance(d)

    # Shadowing standard deviation (log-normal, sigma_dB = 4–7 dB at 5 GHz)
    sigma_dB = rng.uniform(4.0, 7.0)
    sigma    = mu * (10 ** (sigma_dB / 20.0) - 1.0)   # linear equivalent

    # Small-scale fading: Rician for LOS (K-factor 3–8 dB), Rayleigh for NLOS
    K = rng.uniform(1.5, 5.0)
    phase = rng.uniform(0, 2 * np.pi, (ANTENNA_PAIRS * SUBCARRIERS,))

    t = np.arange(seq_len) / SAMPLE_RATE
    out = np.zeros((seq_len, ANTENNA_PAIRS * SUBCARRIERS), dtype=np.float32)

    for f in range(ANTENNA_PAIRS * SUBCARRIERS):
        # Rician envelope: LOS component + multipath scatter
        s_los = np.sqrt(K / (K + 1)) * mu
        s_sc  = np.sqrt(1 / (K + 1)) * sigma
        los   = s_los * np.cos(2 * np.pi * 0.5 * t + phase[f])
        sc    = s_sc  * rng.standard_normal(seq_len)
        out[:, f] = np.clip(mu + los + sc, 0, 577.66)

    return out


# ── Zone classifier (MLP) ─────────────────────────────────────────────────────

class ZoneClassifier(nn.Module):
    """
    Lightweight MLP mapping 16 amplitude statistics → zone label.

    Architecture: 16 → 64 → 32 → 4
    Input features are standardised before training.

    This is intentionally simple: the zone is determined primarily by
    signal strength (a monotonic function of distance), so a deep
    network is not warranted.
    """
    def __init__(self, in_features: int = 16, num_zones: int = NUM_ZONES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_zones),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @torch.no_grad()
    def predict(self, features: np.ndarray,
                mean: np.ndarray, std: np.ndarray) -> tuple[int, np.ndarray]:
        """
        Parameters
        ----------
        features : (16,)    raw features from extract_zone_features()
        mean, std : (16,)   normalisation statistics from training

        Returns
        -------
        zone_idx : int       predicted zone (0–3)
        probs    : (4,)      softmax probabilities
        """
        x = torch.tensor((features - mean) / (std + 1e-8),
                          dtype=torch.float32).unsqueeze(0)
        self.eval()
        logits = self(x)
        probs  = F.softmax(logits, dim=1).squeeze(0).numpy()
        return int(np.argmax(probs)), probs


# ── Dataset for zone training ──────────────────────────────────────────────────

class ZoneDataset(torch.utils.data.Dataset):
    def __init__(self, n_per_zone: int = 500, seq_len: int = 128):
        self.X: list[np.ndarray] = []
        self.y: list[int]        = []
        for z in range(NUM_ZONES):
            for i in range(n_per_zone):
                amp  = simulate_zone(z, seq_len, seed=z * 10000 + i)
                feat = extract_zone_features(amp)
                self.X.append(feat)
                self.y.append(z)

        self.X = np.stack(self.X)       # (N, 16)
        self.y = np.array(self.y)

        # Compute normalisation statistics on training data
        self.mean = self.X.mean(axis=0)
        self.std  = self.X.std(axis=0) + 1e-8
        self.X_norm = ((self.X - self.mean) / self.std).astype(np.float32)

        idx = np.random.permutation(len(self.y))
        self.X_norm = self.X_norm[idx]
        self.y      = self.y[idx]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X_norm[idx]), torch.tensor(self.y[idx], dtype=torch.long)
