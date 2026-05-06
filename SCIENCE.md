# Scientific Documentation — CSI-HAR System
## Wi-Fi Channel State Information for Human Activity Recognition and Zone Estimation

**TFG — Computer Science / Telecommunications Engineering**
**Author:** Mario Díaz Gómez
**Date:** April 2026

---

## Table of Contents
1. [Introduction](#1-introduction)
2. [Hardware and Dataset](#2-hardware-and-dataset)
3. [CSI Signal Model](#3-csi-signal-model)
4. [Feature Extraction Pipeline](#4-feature-extraction-pipeline)
5. [Activity Recognition Models](#5-activity-recognition-models)
6. [Zone-Based Position Estimation](#6-zone-based-position-estimation)
7. [Training Methodology](#7-training-methodology)
8. [Experimental Results](#8-experimental-results)
9. [System Architecture](#9-system-architecture)
10. [Limitations and Future Work](#10-limitations-and-future-work)
11. [References](#11-references)

---

## 1. Introduction

Wi-Fi-based Human Activity Recognition (HAR) is a passive sensing paradigm that
exploits the perturbations that a human body causes on wireless channel state
information (CSI) to infer physical activities — without cameras, wearables, or
any instrumentation on the subject.

The fundamental insight is that human movement displaces air and reflects
electromagnetic waves differently depending on body posture and motion.
A Wi-Fi transceiver measures these perturbations as complex channel coefficients
(the CSI matrix) at the subcarrier level of an OFDM signal.

### Why CSI over RSSI?
Received Signal Strength Indication (RSSI) is a single scalar, whereas CSI
provides a complex vector of length N_subcarriers × N_rx × N_tx. This rich
multi-dimensional measurement captures:
- Spatial diversity (multiple antenna pairs)
- Frequency diversity (per-subcarrier response)
- Temporal dynamics (changes due to movement)

RSSI cannot resolve fine-grained motion; CSI can discriminate between
activities as similar as sitting and standing [Wang et al., 2015].

---

## 2. Hardware and Dataset

### Physical Setup
| Component           | Specification |
|---------------------|---------------|
| Router              | TP-Link WDR3600 (Atheros AR9344) |
| CSI extraction tool | Atheros CSI Tool (modified firmware) |
| Band                | 5 GHz (802.11n, 40 MHz bandwidth) |
| Subcarriers         | **114** (after removing guard and DC subcarriers) |
| MIMO configuration  | 2×2 → **4 antenna pairs** (2 TX × 2 RX) |
| Sampling rate       | ≈ 20 Hz |
| Amplitude range     | [0, 577.66] (raw ADC units) |

### Dataset Structure (source repository)
The dataset was collected in three real indoor environments:

```
dataset/
├── bedroom_lviv/       (4 recording sessions)
├── parents_home/       (1 recording session)
└── vitalnia_lviv/      (5 recording sessions)
```

Each session contains:
- `data.csv`   — CSI amplitudes and phases for all subcarrier × antenna pairs
- `label.csv`  — ground-truth activity label per timestamp
- `images/`    — synchronized video frames (used for YOLO-based annotation)

**Total dataset size:** ~9 GB (not included in this repository due to size).

### Activity Classes (from `dataset.py → class_to_idx`)
| Index | Label      | Description |
|-------|------------|-------------|
| 0     | Standing   | Static upright posture |
| 1     | Walking    | Ambulation within the room |
| 2     | Get Down   | Transition standing → lying/sitting |
| 3     | Sitting    | Seated posture |
| 4     | Get Up     | Transition lying/sitting → standing |
| 5     | Lying      | Horizontal posture on bed/floor |
| 6     | No Person  | Empty room (reference state) |

### Class Distribution (from `train_lstm.py`)
```python
CLASS_PROPORTIONS = [0.113, 0.439, 0.0379, 0.1515, 0.0379, 0.1212, 0.1363]
#                   Stand  Walk  GetDn  Sit    GetUp  Lying  NoPer
```
Walking dominates (43.9%) due to long walking segments in recordings.
Transition activities (Get Up, Get Down) are the rarest (3.8% each).

---

## 3. CSI Signal Model

### OFDM Channel Matrix
For an 802.11n OFDM system with N_t transmitters, N_r receivers, and
N_sc subcarriers, the CSI at subcarrier k is modeled as:

```
Y_k = H_k · X_k + N_k
```

where:
- `H_k ∈ ℂ^{N_r × N_t}` — channel matrix at subcarrier k
- `X_k` — transmitted pilot symbol
- `N_k` — additive white Gaussian noise (AWGN)

The Atheros CSI Tool extracts the **complex channel coefficient** H_k for each
(transmitter, receiver, subcarrier) triplet. The amplitude is:

```
|H_k(t)| = A_k(t)
```

and the phase is:
```
∠H_k(t) = φ_k(t)
```

### Effect of Human Presence
A person located at position p introduces additional scattering paths.
The modified channel at time t becomes:

```
H_k(t) = H_static + Σ_i α_i(t) · e^{-j2πf_k τ_i(t)} · e^{j2πf_D,i t}
```

where:
- `H_static` — static multipath background (walls, furniture)
- `α_i(t)` — complex amplitude of the i-th human-induced scattering path
- `τ_i(t)` — time delay of path i (function of body position)
- `f_D,i` — Doppler shift due to body movement velocity

### Doppler Signatures per Activity
| Activity   | Doppler bandwidth | Amplitude variance | Notes |
|------------|------------------|--------------------|-------|
| No Person  | ~0 Hz            | Very low (< 3)     | Only thermal noise |
| Lying      | < 0.1 Hz         | Very low (< 5)     | Breathing ~0.3 Hz |
| Sitting    | 0.1–0.3 Hz       | Low (~8)           | Breathing + micro-motion |
| Standing   | 0.1–0.5 Hz       | Low–Medium (~15)   | Breathing ~0.3 Hz |
| Walking    | 0.5–5 Hz         | High (~40)         | Gait cycle ~2 Hz |
| Get Up     | Transient burst  | High (decaying)    | Burst at t=0 |
| Get Down   | Transient burst  | High (growing)     | Burst at t=T |

These signatures are the physical basis for the physics-inspired simulation
used when the real 9 GB dataset is unavailable.

---

## 4. Feature Extraction Pipeline

The pipeline replicates exactly `dataset.py` and `data_calibration.py` from
the source repository:

```
Raw CSI packets
      │
      ▼
[1] Amplitude extraction
    ├── 114 subcarriers × 4 antenna pairs = 456 values per packet
    └── Phase discarded (noisy at 5 GHz due to phase offset calibration issues)
      │
      ▼
[2] Hampel Filter  (per subcarrier, over time)
    ├── Window k = 7, threshold t₀ = 3σ, L = 1.4826 (Gaussian consistency)
    └── Replaces outliers with local median
      │
      ▼
[3] Wavelet Denoising  (per subcarrier, over time)
    ├── Wavelet: sym5 (Symlet-5), level = min(3, dwt_max_level)
    ├── Threshold: 0.06 × max|detail coefficients|
    └── Soft thresholding on detail coefficients, reconstruction
      │
      ▼
[4] PCA Dimensionality Reduction
    ├── Applied separately to each antenna pair (114 subcarriers)
    ├── 3 principal components retained per pair → 12 features total
    └── Captures dominant spatial correlation structure
      │
      ▼
[5] Feature Concatenation + Normalisation
    ├── [456 amplitudes | 12 PCA features] = 468 features per timestep
    └── Normalised to [0, 1] using AMP_MIN=0, AMP_MAX=577.66
      │
      ▼
[6] Windowing
    ├── Window size: 128 timesteps (real system: 1024, per repo train_lstm.py)
    └── Stride: variable (real-time: 1 packet; batch: 8 packets per repo)
      │
      ▼
Feature tensor: (batch, 128, 468)  → input to all models
```

### Implementation Note on Preprocessing Speed
The Hampel filter is O(N·k·F) where N=seq_len, k=7, F=456 subcarriers.
For N=128: 128 × 7 × 456 ≈ 408,960 comparisons per window.
For 3000 training windows: ~1.2 billion comparisons → ~400 seconds.

**Solution applied:** Hampel and wavelet denoising are applied to real
captured CSI in `full_preprocess()`. For training on simulated data (which
is already clean), they are skipped. PCA is batch-computed on the full
dataset corpus (fit once, transform all) reducing time from ~120s to ~8s.

---

## 5. Activity Recognition Models

Four architectures are trained and compared. All share:
- Input: `(batch, 128, 468)` normalised CSI windows
- Output: logits `(batch, 7)` → softmax → class probabilities
- Loss: CrossEntropyLoss with **inverse-frequency class weights**
- Optimizer: Adam (lr = 1.46 × 10⁻³, from repo `train_lstm.py`)
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=4)

### 5.1 SimpleLSTM
```
Input (B, 128, 468)
  └─ Linear(468→64, no bias) → LayerNorm(64)   ← CPU projection (7× speed-up)
       └─ LSTM(64→256, 2 layers, dropout=0.2)
            └─ Last hidden state (B, 256)
                 └─ LayerNorm(256) → Linear(256→7)
Output logits (B, 7)

Parameters: 888,455
Source: models/SimpleLSTMClassifier.py (repo) + input projection optimisation
```

**Rationale:** LSTMs model sequential dependencies via gated recurrent units.
The hidden state at time T summarises the entire window history, making them
effective for temporally ordered patterns like gait cycles.

**Input projection:** Direct 468-dim LSTM input takes ~250 ms/batch on CPU.
A `Linear(468→64)` projection reduces LSTM computation by 7×, bringing
batch time to ~35 ms while preserving temporal modelling capacity.

**Limitation:** Unidirectional — cannot use future context within the window.

### 5.2 BidirectionalLSTM
```
Input (B, 128, 468)
  └─ Linear(468→64, no bias) → LayerNorm(64)   ← CPU projection (7× speed-up)
       └─ BiLSTM(64→128×2, 2 layers, dropout=0.2)
            ├─ Forward hidden  (B, 128)
            └─ Backward hidden (B, 128)
                 concat → (B, 256)
                 └─ LayerNorm(256) → Linear(256→128) → ReLU
                      → Dropout(0.2) → Linear(128→7)
Output logits (B, 7)

Parameters: 658,311
Source: extended from models/LSTMClassifier.py (repo) + input projection
```

**Rationale:** Processes the window in both temporal directions. Beneficial
for transition activities (Get Up, Get Down) where the key signal pattern
spans the full window symmetrically.

### 5.3 FCN (Fully Convolutional Network)
```
Input (B, 128, 468) → transpose → (B, 468, 128)
  ├─ Conv1d(468→128, k=8) → BN → ReLU → Dropout(0.2)
  ├─ Conv1d(128→256, k=5) → BN → ReLU → Dropout(0.2)
  └─ Conv1d(256→128, k=3) → BN → ReLU → Dropout(0.2)
       └─ Global Average Pooling → (B, 128)
            └─ Linear(128→7)
Output logits (B, 7)

Parameters: 743,303
Source: models/FCNBaseline.py (repo) — two bugs fixed
```

**Bug fixes applied:**
1. **Dimension mismatch:** Original has `Linear(256, 7)` after a conv block
   that outputs 128 channels — runtime error. Fixed to `Linear(128, 7)`.
2. **Overfitting:** Original FCN has no regularisation. Without dropout,
   validation accuracy dropped to ~33% despite 97% training accuracy.
   Added `Dropout(0.2)` after each conv block.

**Rationale:** Convolutional networks efficiently capture local temporal
patterns (short motions) via shared filters. GAP removes dependency on
sequence length, enabling generalisation to variable-length windows.

**Reference:** Wang et al. (2017). Time Series Classification from Scratch
with Deep Neural Networks: A Strong Baseline. IJCNN 2017.

### 5.4 CSI Transformer (this work)
```
Input (B, 128, 468)
  └─ ConvStem: Conv1d(468→64,k=7) → BN → GELU → Conv1d(64→64,k=3) → BN → GELU
       └─ (B, 128, 64)
  └─ LearnablePositionalEncoding(64, max_len=512)
  └─ TransformerEncoder [2 layers, Pre-LN]:
       each layer: MultiHeadAttention(d=64, 4 heads) + FFN(128)
       └─ (B, 128, 64)
  └─ Aggregation: concat[mean, max] → (B, 128)
  └─ Linear(128→64) → LayerNorm → GELU → Dropout → Linear(64→7)
Output logits (B, 7)

Parameters: ~331K
```

**Design decisions:**
- **ConvStem** instead of linear projection: captures local patterns before
  global attention, reducing quadratic attention cost for long sequences.
- **Pre-LN** (norm_first=True): more stable gradient flow than post-LN.
- **Learnable positional encoding**: outperforms sinusoidal for short
  fixed-length windows (128 timesteps).
- **Mean+Max pooling**: richer aggregation than mean-only; max captures
  peak activity moments.

**Reference:** Vaswani et al. (2017). Attention Is All You Need. NeurIPS 2017.

### 5.5 Class Weighting Strategy
```python
CLASS_PROPORTIONS = [0.113, 0.439, 0.0379, 0.1515, 0.0379, 0.1212, 0.1363]
class_weight_i = 1 / proportion_i   (normalised)
```

Inverse-frequency weighting penalises errors on rare classes (Get Up,
Get Down at 3.79%) proportionally more than errors on common classes
(Walking at 43.9%). This prevents the model from ignoring rare but
important transition activities.

---

## 6. Zone-Based Position Estimation

### Epistemological Honesty
**What this system does:** classifies the person into one of 4 distance zones
from the router based on CSI amplitude statistics.

**What this system does NOT do:**
- Provide x,y coordinates (requires measurement campaigns)
- Work without calibration (zone boundaries are model-derived, not measured)
- Handle NLOS scenarios correctly (path loss model assumes LOS or light NLOS)

### Physical Basis: Path Loss Model
The mean CSI amplitude at distance d follows the ITU-R P.1238 indoor model:

```
|H(d)| ≈ A₀ · (d₀/d)^(n/2)
```

Parameters calibrated to TP-Link WDR3600 / 5 GHz band:
- A₀ = 380 (reference amplitude at d₀ = 1 m)
- n = 2.8 (path loss exponent, indoor residential/office)
- d₀ = 1 m (reference distance)

| Zone | Name       | Distance  | Expected |H| mean |
|------|------------|-----------|-----------------|
| 0    | Proximity  | 0–1.5 m   | ~290–500        |
| 1    | Near       | 1.5–3 m   | ~120–290        |
| 2    | Mid-range  | 3–5 m     | ~45–120         |
| 3    | Far        | 5–8 m     | ~15–45          |

### Fading Model
The simulation includes Rician fading (Line-of-Sight component) with:
- K-factor: 1.5–5 (dB), typical for indoor 5 GHz
- Log-normal shadowing: σ = 4–7 dB (ITU-R recommendation)

### Zone Classifier Architecture
```
Input: 16 features  (mean, std, p90-p10 range, energy per antenna pair × 4)
  └─ Linear(16→64) → BatchNorm → ReLU → Dropout(0.2)
  └─ Linear(64→32) → ReLU
  └─ Linear(32→4)
Output: zone logits → softmax → zone probabilities
```

### Feature Vector
```
For each antenna pair i ∈ {0,1,2,3}:
  mean_i     = mean(|H_{i,k}|) over subcarriers k and timesteps t
  std_i      = std(|H_{i,k}|)
  range_i    = percentile90 - percentile10  (robust spread)
  energy_i   = mean(|H_{i,k}|²)            (signal power)

→ concat → f ∈ ℝ^16
```

### Limitations (mandatory for scientific rigour)
1. **Calibration gap:** Zone boundaries derived from theoretical path loss,
   not real measurements. Real accuracy without calibration: ±1–2 zones.
2. **Angular ambiguity:** Two positions at the same distance but different
   angles produce similar amplitude statistics. Resolving this requires
   phase-based angle-of-arrival (AoA) estimation [Wu et al., 2012].
3. **NLOS degradation:** Walls, doors, and furniture cause 3–10 dB excess
   loss beyond the model, causing distance overestimation.
4. **Activity coupling:** Walking increases amplitude variance, which may
   be misinterpreted as proximity by the zone classifier.
5. **Simulated training data:** Real deployment requires a site survey with
   measurements at known positions.

---

## 7. Training Methodology

### Simulation Approach
Since the real dataset (9 GB) is impractical for continuous offline training,
we generate physics-inspired synthetic windows:

```python
# Example: Walking simulation
freq = U(1.8, 2.2) Hz          # gait frequency
amp_gait = U(30, 60, 456)      # per-subcarrier gait amplitude
phase = U(0, 2π, 456)          # per-subcarrier phase offset

amplitude[t, f] = base_mean[f] + amp_gait[f] · sin(2π·freq·t/Fs + phase[f])
                + N(0, 15)     # thermal noise
```

The models trained on simulated data demonstrate the architectural
differences clearly. When real CSI data is loaded via `dataset.py`
and `read_csi.py` (from the source repo), the same pipeline applies
without modification.

### Training Split
- Total windows: 2000 (uniform distribution: ~285 per class)
- Train / Val: 80% / 20% (random shuffle with fixed seed 42)
- No test set: evaluation on val set (standard for small simulation datasets)

### Dataset Balancing Strategy
The simulated dataset uses **uniform class distribution** (equal samples per
class) rather than following the real dataset's skewed proportions. Class
imbalance is handled exclusively by inverse-frequency loss weights. This
prevents the model from collapsing to predict frequent classes (Walking 43.9%)
and ensures the rare transition activities (Get Up/Down, 3.8%) receive enough
gradient signal.

### Hyperparameters
```
Batch size:         32
Epochs:             25 (HAR models), 30 (zone classifier)
Optimizer:          Adam (β₁=0.9, β₂=0.999, ε=10⁻⁸)
Weight decay:       10⁻⁴ (L₂ regularisation)
LR scheduler:       ReduceLROnPlateau (factor=0.5, patience=4)
Gradient clipping:  max_norm = 1.0

Model-specific learning rates (LSTMs need lower LR to avoid gradient collapse):
  SimpleLSTM:   3.0 × 10⁻⁴
  BiLSTM:       3.0 × 10⁻⁴
  FCN:          1.0 × 10⁻³
  Transformer:  1.46 × 10⁻³  (from source repo train_lstm.py)
```

---

## 8. Experimental Results

All results are on the held-out validation set (20% of 2000 simulated windows,
i.e. 400 samples). Data from `checkpoints/benchmark.json` — generated by
`train_all.py` (25 epochs, Adam, model-specific learning rates, CPU).

### HAR Model Comparison

| Model       | Val Acc | Val F1 (w) | Params  | Latency (CPU) |
|-------------|---------|------------|---------|---------------|
| SimpleLSTM  | 50.7%   | 0.471      | 888,455 | 9.6 ms        |
| BiLSTM      | 78.2%   | 0.768      | 658,311 | 8.8 ms        |
| FCN         | 30.8%   | 0.305      | 743,303 | 5.4 ms        |
| Transformer | **82.5%** | **0.826** | **330,887** | **5.3 ms** |

*Val F1 (w) = weighted F1-score. Latency = mean single-window inference over 50 runs.*

### Per-Class F1 — CSI Transformer (best model)

| Activity  | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Standing  | 0.500     | 0.556  | 0.526    | 63      |
| Walking   | 0.900     | 0.885  | 0.893    | 61      |
| Get Down  | 1.000     | 0.981  | 0.991    | 54      |
| Sitting   | 1.000     | 1.000  | 1.000    | 59      |
| Get Up    | 1.000     | 1.000  | 1.000    | 51      |
| Lying     | 0.947     | 0.947  | 0.947    | 57      |
| No Person | 0.480     | 0.436  | 0.457    | 55      |
| **Macro** | **0.832** | **0.829** | **0.831** | 400  |

### Per-Class F1 — BiLSTM

| Activity  | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Standing  | 0.420     | 0.794  | 0.549    | 63      |
| Walking   | 1.000     | 0.557  | 0.716    | 61      |
| Get Down  | 1.000     | 1.000  | 1.000    | 54      |
| Sitting   | 0.983     | 1.000  | 0.992    | 59      |
| Get Up    | 1.000     | 1.000  | 1.000    | 51      |
| Lying     | 0.824     | 0.982  | 0.896    | 57      |
| No Person | 0.643     | 0.164  | 0.261    | 55      |
| **Macro** | **0.839** | **0.785** | **0.773** | 400  |

### Per-Class F1 — SimpleLSTM

| Activity  | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Standing  | 0.321     | 0.143  | 0.198    | 63      |
| Walking   | 0.444     | 0.262  | 0.330    | 61      |
| Get Down  | 0.643     | 1.000  | 0.783    | 54      |
| Sitting   | 0.925     | 0.831  | 0.875    | 59      |
| Get Up    | 0.254     | 0.588  | 0.355    | 51      |
| Lying     | 0.556     | 0.702  | 0.620    | 57      |
| No Person | 0.556     | 0.091  | 0.156    | 55      |
| **Macro** | **0.528** | **0.517** | **0.474** | 400  |

### Per-Class F1 — FCN

| Activity  | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Standing  | 0.452     | 0.222  | 0.298    | 63      |
| Walking   | 0.198     | 0.410  | 0.267    | 61      |
| Get Down  | 0.156     | 0.093  | 0.116    | 54      |
| Sitting   | 0.789     | 0.763  | 0.776    | 59      |
| Get Up    | 0.237     | 0.176  | 0.202    | 51      |
| Lying     | 0.246     | 0.246  | 0.246    | 57      |
| No Person | 0.186     | 0.200  | 0.193    | 55      |
| **Macro** | **0.324** | **0.301** | **0.300** | 400  |

### Zone Classifier Results

| Zone       | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|---------|
| Proximity  | 0.973     | 1.000  | 0.986    | 71      |
| Near       | 0.965     | 0.976  | 0.971    | 85      |
| Mid-range  | 0.987     | 0.939  | 0.963    | 82      |
| Far        | 0.976     | 0.988  | 0.982    | 82      |
| **Macro**  | **0.975** | **0.976** | **0.975** | 320  |

Best zone classifier accuracy: **98.1%** (val set, 30 epochs).

### Interpretation

**Transformer (F1=0.826)** is the best model with the fewest parameters (330K).
The ConvStem captures local temporal patterns before global attention, enabling
efficient generalisation. Its 5.3 ms CPU latency makes real-time inference viable.

**BiLSTM (F1=0.768)** outperforms SimpleLSTM by a large margin (+0.297 F1),
confirming that bidirectional context is important for transition activities.
Both achieve F1=1.000 on Get Down and Get Up, where the burst-envelope
simulation is highly discriminative.

**FCN (F1=0.305)** severely overfits despite Dropout(0.2): training accuracy
reaches 100% while validation loss exceeds 3.5. The Conv1d architecture does
not generalise well to the simulated CSI regime with 468 input channels and
only 128 timesteps. With real data and longer sequences (1024 timesteps as
used in the source repo), FCN may perform comparably.

**Standing vs. No Person** is the hardest pair across all models. Both states
have near-zero Doppler and low amplitude variance; only the subtle breathing
oscillation (~0.3 Hz) in Standing distinguishes them from the static room.

**Zone estimation (98.1%)** is near-perfect on simulated data. The amplitude
statistics (mean, std, energy) derived from the ITU-R path loss model are
sufficiently separable across the 4 distance zones. Real-world accuracy without
a measurement campaign is expected to be substantially lower due to NLOS paths
and environment-specific attenuation.

---

## 9. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CSI-HAR SYSTEM                           │
├──────────────┬──────────────────────────────────────────────┤
│  DATA LAYER  │  real_csi.py → read binary packets           │
│              │  OR simulate_activity() → physics model      │
├──────────────┼──────────────────────────────────────────────┤
│  PREPROCESS  │  Hampel filter → wavelet denoise → PCA       │
│              │  Output: (T, 468) float32 tensor             │
├──────────────┼──────────────────────────────────────────────┤
│  HAR MODEL   │  [SimpleLSTM | BiLSTM | FCN | Transformer]   │
│  (selected)  │  Output: softmax probs (7,)                  │
├──────────────┼──────────────────────────────────────────────┤
│  ZONE MODEL  │  extract_zone_features() → ZoneClassifier    │
│              │  Output: zone probs (4,)                     │
├──────────────┼──────────────────────────────────────────────┤
│  GUI         │  CustomTkinter + Matplotlib TkAgg backend    │
│              │  120ms refresh, daemon simulation thread     │
└──────────────┴──────────────────────────────────────────────┘
```

### File Structure
```
TFG CSI/
├── model/
│   ├── transformer_model.py   CSI Transformer architecture
│   ├── models_zoo.py          SimpleLSTM, BiLSTM, FCN + registry
│   ├── position_estimator.py  ZoneClassifier + physics simulator
│   └── data_loader.py         Dataset, preprocessing, simulation
├── gui/
│   └── app.py                 Multi-panel monitoring interface
├── train.py                   Single-model training script
├── train_all.py               All-models training + benchmark
├── checkpoints/
│   ├── SimpleLSTM.pth
│   ├── BiLSTM.pth
│   ├── FCN.pth
│   ├── Transformer.pth
│   ├── zone_classifier.pth
│   ├── zone_stats.npz
│   ├── benchmark.json
│   ├── confusion_*.png
│   └── training_curves_all.png
└── SCIENCE.md                 This document
```

---

## 10. Limitations and Future Work

### Current Limitations

**Simulated data:** The biggest limitation. Real CSI data contains:
- Hardware-specific phase offsets (requires calibration)
- Packet loss and timing jitter
- Environmental non-stationarity (people, temperature, furniture changes)
- Domain shift between recording sessions

**Position estimation:** Without a measurement campaign, zone estimation
is model-derived, not empirically validated. Expected real-world accuracy
without calibration: zone-level (~1.5–2m granularity) at best.

**Sequence length:** We use 128-sample windows (6.4 s at 20 Hz).
The repository uses 1024-sample windows (51 s), which captures full
gait cycles and breathing patterns more reliably.

**CPU inference:** All models run on CPU. Real-time deployment may
require GPU or quantisation for latency < 10 ms.

### Future Work

1. **Real data integration:** Use `data_retrieval/` scripts to capture
   live CSI and feed directly into the inference pipeline.

2. **AoA-based localization:** Use phase differences between antenna pairs
   to estimate angle-of-arrival and triangulate position [SpotFi, 2015].

3. **Domain adaptation:** Train on one room, adapt to another using
   few-shot or self-supervised techniques.

4. **Continuous learning:** Update model weights online as new data arrives.

5. **Multi-person detection:** Extend to detect and track multiple occupants.

---

## 11. References

[1] Kovalenko, V. et al. (2021). *A Comprehensive Dataset for Human Activity
    Recognition Using Wi-Fi CSI*. IEEE DataPort.
    (Source repository and dataset)

[2] Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS 2017.
    (Transformer architecture)

[3] Wang, J. et al. (2015). *Understanding and Modeling of WiFi Signal Based
    Human Activity Recognition*. MobiCom 2015.
    (CSI-based HAR foundations)

[4] Wang, Z. et al. (2017). *Time Series Classification from Scratch with
    Deep Neural Networks: A Strong Baseline*. IJCNN 2017.
    (FCN architecture)

[5] Hochreiter, S. & Schmidhuber, J. (1997). *Long Short-Term Memory*.
    Neural Computation, 9(8), 1735–1780.
    (LSTM architecture)

[6] Chapre, Y. et al. (2014). *CSI-MIMO for Indoor Localisation*.
    Proc. 39th IEEE LCN, pp. 1–7.
    (CSI-based localization overview)

[7] Wu, C. et al. (2012). *FILA: Fine-grained Indoor Localisation*.
    IEEE INFOCOM 2012, pp. 2210–2218.
    (Fingerprinting-based WiFi localization)

[8] ITU-R P.1238-10 (2019). *Propagation data and prediction methods for
    the planning of indoor radiocommunication systems and radio local area
    networks in the frequency range 300 MHz to 450 GHz*.
    (Path loss model for zone estimation)

[9] Schuster, M. & Paliwal, K. (1997). *Bidirectional Recurrent Neural Networks*.
    IEEE Trans. Signal Processing, 45(11), 2673–2681.
    (BiLSTM architecture)

[10] Xiao, J. et al. (2021). *A Survey on Deep Learning-Based Human Activity
     Recognition Using WiFi CSI*. Sensors, 21(22), 7626.
     (Comprehensive survey — recommended for TFG background chapter)
