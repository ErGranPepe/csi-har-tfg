"""
train_all.py — Multi-Model Training & Benchmark
================================================
Trains all HAR models + the zone classifier sequentially.
Saves:
  checkpoints/<model_name>.pth          — trained weights
  checkpoints/benchmark.json            — accuracy/F1/latency comparison table
  checkpoints/confusion_<model>.png     — per-model confusion matrix
  checkpoints/training_curves_all.png   — overlaid loss/acc curves
  checkpoints/zone_classifier.pth       — zone estimator weights
  checkpoints/zone_stats.npz           — normalisation statistics for zone model
"""

import os, json, time, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, confusion_matrix,
                              f1_score, accuracy_score)
from torch.utils.data import DataLoader, random_split

from model.models_zoo import MODEL_CONFIGS, build_model, count_parameters
from model.data_loader import (get_dataloader, get_class_weights,
                                ACTIVITY_NAMES, TOTAL_FEATURES)
from model.position_estimator import (ZoneClassifier, ZoneDataset, ZONE_NAMES)

warnings.filterwarnings("ignore")
os.makedirs("checkpoints", exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
EPOCHS       = 25
BATCH_SIZE   = 32
NUM_SAMPLES  = 2000
SEQ_LEN      = 128
NUM_CLASSES  = 7
VAL_SPLIT    = 0.2

# Model-specific learning rates (LSTMs need lower LR to avoid collapse)
MODEL_LR = {
    "SimpleLSTM": 3e-4,
    "BiLSTM":     3e-4,
    "FCN":        1e-3,
    "Transformer": 1.46e-3,
}

ZONE_EPOCHS      = 30
ZONE_N_PER_ZONE  = 400


# ── Training loop (shared across all models) ──────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train() if train else model.eval()
    total_loss, all_pred, all_true = 0.0, [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss   = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * len(y)
            all_pred.extend(logits.argmax(1).cpu().numpy())
            all_true.extend(y.cpu().numpy())
    n   = len(all_true)
    acc = accuracy_score(all_true, all_pred)
    f1  = f1_score(all_true, all_pred, average="weighted", zero_division=0)
    return total_loss / n, acc, f1, np.array(all_true), np.array(all_pred)


def measure_latency(model, device, n_runs=50) -> float:
    """Mean inference time [ms] for one window (1, 128, 468)."""
    x = torch.randn(1, SEQ_LEN, TOTAL_FEATURES).to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(5):   # warmup
            model(x)
        t0 = time.perf_counter()
        for _ in range(n_runs):
            model(x)
        return (time.perf_counter() - t0) / n_runs * 1000


def save_confusion_matrix(y_true, y_pred, model_name):
    cm   = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(ACTIVITY_NAMES, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(ACTIVITY_NAMES, fontsize=9)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=12)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, f"{norm[i,j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if norm[i,j] > 0.5 else "black")
    fig.tight_layout()
    path = f"checkpoints/confusion_{model_name}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"    Confusion matrix  >> {path}")


# ── Train one HAR model ───────────────────────────────────────────────────────

def train_model(name: str, train_loader, val_loader, device):
    print(f"\n{'='*58}")
    print(f"  Training: {name}")
    print(f"{'='*58}")

    model    = build_model(name).to(device)
    n_params = count_parameters(model)
    print(f"  Parameters: {n_params:,}")

    weights   = get_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=MODEL_LR[name], weight_decay=1e-4)
    if name == "Transformer":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=4)

    tr_losses, va_losses, tr_accs, va_accs = [], [], [], []
    best_f1 = 0.0
    ckpt    = f"checkpoints/{name}.pth"

    hdr = f"  {'Ep':>3} | {'TrLoss':>7} {'TrAcc':>7} | {'VaLoss':>7} {'VaAcc':>7} {'VaF1':>6}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for ep in range(1, EPOCHS + 1):
        tr_l, tr_a, _, _, _        = run_epoch(model, train_loader, criterion, optimizer, device, True)
        va_l, va_a, va_f1, vy, vp  = run_epoch(model, val_loader,   criterion, None,      device, False)
        if isinstance(scheduler, optim.lr_scheduler.CosineAnnealingLR):
            scheduler.step()
        else:
            scheduler.step(va_l)
        tr_losses.append(tr_l); va_losses.append(va_l)
        tr_accs.append(tr_a);   va_accs.append(va_a)
        print(f"  {ep:>3} | {tr_l:>7.4f} {tr_a*100:>6.2f}% | {va_l:>7.4f} {va_a*100:>6.2f}% {va_f1:>6.3f}")
        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save(model.state_dict(), ckpt)
            last_vy, last_vp = vy, vp

    # Final metrics
    print(f"\n  Best val F1: {best_f1:.3f}  >>  {ckpt}")
    report = classification_report(last_vy, last_vp,
                                   target_names=ACTIVITY_NAMES,
                                   output_dict=True, zero_division=0)
    print(classification_report(last_vy, last_vp,
                                 target_names=ACTIVITY_NAMES,
                                 digits=3, zero_division=0))
    save_confusion_matrix(last_vy, last_vp, name)

    lat_ms = measure_latency(model, device)

    per_f1   = [round(report[a]["f1-score"],   4) for a in ACTIVITY_NAMES]
    per_prec = [round(report[a]["precision"],  4) for a in ACTIVITY_NAMES]
    per_rec  = [round(report[a]["recall"],     4) for a in ACTIVITY_NAMES]

    return {
        "name":                name,
        "description":         MODEL_CONFIGS[name]["description"],
        "paper":               MODEL_CONFIGS[name]["paper"],
        "n_params":            n_params,
        "best_val_f1":         round(best_f1, 4),
        "best_val_acc":        round(float(accuracy_score(last_vy, last_vp)), 4),
        "latency_ms":          round(lat_ms, 2),
        "per_class_f1":        per_f1,
        "per_class_precision": per_prec,
        "per_class_recall":    per_rec,
        "tr_losses":           tr_losses,
        "va_losses":           va_losses,
        "tr_accs":             tr_accs,
        "va_accs":             va_accs,
    }


# ── Train zone classifier ──────────────────────────────────────────────────────

def train_zone_classifier(device):
    print(f"\n{'='*58}")
    print("  Training: ZoneClassifier (position estimator)")
    print(f"{'='*58}")

    ds = ZoneDataset(n_per_zone=ZONE_N_PER_ZONE)
    np.savez("checkpoints/zone_stats.npz", mean=ds.mean, std=ds.std)

    n_val  = int(len(ds) * 0.2)
    n_tr   = len(ds) - n_val
    tr_ds, va_ds = random_split(ds, [n_tr, n_val],
                                 generator=torch.Generator().manual_seed(42))
    tr_dl = DataLoader(tr_ds, batch_size=64, shuffle=True)
    va_dl = DataLoader(va_ds, batch_size=64)

    model     = ZoneClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_acc = 0.0
    for ep in range(1, ZONE_EPOCHS + 1):
        model.train()
        for x, y in tr_dl:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        model.eval()
        all_p, all_y = [], []
        with torch.no_grad():
            for x, y in va_dl:
                all_p.extend(model(x.to(device)).argmax(1).cpu().numpy())
                all_y.extend(y.numpy())
        acc = accuracy_score(all_y, all_p)
        scheduler.step(1 - acc)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "checkpoints/zone_classifier.pth")
        if ep % 10 == 0:
            print(f"  Ep {ep:>2} | Zone val acc: {acc*100:.1f}%")

    print(f"  Best zone acc: {best_acc*100:.1f}%  >>  checkpoints/zone_classifier.pth")
    zone_report = classification_report(all_y, all_p, target_names=ZONE_NAMES,
                                        output_dict=True, zero_division=0)
    print(classification_report(all_y, all_p, target_names=ZONE_NAMES,
                                 digits=3, zero_division=0))
    return {
        "best_acc":            round(best_acc, 4),
        "per_class_f1":        [round(zone_report[z]["f1-score"],  4) for z in ZONE_NAMES],
        "per_class_precision": [round(zone_report[z]["precision"], 4) for z in ZONE_NAMES],
        "per_class_recall":    [round(zone_report[z]["recall"],    4) for z in ZONE_NAMES],
    }


# ── Combined training curve plot ───────────────────────────────────────────────

def save_all_curves(results):
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for r, c in zip(results, colors):
        e = range(1, len(r["tr_losses"]) + 1)
        axes[0].plot(e, r["va_losses"], color=c, lw=1.8, label=r["name"])
        axes[1].plot(e, [a * 100 for a in r["va_accs"]], color=c, lw=1.8, label=r["name"])

    axes[0].set_title("Validation Loss per Epoch"); axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].set_title("Validation Accuracy per Epoch"); axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("%"); axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.suptitle("CSI-HAR — Model Comparison (Validation Set)", fontsize=13)
    fig.tight_layout()
    fig.savefig("checkpoints/training_curves_all.png", dpi=130)
    plt.close(fig)
    print("\n  All curves  >>  checkpoints/training_curves_all.png")


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(results):
    print("\n" + "=" * 68)
    print(f"  {'Model':<18} {'Params':>8} {'Val Acc':>8} {'Val F1':>7} {'Latency':>10}")
    print("  " + "-" * 56)
    for r in results:
        print(f"  {r['name']:<18} {r['n_params']:>8,} "
              f"{r['best_val_acc']*100:>7.1f}% {r['best_val_f1']:>7.3f} "
              f"{r['latency_ms']:>8.1f} ms")
    print("=" * 68)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}  |  Building shared dataset ({NUM_SAMPLES} windows)…")
    t0 = time.time()
    train_loader, val_loader = get_dataloader(
        batch_size=BATCH_SIZE, num_samples=NUM_SAMPLES,
        val_split=VAL_SPLIT, seq_len=SEQ_LEN)
    print(f"Dataset ready in {time.time()-t0:.1f}s  "
          f"({len(train_loader.dataset)} train / {len(val_loader.dataset)} val)\n")

    results = []
    for name in MODEL_CONFIGS:
        r = train_model(name, train_loader, val_loader, device)
        results.append(r)

    zone_r = train_zone_classifier(device)

    save_all_curves(results)
    print_summary(results)

    # Strip large lists before serialising
    bench = []
    for r in results:
        b = {k: v for k, v in r.items()
             if k not in ("tr_losses", "va_losses", "tr_accs", "va_accs")}
        bench.append(b)
    bench_data = {"har_models": bench, "zone_classifier": zone_r}
    with open("checkpoints/benchmark.json", "w") as f:
        json.dump(bench_data, f, indent=2)
    print("  Benchmark  >>  checkpoints/benchmark.json")
    print("\nDone. Run   py gui/app.py   to open the full interface.\n")


if __name__ == "__main__":
    main()
