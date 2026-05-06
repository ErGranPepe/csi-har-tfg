"""
CSI-HAR Training Script
Matches the repository training setup (train_lstm.py) with:
  - Inverse-frequency class weights
  - Adam + ReduceLROnPlateau scheduler
  - Per-epoch accuracy & loss on train/val
  - Final: per-class Precision / Recall / F1, confusion matrix PNG
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless rendering
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, confusion_matrix,
                              f1_score, accuracy_score)

from model.transformer_model import CSITransformer
from model.data_loader import (get_dataloader, get_class_weights,
                                ACTIVITY_NAMES)

# ── Config (mirrors repo train_lstm.py where applicable) ─────────────────────
BATCH_SIZE   = 32
EPOCHS       = 30
LR           = 1.46e-3      # exact repo learning rate
SEQ_LEN      = 128          # timesteps per window  (repo uses 1024; 128 is enough for sim)
NUM_SAMPLES  = 3000         # simulated windows
INPUT_DIM    = 468
NUM_CLASSES  = 7
CHECKPOINT   = "checkpoints/transformer_csi.pth"
CURVES_PATH  = "checkpoints/training_curves.png"
CM_PATH      = "checkpoints/confusion_matrix.png"

os.makedirs("checkpoints", exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss   = criterion(logits, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * len(y)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.cpu().numpy())

    n    = len(all_labels)
    acc  = accuracy_score(all_labels, all_preds)
    f1   = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    return total_loss / n, acc, f1, np.array(all_labels), np.array(all_preds)


def save_curves(train_losses, val_losses, train_accs, val_accs):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    e = range(1, len(train_losses) + 1)

    axes[0].plot(e, train_losses, label='Train', color='#3498db')
    axes[0].plot(e, val_losses,   label='Val',   color='#e74c3c')
    axes[0].set_title('Loss per Epoch'); axes[0].set_xlabel('Epoch')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(e, [a * 100 for a in train_accs], label='Train', color='#3498db')
    axes[1].plot(e, [a * 100 for a in val_accs],   label='Val',   color='#e74c3c')
    axes[1].set_title('Accuracy per Epoch'); axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('%'); axes[1].legend(); axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(CURVES_PATH, dpi=120)
    plt.close(fig)


def save_confusion_matrix(y_true, y_pred):
    cm   = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(norm, cmap='Blues', vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(ACTIVITY_NAMES, rotation=35, ha='right', fontsize=9)
    ax.set_yticklabels(ACTIVITY_NAMES, fontsize=9)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title('Normalised Confusion Matrix – CSI-HAR Transformer')

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, f"{norm[i, j]:.2f}",
                    ha='center', va='center', fontsize=8,
                    color='white' if norm[i, j] > 0.5 else 'black')

    fig.tight_layout()
    fig.savefig(CM_PATH, dpi=130)
    plt.close(fig)
    print(f"  Confusion matrix saved → {CM_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print(f"  CSI-HAR Transformer  |  device: {device}")
    print(f"  Features: {INPUT_DIM}  |  Classes: {NUM_CLASSES}  |  Seq: {SEQ_LEN}")
    print(f"{'='*55}\n")

    print("Building dataset …")
    t0 = time.time()
    train_loader, val_loader = get_dataloader(
        batch_size=BATCH_SIZE, num_samples=NUM_SAMPLES,
        val_split=0.2, seq_len=SEQ_LEN)
    print(f"  {len(train_loader.dataset)} train  |  {len(val_loader.dataset)} val  "
          f"({time.time()-t0:.1f}s)\n")

    model     = CSITransformer(input_dim=INPUT_DIM, num_classes=NUM_CLASSES).to(device)
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}\n")

    # Inverse-frequency weights (repo exact)
    weights   = get_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)

    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []
    best_val_f1 = 0.0

    header = f"{'Ep':>3} | {'Tr-Loss':>8} {'Tr-Acc':>7} {'Tr-F1':>6} | {'Va-Loss':>8} {'Va-Acc':>7} {'Va-F1':>6} | {'LR':>8}"
    print(header)
    print('-' * len(header))

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc, tr_f1, _, _           = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        va_loss, va_acc, va_f1, va_y, va_pred  = run_epoch(model, val_loader,   criterion, None,      device, train=False)

        scheduler.step(va_loss)
        lr = optimizer.param_groups[0]['lr']

        train_losses.append(tr_loss); val_losses.append(va_loss)
        train_accs.append(tr_acc);    val_accs.append(va_acc)

        print(f"{epoch:>3} | {tr_loss:>8.4f} {tr_acc*100:>6.2f}% {tr_f1:>6.3f} | "
              f"{va_loss:>8.4f} {va_acc*100:>6.2f}% {va_f1:>6.3f} | {lr:.2e}")

        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            torch.save(model.state_dict(), CHECKPOINT)

    print(f"\nBest val F1: {best_val_f1:.3f}  >>  model saved to {CHECKPOINT}")

    # ── Final evaluation ──────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("FINAL EVALUATION ON VALIDATION SET")
    print("="*55)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device, weights_only=True))
    _, _, _, y_true, y_pred = run_epoch(model, val_loader, criterion, None, device, train=False)

    print("\nPer-class Report:")
    print(classification_report(y_true, y_pred, target_names=ACTIVITY_NAMES,
                                 digits=3, zero_division=0))

    save_curves(train_losses, val_losses, train_accs, val_accs)
    save_confusion_matrix(y_true, y_pred)
    print(f"  Training curves saved >> {CURVES_PATH}")
    print("\nDone. Run  python gui/app.py  to open the interface.\n")


if __name__ == "__main__":
    train()
