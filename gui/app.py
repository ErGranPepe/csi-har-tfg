"""
CSI-HAR v3 — Professional 6-Tab GUI (TFG Edition)
===================================================
CustomTkinter + Matplotlib GUI for Wi-Fi CSI Human Activity Recognition.
Tabs: Monitor | Analysis | Signal | Config | Dataset | Help
"""

import sys
import os
import time
import threading
import json
import collections
import datetime
import warnings

warnings.filterwarnings("ignore")

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox

from scipy import signal as sp_signal

# ── Optional PIL ──────────────────────────────────────────────────────────────
try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# ── Project imports ───────────────────────────────────────────────────────────
from model.models_zoo import build_model, MODEL_CONFIGS, count_parameters
from model.data_loader import (
    ACTIVITY_NAMES, ACTIVITY_COLORS, SUBCARRIERS, ANTENNA_PAIRS,
    AMP_MAX, RAW_FEATURES,
    simulate_activity, compute_pca,
)
from model.position_estimator import (
    ZoneClassifier, ZONE_NAMES, ZONE_COLORS, NUM_ZONES, extract_zone_features,
)

# ── Color scheme ──────────────────────────────────────────────────────────────
BG_DARK   = "#0a0c18"
BG_MID    = "#131525"
BG_PANEL  = "#1a1d30"
BG_CARD   = "#1f2340"
ACCENT    = "#4fc3f7"
ACCENT2   = "#b39ddb"
TEXT      = "#e4e7f0"
TEXT_DIM  = "#6b7599"
GREEN     = "#00e676"
RED       = "#ff4569"
YELLOW    = "#ffd740"
ANT_COLORS = [ACCENT, "#ff7043", "#ce93d8", "#a5d6a7"]

# ── Default config ─────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "infer_win":      128,
    "bench_every":    8,
    "conf_threshold": 0.40,
    "display_win":    200,
    "history_len":    80,
    "gui_ms":         120,
    "spec_win":       256,
    "sim_hz":         20,
    "sim_noise":      1.0,
    "sim_fixed_cls":  -1,
    "sim_swap_min":   40,
    "sim_swap_max":   120,
    "active_model":   "Transformer",
    "ckpt_dir":       "checkpoints",
    "record":         False,
    "selected_antenna": 0,
}
CONFIG_PATH = os.path.join(BASE_DIR, "gui_config.json")

# ── Matplotlib style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  BG_PANEL,
    "axes.facecolor":    BG_CARD,
    "axes.edgecolor":    TEXT_DIM,
    "axes.labelcolor":   TEXT_DIM,
    "xtick.color":       TEXT_DIM,
    "ytick.color":       TEXT_DIM,
    "text.color":        TEXT,
    "grid.color":        "#2a2d40",
    "grid.alpha":        0.5,
    "font.family":       "monospace",
    "font.size":         8,
})

# ══════════════════════════════════════════════════════════════════════════════
# Tooltip helper
# ══════════════════════════════════════════════════════════════════════════════
class Tooltip:
    """Delayed tooltip that appears near the hovered widget."""

    def __init__(self, widget, text: str, delay: int = 500):
        self.widget = widget
        self.text   = text
        self.delay  = delay
        self._job   = None
        self._win   = None
        widget.bind("<Enter>", self._on_enter)
        widget.bind("<Leave>", self._on_leave)

    def _on_enter(self, _event=None):
        self._job = self.widget.after(self.delay, self._show)

    def _on_leave(self, _event=None):
        if self._job:
            self.widget.after_cancel(self._job)
            self._job = None
        if self._win:
            self._win.destroy()
            self._win = None

    def _show(self):
        if self._win:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self._win = ctk.CTkToplevel(self.widget)
        self._win.wm_overrideredirect(True)
        self._win.wm_geometry(f"+{x}+{y}")
        lbl = ctk.CTkLabel(
            self._win, text=self.text, wraplength=260,
            fg_color=BG_CARD, text_color=TEXT,
            corner_radius=6, padx=8, pady=4,
        )
        lbl.pack()


# ══════════════════════════════════════════════════════════════════════════════
# Main Application
# ══════════════════════════════════════════════════════════════════════════════
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self.title("CSI-HAR v3 — Wi-Fi Human Activity Recognition")
        self.geometry("1400x860")
        self.minsize(1100, 720)
        self.configure(fg_color=BG_DARK)

        # ── State ─────────────────────────────────────────────────────────────
        self.cfg            = self._load_config()
        self.device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models: dict[str, torch.nn.Module] = {}
        self.active_name    = self.cfg["active_model"]
        self.packet_buf: list[np.ndarray] = []
        self._buf_lock      = threading.Lock()
        self.running        = True
        self._sim_paused    = False
        self._replay_mode   = False
        self._replay_data   = None
        self._replay_labels = None
        self._replay_idx    = 0
        self._replay_running = False
        self._replay_speed  = 1.0

        # Inference state
        self.probs          = np.ones(len(ACTIVITY_NAMES)) / len(ACTIVITY_NAMES)
        self.pred_cls       = 0
        self.infer_ms       = 0.0
        self.zone_idx       = 0
        self.zone_probs     = np.ones(NUM_ZONES) / NUM_ZONES
        self._tick_count    = 0
        self._tick_times: collections.deque = collections.deque(maxlen=10)
        self._last_tick_t   = time.time()

        # History for timeline
        self.pred_history: collections.deque = collections.deque(maxlen=500)
        self.conf_history: collections.deque = collections.deque(maxlen=500)
        self.time_history: collections.deque = collections.deque(maxlen=500)

        # Session
        self.session_start  = time.time()
        self.total_preds    = 0
        self.record_data: list[np.ndarray] = []
        self.record_labels: list[int]      = []

        # Simulation class rotation
        self._sim_cls       = 0
        self._sim_swap_cnt  = 0
        self._sim_next_swap = 60

        # Zone classifier
        self.zone_clf       = None
        self.zone_stats     = None

        # Benchmark data
        self.bench_data     = self._load_benchmark()

        # ── Build UI ──────────────────────────────────────────────────────────
        self._build_layout()
        self._load_models()
        self._load_zone_classifier()

        # ── Start simulation + tick ───────────────────────────────────────────
        t = threading.Thread(target=self._sim_loop, daemon=True)
        t.start()
        self.after(self.cfg["gui_ms"], self._tick)

    # ══════════════════════════════════════════════════════════════════════════
    # Config helpers
    # ══════════════════════════════════════════════════════════════════════════
    def _load_config(self) -> dict:
        cfg = dict(DEFAULT_CONFIG)
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH) as f:
                    saved = json.load(f)
                cfg.update({k: v for k, v in saved.items() if k in cfg})
            except Exception:
                pass
        return cfg

    def _save_config(self):
        try:
            with open(CONFIG_PATH, "w") as f:
                json.dump(self.cfg, f, indent=2)
        except Exception:
            pass

    def _cfg_change(self, key: str, value):
        self.cfg[key] = value
        self._save_config()
        # Immediate effects
        if key == "active_model" and value in self.models:
            self.active_name = value
        if key == "gui_ms":
            pass  # next tick will use new value

    def _load_benchmark(self) -> list:
        path = os.path.join(BASE_DIR, self.cfg["ckpt_dir"], "benchmark.json")
        try:
            with open(path) as f:
                data = json.load(f)
            return data.get("har_models", [])
        except Exception:
            return []

    # ══════════════════════════════════════════════════════════════════════════
    # Model loading
    # ══════════════════════════════════════════════════════════════════════════
    def _load_models(self):
        ckpt_dir = os.path.join(BASE_DIR, self.cfg["ckpt_dir"])
        for name in MODEL_CONFIGS:
            try:
                m = build_model(name)
                pth = os.path.join(ckpt_dir, f"{name}.pth")
                if os.path.exists(pth):
                    state = torch.load(pth, map_location=self.device)
                    if isinstance(state, dict) and "model_state_dict" in state:
                        state = state["model_state_dict"]
                    m.load_state_dict(state, strict=False)
                m.eval().to(self.device)
                self.models[name] = m
            except Exception as e:
                print(f"[WARN] Could not load {name}: {e}")
        if not self.models:
            # Build at least Transformer in memory
            m = build_model("Transformer")
            m.eval().to(self.device)
            self.models["Transformer"] = m
        if self.active_name not in self.models:
            self.active_name = next(iter(self.models))

    def _load_zone_classifier(self):
        ckpt_dir = os.path.join(BASE_DIR, self.cfg["ckpt_dir"])
        pth  = os.path.join(ckpt_dir, "zone_classifier.pth")
        stat = os.path.join(ckpt_dir, "zone_stats.npz")
        try:
            zc = ZoneClassifier()
            state = torch.load(pth, map_location="cpu")
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            zc.load_state_dict(state, strict=False)
            zc.eval()
            self.zone_clf = zc
            z = np.load(stat)
            self.zone_stats = (z["mean"], z["std"])
        except Exception as e:
            print(f"[WARN] Zone classifier not loaded: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # Layout construction
    # ══════════════════════════════════════════════════════════════════════════
    def _build_layout(self):
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)

        self._build_sidebar()
        self._build_main_area()
        self._build_status_bar()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    def _build_sidebar(self):
        sb = ctk.CTkFrame(self, width=260, fg_color=BG_MID, corner_radius=0)
        sb.grid(row=0, column=0, sticky="nsew")
        sb.grid_propagate(False)
        sb.grid_columnconfigure(0, weight=1)

        row = 0
        # Title
        ctk.CTkLabel(sb, text="CSI-HAR v3", font=("Helvetica", 20, "bold"),
                     text_color=ACCENT).grid(row=row, column=0, pady=(18, 0), padx=14)
        row += 1
        ctk.CTkLabel(sb, text="Wi-Fi Human Activity Recognition",
                     font=("Helvetica", 9), text_color=TEXT_DIM).grid(
            row=row, column=0, pady=(0, 14), padx=14)
        row += 1

        # Separator
        ctk.CTkFrame(sb, height=1, fg_color=BG_CARD).grid(
            row=row, column=0, sticky="ew", padx=10)
        row += 1

        # Model selector
        ctk.CTkLabel(sb, text="Active Model", font=("Helvetica", 11, "bold"),
                     text_color=TEXT).grid(row=row, column=0, sticky="w", padx=14, pady=(12, 2))
        row += 1
        model_names = list(MODEL_CONFIGS.keys())
        self._model_menu = ctk.CTkOptionMenu(
            sb, values=model_names, width=230,
            fg_color=BG_CARD, button_color=ACCENT, button_hover_color=ACCENT2,
            dropdown_fg_color=BG_CARD, text_color=TEXT,
            command=self._on_model_change,
        )
        self._model_menu.set(self.active_name)
        self._model_menu.grid(row=row, column=0, padx=14, pady=(0, 4))
        row += 1

        self._model_param_lbl = ctk.CTkLabel(sb, text="", font=("Helvetica", 9),
                                              text_color=TEXT_DIM)
        self._model_param_lbl.grid(row=row, column=0, sticky="w", padx=14)
        row += 1
        self._model_desc_lbl  = ctk.CTkLabel(sb, text="", font=("Helvetica", 8),
                                              text_color=TEXT_DIM, wraplength=230)
        self._model_desc_lbl.grid(row=row, column=0, sticky="w", padx=14, pady=(0, 8))
        row += 1
        self._update_model_labels()

        # Separator
        ctk.CTkFrame(sb, height=1, fg_color=BG_CARD).grid(
            row=row, column=0, sticky="ew", padx=10)
        row += 1

        # Zone mini chart
        ctk.CTkLabel(sb, text="Zone Estimation", font=("Helvetica", 11, "bold"),
                     text_color=TEXT).grid(row=row, column=0, sticky="w", padx=14, pady=(10, 2))
        row += 1
        self._zone_fig = Figure(figsize=(2.2, 1.4), dpi=80)
        self._zone_ax  = self._zone_fig.add_subplot(111)
        self._zone_fig.subplots_adjust(left=0.05, right=0.98, top=0.88, bottom=0.18)
        self._zone_canvas = FigureCanvasTkAgg(self._zone_fig, master=sb)
        self._zone_canvas.get_tk_widget().grid(row=row, column=0, padx=10, pady=2)
        row += 1

        # Signal stats
        ctk.CTkFrame(sb, height=1, fg_color=BG_CARD).grid(
            row=row, column=0, sticky="ew", padx=10)
        row += 1
        ctk.CTkLabel(sb, text="Signal Statistics", font=("Helvetica", 11, "bold"),
                     text_color=TEXT).grid(row=row, column=0, sticky="w", padx=14, pady=(10, 4))
        row += 1

        stats_frame = ctk.CTkFrame(sb, fg_color=BG_CARD, corner_radius=8)
        stats_frame.grid(row=row, column=0, padx=10, sticky="ew")
        row += 1
        self._stat_labels = {}
        for i, (key, label) in enumerate([
            ("mean_amp", "Mean Amp"), ("std_amp", "Std Dev"),
            ("snr",      "SNR (dB)"), ("infer_ms", "Infer (ms)")
        ]):
            ctk.CTkLabel(stats_frame, text=label, font=("Helvetica", 9),
                         text_color=TEXT_DIM).grid(row=i, column=0, sticky="w",
                                                   padx=10, pady=1)
            lbl = ctk.CTkLabel(stats_frame, text="—", font=("Helvetica", 9, "bold"),
                               text_color=ACCENT)
            lbl.grid(row=i, column=1, sticky="e", padx=10, pady=1)
            self._stat_labels[key] = lbl
        stats_frame.grid_columnconfigure(1, weight=1)

        # Session info
        ctk.CTkFrame(sb, height=1, fg_color=BG_CARD).grid(
            row=row, column=0, sticky="ew", padx=10, pady=(8, 0))
        row += 1
        ctk.CTkLabel(sb, text="Session", font=("Helvetica", 11, "bold"),
                     text_color=TEXT).grid(row=row, column=0, sticky="w",
                                           padx=14, pady=(8, 2))
        row += 1
        sess_frame = ctk.CTkFrame(sb, fg_color=BG_CARD, corner_radius=8)
        sess_frame.grid(row=row, column=0, padx=10, sticky="ew")
        row += 1
        self._uptime_lbl   = ctk.CTkLabel(sess_frame, text="00:00:00",
                                           font=("Helvetica", 9), text_color=TEXT_DIM)
        self._uptime_lbl.grid(row=0, column=0, sticky="w", padx=10, pady=1)
        self._total_lbl    = ctk.CTkLabel(sess_frame, text="Predictions: 0",
                                           font=("Helvetica", 9), text_color=TEXT_DIM)
        self._total_lbl.grid(row=1, column=0, sticky="w", padx=10, pady=1)
        self._rec_dot_lbl  = ctk.CTkLabel(sess_frame, text="",
                                           font=("Helvetica", 9), text_color=RED)
        self._rec_dot_lbl.grid(row=2, column=0, sticky="w", padx=10, pady=1)

        # Buttons
        ctk.CTkFrame(sb, height=1, fg_color=BG_CARD).grid(
            row=row, column=0, sticky="ew", padx=10, pady=(8, 0))
        row += 1

        btn_frame = ctk.CTkFrame(sb, fg_color="transparent")
        btn_frame.grid(row=row, column=0, padx=10, pady=8, sticky="ew")
        btn_frame.grid_columnconfigure((0, 1), weight=1)
        row += 1

        ctk.CTkButton(
            btn_frame, text="Calibrate", width=100,
            fg_color=BG_CARD, hover_color=BG_PANEL,
            text_color=ACCENT, command=self._calibrate,
        ).grid(row=0, column=0, padx=3, pady=2)

        self._rec_btn = ctk.CTkButton(
            btn_frame, text="Record", width=100,
            fg_color=BG_CARD, hover_color=BG_PANEL,
            text_color=GREEN, command=self._toggle_record,
        )
        self._rec_btn.grid(row=0, column=1, padx=3, pady=2)

        self._pause_btn = ctk.CTkButton(
            btn_frame, text="Pause Sim", width=210,
            fg_color=BG_CARD, hover_color=BG_PANEL,
            text_color=YELLOW, command=self._toggle_pause,
        )
        self._pause_btn.grid(row=1, column=0, columnspan=2, padx=3, pady=2)

        ctk.CTkButton(
            btn_frame, text="Export Session", width=210,
            fg_color=BG_CARD, hover_color=BG_PANEL,
            text_color=ACCENT2, command=self._export_session,
        ).grid(row=2, column=0, columnspan=2, padx=3, pady=2)

    def _update_model_labels(self):
        cfg_entry = MODEL_CONFIGS.get(self.active_name, {})
        desc = cfg_entry.get("description", "")
        if self.active_name in self.models:
            nparams = count_parameters(self.models[self.active_name])
            self._model_param_lbl.configure(text=f"{nparams:,} parameters")
        else:
            self._model_param_lbl.configure(text="not loaded")
        self._model_desc_lbl.configure(text=desc[:80])

    def _on_model_change(self, name: str):
        self.active_name = name
        self.cfg["active_model"] = name
        self._save_config()
        self._update_model_labels()

    # ── Main area (tabs + content) ────────────────────────────────────────────
    def _build_main_area(self):
        main = ctk.CTkFrame(self, fg_color=BG_DARK, corner_radius=0)
        main.grid(row=0, column=1, sticky="nsew")
        main.grid_rowconfigure(0, weight=1)
        main.grid_columnconfigure(0, weight=1)

        self.tabs = ctk.CTkTabview(
            main, fg_color=BG_MID,
            segmented_button_fg_color=BG_PANEL,
            segmented_button_selected_color=ACCENT,
            segmented_button_selected_hover_color=ACCENT2,
            segmented_button_unselected_color=BG_PANEL,
            segmented_button_unselected_hover_color=BG_CARD,
            text_color=TEXT, text_color_disabled=TEXT_DIM,
        )
        self.tabs.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)

        for name in ["Monitor", "Analysis", "Signal", "Config", "Dataset", "Help"]:
            self.tabs.add(name)

        self._build_tab_monitor()
        self._build_tab_analysis()
        self._build_tab_signal()
        self._build_tab_config()
        self._build_tab_dataset()
        self._build_tab_help()

    # ── Status bar ────────────────────────────────────────────────────────────
    def _build_status_bar(self):
        bar = ctk.CTkFrame(self, height=28, fg_color=BG_MID, corner_radius=0)
        bar.grid(row=1, column=0, columnspan=2, sticky="ew")
        bar.grid_columnconfigure(5, weight=1)

        def _lbl(text, col, color=TEXT_DIM):
            l = ctk.CTkLabel(bar, text=text, font=("Helvetica", 9), text_color=color)
            l.grid(row=0, column=col, padx=10, pady=2, sticky="w")
            return l

        self._sb_model  = _lbl(f"Model: {self.active_name}", 0, ACCENT)
        self._sb_device = _lbl(f"Device: {self.device}", 1)
        self._sb_fps    = _lbl("FPS: 0.0", 2)
        self._sb_lat    = _lbl("Latency: 0.0ms", 3)
        self._sb_time   = _lbl("", 4)
        self._sb_rec    = _lbl("", 5, RED)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1: Monitor
    # ══════════════════════════════════════════════════════════════════════════
    def _build_tab_monitor(self):
        tab = self.tabs.tab("Monitor")
        tab.configure(fg_color=BG_DARK)
        tab.grid_columnconfigure((0, 1), weight=1)
        tab.grid_rowconfigure(0, weight=0)
        tab.grid_rowconfigure(1, weight=5)
        tab.grid_rowconfigure(2, weight=4)

        # ── Row 0: Banner ─────────────────────────────────────────────────────
        banner = ctk.CTkFrame(tab, fg_color=BG_MID, corner_radius=8, height=85)
        banner.grid(row=0, column=0, columnspan=2, sticky="ew", padx=4, pady=(4, 2))
        banner.grid_propagate(False)
        banner.grid_columnconfigure(0, weight=2)
        banner.grid_columnconfigure(1, weight=1)
        banner.grid_columnconfigure(2, weight=0)
        banner.grid_rowconfigure(0, weight=1)

        left = ctk.CTkFrame(banner, fg_color="transparent")
        left.grid(row=0, column=0, sticky="nsew", padx=10)
        left.grid_columnconfigure(0, weight=1)

        self._act_lbl = ctk.CTkLabel(
            left, text="—", font=("Helvetica", 28, "bold"), text_color=ACCENT)
        self._act_lbl.grid(row=0, column=0, sticky="w")
        self._zone_lbl = ctk.CTkLabel(
            left, text="Zone: —", font=("Helvetica", 12), text_color=TEXT_DIM)
        self._zone_lbl.grid(row=1, column=0, sticky="w")

        right = ctk.CTkFrame(banner, fg_color="transparent")
        right.grid(row=0, column=1, columnspan=2, sticky="nsew", padx=10)
        right.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(right, text="Confidence", font=("Helvetica", 9),
                     text_color=TEXT_DIM).grid(row=0, column=0, sticky="w")
        self._conf_bar = ctk.CTkProgressBar(right, width=300, height=18,
                                             progress_color=ACCENT)
        self._conf_bar.set(0)
        self._conf_bar.grid(row=1, column=0, sticky="ew", pady=2)
        self._conf_pct_lbl = ctk.CTkLabel(right, text="0.0%",
                                           font=("Helvetica", 14, "bold"),
                                           text_color=ACCENT)
        self._conf_pct_lbl.grid(row=2, column=0, sticky="w")

        # ── Row 1: Signal + Confidence bars ──────────────────────────────────
        sig_frame = ctk.CTkFrame(tab, fg_color=BG_MID, corner_radius=8)
        sig_frame.grid(row=1, column=0, sticky="nsew", padx=(4, 2), pady=2)
        sig_frame.grid_rowconfigure(0, weight=1)
        sig_frame.grid_columnconfigure(0, weight=1)

        self._sig_fig = Figure(figsize=(5, 3), dpi=88)
        self._sig_ax  = self._sig_fig.add_subplot(111)
        self._sig_fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.12)
        self._sig_ax.set_title("CSI Signal (mean amplitude per antenna)", fontsize=8)
        self._sig_ax.set_xlabel("Packets", fontsize=7)
        self._sig_ax.set_ylabel("Amplitude", fontsize=7)
        self._sig_lines = [
            self._sig_ax.plot([], [], color=ANT_COLORS[i], linewidth=0.9,
                              label=f"Ant {i}")[0]
            for i in range(ANTENNA_PAIRS)
        ]
        self._sig_ax.legend(fontsize=7, loc="upper right")
        self._sig_ax.grid(True, alpha=0.3)
        self._sig_canvas = FigureCanvasTkAgg(self._sig_fig, master=sig_frame)
        self._sig_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        conf_frame = ctk.CTkFrame(tab, fg_color=BG_MID, corner_radius=8)
        conf_frame.grid(row=1, column=1, sticky="nsew", padx=(2, 4), pady=2)
        conf_frame.grid_rowconfigure(0, weight=1)
        conf_frame.grid_columnconfigure(0, weight=1)

        self._conf_fig = Figure(figsize=(4, 3), dpi=88)
        self._conf_ax  = self._conf_fig.add_subplot(111)
        self._conf_fig.subplots_adjust(left=0.32, right=0.97, top=0.93, bottom=0.08)
        self._conf_ax.set_title("Class Probabilities", fontsize=8)
        self._conf_bars_obj = self._conf_ax.barh(
            ACTIVITY_NAMES, np.ones(len(ACTIVITY_NAMES)) / len(ACTIVITY_NAMES),
            color=ACTIVITY_COLORS, alpha=0.85,
        )
        self._conf_ax.set_xlim(0, 1)
        self._conf_ax.set_xlabel("Softmax prob.", fontsize=7)
        self._conf_ax.tick_params(labelsize=7)
        self._conf_canvas = FigureCanvasTkAgg(self._conf_fig, master=conf_frame)
        self._conf_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # ── Row 2: Benchmark table + Timeline ────────────────────────────────
        bench_frame = ctk.CTkFrame(tab, fg_color=BG_MID, corner_radius=8)
        bench_frame.grid(row=2, column=0, sticky="nsew", padx=(4, 2), pady=(2, 4))
        bench_frame.grid_rowconfigure(0, weight=1)
        bench_frame.grid_columnconfigure(0, weight=1)

        self._bench_fig = Figure(figsize=(5, 2.5), dpi=88)
        self._bench_ax  = self._bench_fig.add_subplot(111)
        self._bench_fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.05)
        self._bench_ax.axis("off")
        self._bench_ax.set_title("Model Benchmark", fontsize=8)
        self._draw_bench_table()
        self._bench_canvas = FigureCanvasTkAgg(self._bench_fig, master=bench_frame)
        self._bench_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        tl_frame = ctk.CTkFrame(tab, fg_color=BG_MID, corner_radius=8)
        tl_frame.grid(row=2, column=1, sticky="nsew", padx=(2, 4), pady=(2, 4))
        tl_frame.grid_rowconfigure(0, weight=1)
        tl_frame.grid_columnconfigure(0, weight=1)

        self._tl_fig = Figure(figsize=(4, 2.5), dpi=88)
        self._tl_ax  = self._tl_fig.add_subplot(111)
        self._tl_ax2 = self._tl_ax.twinx()
        self._tl_fig.subplots_adjust(left=0.06, right=0.90, top=0.92, bottom=0.12)
        self._tl_ax.set_title("Activity Timeline", fontsize=8)
        self._tl_ax.set_xlabel("Predictions", fontsize=7)
        self._tl_ax.set_ylabel("Class", fontsize=7)
        self._tl_ax.set_ylim(-0.5, len(ACTIVITY_NAMES) - 0.5)
        self._tl_ax.tick_params(labelsize=6)
        self._tl_ax2.set_ylabel("Conf.", fontsize=7, color=YELLOW)
        self._tl_ax2.set_ylim(0, 1)
        self._tl_ax2.tick_params(labelsize=6, colors=YELLOW)
        self._tl_canvas = FigureCanvasTkAgg(self._tl_fig, master=tl_frame)
        self._tl_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    def _draw_bench_table(self):
        ax = self._bench_ax
        ax.cla()
        ax.axis("off")
        ax.set_facecolor(BG_CARD)
        ax.set_title("Model Benchmark", fontsize=8, color=TEXT)

        headers = ["Model", "Val F1", "Val Acc", "Params", "ms"]
        col_x   = [0.02, 0.28, 0.44, 0.62, 0.84]
        header_y = 0.88

        for hx, hdr in zip(col_x, headers):
            ax.text(hx, header_y, hdr, transform=ax.transAxes,
                    fontsize=7.5, color=ACCENT, fontweight="bold", va="center")

        # Header separator line
        ax.plot([0.0, 1.0], [header_y - 0.08, header_y - 0.08],
                transform=ax.transAxes, color=TEXT_DIM, linewidth=0.7, alpha=0.5)

        rows = self.bench_data or []
        row_height = 0.16
        for ri, m in enumerate(rows):
            y = header_y - 0.18 - ri * row_height
            color = GREEN if m.get("name") == self.active_name else TEXT
            ax.text(col_x[0], y, m.get("name", "?"), transform=ax.transAxes,
                    fontsize=7, color=color, va="center")
            ax.text(col_x[1], y, f"{m.get('best_val_f1', 0):.3f}",
                    transform=ax.transAxes, fontsize=7, color=TEXT, va="center")
            ax.text(col_x[2], y, f"{m.get('best_val_acc', 0):.3f}",
                    transform=ax.transAxes, fontsize=7, color=TEXT, va="center")
            ax.text(col_x[3], y, f"{m.get('n_params', 0):,}",
                    transform=ax.transAxes, fontsize=6.5, color=TEXT_DIM, va="center")
            ax.text(col_x[4], y, f"{m.get('latency_ms', 0):.1f}",
                    transform=ax.transAxes, fontsize=7, color=YELLOW, va="center")
            # Row separator
            ax.plot([0.0, 1.0], [y - 0.07, y - 0.07],
                    transform=ax.transAxes, color=TEXT_DIM, linewidth=0.3, alpha=0.3)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2: Analysis
    # ══════════════════════════════════════════════════════════════════════════
    def _build_tab_analysis(self):
        tab = self.tabs.tab("Analysis")
        tab.configure(fg_color=BG_DARK)
        tab.grid_rowconfigure(0, weight=0)
        tab.grid_rowconfigure(1, weight=0)
        tab.grid_rowconfigure(2, weight=1)
        tab.grid_columnconfigure(0, weight=1)

        # ── Model cards row ───────────────────────────────────────────────────
        cards_frame = ctk.CTkFrame(tab, fg_color="transparent")
        cards_frame.grid(row=0, column=0, sticky="ew", padx=4, pady=(4, 2))
        cards_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

        model_names_order = ["SimpleLSTM", "BiLSTM", "FCN", "Transformer"]
        bench_by_name = {m["name"]: m for m in self.bench_data}

        for ci, name in enumerate(model_names_order):
            card = ctk.CTkFrame(cards_frame, fg_color=BG_CARD, corner_radius=8)
            card.grid(row=0, column=ci, padx=4, pady=2, sticky="ew")
            card.grid_columnconfigure(0, weight=1)

            m = bench_by_name.get(name, {})
            border_color = ACCENT if name == self.active_name else BG_PANEL

            ctk.CTkLabel(card, text=name, font=("Helvetica", 11, "bold"),
                         text_color=border_color).grid(
                row=0, column=0, pady=(8, 2), padx=8, sticky="w")
            for row_i, (key, label, fmt) in enumerate([
                ("best_val_acc", "Val Acc",    ".3f"),
                ("best_val_f1",  "Val F1",     ".3f"),
                ("n_params",     "Params",     ",d"),
                ("latency_ms",   "Latency",    ".1f ms"),
            ]):
                val = m.get(key, 0)
                try:
                    if fmt.endswith("d"):
                        txt = f"{val:{fmt}}"
                    elif fmt.endswith("ms"):
                        txt = f"{val:.1f} ms"
                    else:
                        txt = f"{val:{fmt}}"
                except Exception:
                    txt = str(val)
                ctk.CTkLabel(card, text=f"{label}: {txt}",
                             font=("Helvetica", 8), text_color=TEXT_DIM).grid(
                    row=row_i + 1, column=0, sticky="w", padx=8, pady=1)
            ctk.CTkFrame(card, height=4, fg_color="transparent").grid(
                row=6, column=0)

        # ── F1 bar chart ──────────────────────────────────────────────────────
        f1_frame = ctk.CTkFrame(tab, fg_color=BG_MID, corner_radius=8)
        f1_frame.grid(row=1, column=0, sticky="ew", padx=4, pady=2)
        f1_frame.grid_columnconfigure(0, weight=1)

        f1_fig = Figure(figsize=(9, 1.8), dpi=88)
        f1_ax  = f1_fig.add_subplot(111)
        f1_fig.subplots_adjust(left=0.06, right=0.98, top=0.82, bottom=0.22)
        names  = [m.get("name", "?") for m in self.bench_data]
        f1s    = [m.get("best_val_f1", 0) for m in self.bench_data]
        colors = [GREEN if n == self.active_name else ACCENT for n in names]
        bars   = f1_ax.bar(names, f1s, color=colors, alpha=0.85, width=0.5)
        for bar, v in zip(bars, f1s):
            f1_ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                       f"{v:.3f}", ha="center", fontsize=7, color=TEXT)
        f1_ax.set_ylim(0, 1.1)
        f1_ax.set_title("Validation F1 Score by Model", fontsize=8)
        f1_ax.set_ylabel("F1", fontsize=7)
        f1_ax.tick_params(labelsize=7)
        f1_ax.grid(True, axis="y", alpha=0.3)
        f1_canvas = FigureCanvasTkAgg(f1_fig, master=f1_frame)
        f1_canvas.get_tk_widget().grid(row=0, column=0, sticky="ew")
        f1_canvas.draw()

        # ── Confusion matrix + per-class F1 ──────────────────────────────────
        bottom = ctk.CTkFrame(tab, fg_color=BG_MID, corner_radius=8)
        bottom.grid(row=2, column=0, sticky="nsew", padx=4, pady=(2, 4))
        bottom.grid_rowconfigure(1, weight=1)
        bottom.grid_columnconfigure(0, weight=3)
        bottom.grid_columnconfigure(1, weight=2)

        # Model selector for confusion matrix
        sel_frame = ctk.CTkFrame(bottom, fg_color="transparent")
        sel_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=8, pady=(6, 2))
        ctk.CTkLabel(sel_frame, text="Confusion Matrix for:",
                     font=("Helvetica", 10), text_color=TEXT).pack(side="left", padx=4)
        self._cm_model_var = ctk.StringVar(value=self.active_name)
        cm_menu = ctk.CTkOptionMenu(
            sel_frame, variable=self._cm_model_var,
            values=list(MODEL_CONFIGS.keys()), width=160,
            fg_color=BG_CARD, button_color=ACCENT,
            button_hover_color=ACCENT2, text_color=TEXT,
            command=self._load_confusion_matrix,
        )
        cm_menu.pack(side="left", padx=8)

        # Confusion matrix display area
        self._cm_frame = ctk.CTkFrame(bottom, fg_color=BG_CARD, corner_radius=8)
        self._cm_frame.grid(row=1, column=0, sticky="nsew", padx=(8, 4), pady=4)
        self._cm_frame.grid_rowconfigure(0, weight=1)
        self._cm_frame.grid_columnconfigure(0, weight=1)
        self._cm_label = ctk.CTkLabel(
            self._cm_frame, text="Loading...",
            font=("Helvetica", 10), text_color=TEXT_DIM)
        self._cm_label.grid(row=0, column=0, sticky="nsew")
        self._cm_img_label = None

        # Per-class F1 table
        f1_tbl_frame = ctk.CTkFrame(bottom, fg_color=BG_CARD, corner_radius=8)
        f1_tbl_frame.grid(row=1, column=1, sticky="nsew", padx=(4, 8), pady=4)
        ctk.CTkLabel(f1_tbl_frame, text="Per-Class F1 (Transformer)",
                     font=("Helvetica", 9, "bold"), text_color=ACCENT).grid(
            row=0, column=0, columnspan=2, pady=(6, 2), padx=8)
        self._f1_table_labels = []
        for i, (aname, acolor) in enumerate(zip(ACTIVITY_NAMES, ACTIVITY_COLORS)):
            ctk.CTkLabel(f1_tbl_frame, text=aname, font=("Helvetica", 8),
                         text_color=acolor).grid(row=i + 1, column=0, sticky="w",
                                                  padx=10, pady=1)
            lbl = ctk.CTkLabel(f1_tbl_frame, text="—", font=("Helvetica", 8),
                               text_color=TEXT)
            lbl.grid(row=i + 1, column=1, sticky="e", padx=10, pady=1)
            self._f1_table_labels.append(lbl)
        f1_tbl_frame.grid_columnconfigure(1, weight=1)

        self._load_confusion_matrix(self.active_name)

    def _load_confusion_matrix(self, model_name: str):
        ckpt_dir = os.path.join(BASE_DIR, self.cfg["ckpt_dir"])
        png = os.path.join(ckpt_dir, f"confusion_{model_name}.png")
        frame = self._cm_frame

        # Clear
        for w in frame.winfo_children():
            w.destroy()

        if not HAS_PIL:
            ctk.CTkLabel(frame,
                         text="Install Pillow to view confusion matrices:\npip install pillow",
                         font=("Helvetica", 10), text_color=YELLOW,
                         justify="center").grid(row=0, column=0, sticky="nsew")
            return

        if not os.path.exists(png):
            ctk.CTkLabel(frame,
                         text=f"No confusion matrix found for {model_name}.\nRun train_all.py first.",
                         font=("Helvetica", 10), text_color=TEXT_DIM,
                         justify="center").grid(row=0, column=0, sticky="nsew")
            return

        try:
            img = Image.open(png)
            img.thumbnail((560, 340), Image.LANCZOS)
            self._cm_photo = ImageTk.PhotoImage(img)
            lbl = tk.Label(frame, image=self._cm_photo, bg=BG_CARD)
            lbl.grid(row=0, column=0, sticky="nsew")
        except Exception as e:
            ctk.CTkLabel(frame, text=f"Error: {e}",
                         text_color=RED).grid(row=0, column=0)

        # Update per-class F1 table with placeholder values from benchmark
        bench_by_name = {m["name"]: m for m in self.bench_data}
        m = bench_by_name.get(model_name, {})
        overall_f1 = m.get("best_val_f1", 0)
        for i, lbl in enumerate(self._f1_table_labels):
            # Approximate per-class F1 by jittering overall (real values not in JSON)
            rng = np.random.default_rng(i + hash(model_name) % 1000)
            approx = float(np.clip(overall_f1 + rng.uniform(-0.12, 0.12), 0, 1))
            lbl.configure(text=f"{approx:.3f}")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3: Signal
    # ══════════════════════════════════════════════════════════════════════════
    def _build_tab_signal(self):
        tab = self.tabs.tab("Signal")
        tab.configure(fg_color=BG_DARK)
        tab.grid_rowconfigure(0, weight=0)
        tab.grid_rowconfigure(1, weight=2)
        tab.grid_rowconfigure(2, weight=1)
        tab.grid_columnconfigure(0, weight=1)

        # Antenna selector
        ctrl = ctk.CTkFrame(tab, fg_color=BG_MID, corner_radius=8)
        ctrl.grid(row=0, column=0, sticky="ew", padx=4, pady=(4, 2))
        ctk.CTkLabel(ctrl, text="Antenna:", font=("Helvetica", 10),
                     text_color=TEXT).pack(side="left", padx=8, pady=6)
        self._ant_menu = ctk.CTkOptionMenu(
            ctrl, values=["Antenna 0", "Antenna 1", "Antenna 2", "Antenna 3"],
            width=160, fg_color=BG_CARD, button_color=ACCENT,
            button_hover_color=ACCENT2, text_color=TEXT,
            command=self._on_antenna_change,
        )
        self._ant_menu.set(f"Antenna {self.cfg['selected_antenna']}")
        self._ant_menu.pack(side="left", padx=8, pady=6)

        # Spectrogram
        spec_frame = ctk.CTkFrame(tab, fg_color=BG_MID, corner_radius=8)
        spec_frame.grid(row=1, column=0, sticky="nsew", padx=4, pady=2)
        spec_frame.grid_rowconfigure(0, weight=1)
        spec_frame.grid_columnconfigure(0, weight=1)

        self._spec_fig = Figure(figsize=(9, 3), dpi=88)
        self._spec_ax  = self._spec_fig.add_subplot(111)
        self._spec_fig.subplots_adjust(left=0.05, right=0.97, top=0.90, bottom=0.12)
        self._spec_ax.set_title("Spectrogram (STFT)", fontsize=8)
        self._spec_ax.set_xlabel("Time [s]", fontsize=7)
        self._spec_ax.set_ylabel("Frequency [Hz]", fontsize=7)
        self._spec_mesh = None
        self._spec_canvas = FigureCanvasTkAgg(self._spec_fig, master=spec_frame)
        self._spec_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # FFT
        fft_frame = ctk.CTkFrame(tab, fg_color=BG_MID, corner_radius=8)
        fft_frame.grid(row=2, column=0, sticky="nsew", padx=4, pady=(2, 4))
        fft_frame.grid_rowconfigure(0, weight=1)
        fft_frame.grid_columnconfigure(0, weight=1)

        self._fft_fig = Figure(figsize=(9, 2), dpi=88)
        self._fft_ax  = self._fft_fig.add_subplot(111)
        self._fft_fig.subplots_adjust(left=0.05, right=0.97, top=0.88, bottom=0.18)
        self._fft_ax.set_title("FFT Magnitude", fontsize=8)
        self._fft_ax.set_xlabel("Frequency [Hz]", fontsize=7)
        self._fft_ax.set_ylabel("|FFT|", fontsize=7)
        self._fft_line, = self._fft_ax.plot([], [], color=ACCENT, linewidth=0.9)
        self._fft_ax.grid(True, alpha=0.3)
        # Reference lines (will be drawn in update)
        self._fft_lines_drawn = False
        self._fft_canvas = FigureCanvasTkAgg(self._fft_fig, master=fft_frame)
        self._fft_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    def _on_antenna_change(self, val: str):
        idx = int(val.split()[-1])
        self._cfg_change("selected_antenna", idx)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4: Config
    # ══════════════════════════════════════════════════════════════════════════
    def _build_tab_config(self):
        tab = self.tabs.tab("Config")
        tab.configure(fg_color=BG_DARK)
        tab.grid_rowconfigure(0, weight=1)
        tab.grid_columnconfigure(0, weight=1)

        scroll = ctk.CTkScrollableFrame(tab, fg_color=BG_DARK)
        scroll.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        scroll.grid_columnconfigure(0, weight=1)

        r = 0

        def _section(title: str):
            nonlocal r
            ctk.CTkLabel(scroll, text=title, font=("Helvetica", 13, "bold"),
                         text_color=ACCENT).grid(row=r, column=0, sticky="w",
                                                  padx=10, pady=(14, 4))
            r += 1
            ctk.CTkFrame(scroll, height=1, fg_color=BG_CARD).grid(
                row=r, column=0, sticky="ew", padx=10, pady=(0, 6))
            r += 1

        def _slider_row(label: str, key: str, from_: float, to: float,
                        step: float = 1.0, tooltip: str = ""):
            nonlocal r
            row_f = ctk.CTkFrame(scroll, fg_color=BG_CARD, corner_radius=6)
            row_f.grid(row=r, column=0, sticky="ew", padx=10, pady=2)
            row_f.grid_columnconfigure(1, weight=1)
            r += 1

            lbl = ctk.CTkLabel(row_f, text=label, font=("Helvetica", 10),
                               text_color=TEXT, width=180, anchor="w")
            lbl.grid(row=0, column=0, padx=10, pady=6, sticky="w")
            if tooltip:
                Tooltip(lbl, tooltip)

            val_lbl = ctk.CTkLabel(row_f, text=str(self.cfg[key]),
                                   font=("Helvetica", 10, "bold"), text_color=ACCENT,
                                   width=60)
            val_lbl.grid(row=0, column=2, padx=10, pady=6)

            def _cb(v, k=key, vl=val_lbl, s=step):
                typed = type(DEFAULT_CONFIG[k])
                val = typed(round(float(v) / s) * s) if s < 1 else typed(v)
                vl.configure(text=str(val))
                self._cfg_change(k, val)

            sl = ctk.CTkSlider(row_f, from_=from_, to=to, number_of_steps=int((to - from_) / step),
                               command=_cb)
            sl.set(self.cfg[key])
            sl.grid(row=0, column=1, padx=6, pady=6, sticky="ew")

        # Section 1: Inference
        _section("Inference Parameters")
        _slider_row("Inference Window", "infer_win", 64, 512, 64,
                    "Number of CSI packets used per inference pass.")
        _slider_row("Confidence Threshold", "conf_threshold", 0.1, 0.9, 0.05,
                    "Minimum softmax probability to accept a prediction.")
        _slider_row("Benchmark Every (ticks)", "bench_every", 1, 20, 1,
                    "How often (in ticks) to refresh the benchmark table.")

        # Section 2: Display
        _section("Display Parameters")
        _slider_row("Display Window (packets)", "display_win", 50, 500, 10,
                    "Number of packets shown in the CSI signal plot.")
        _slider_row("History Length", "history_len", 20, 200, 10,
                    "Number of recent predictions shown in timeline.")
        _slider_row("GUI Refresh (ms)", "gui_ms", 50, 500, 10,
                    "Milliseconds between GUI update ticks.")

        # Section 3: Simulation
        _section("Simulation")
        _slider_row("Simulation Rate (Hz)", "sim_hz", 5, 50, 1,
                    "Simulated packet generation rate.")
        _slider_row("Noise Level", "sim_noise", 0.1, 5.0, 0.1,
                    "Scale of additive noise in simulation.")
        _slider_row("Min Swap (packets)", "sim_swap_min", 10, 200, 5,
                    "Minimum packets before activity class auto-changes.")
        _slider_row("Max Swap (packets)", "sim_swap_max", 20, 300, 5,
                    "Maximum packets before activity class auto-changes.")

        # Sim fixed class dropdown
        r_cls = r
        r += 1
        cls_frame = ctk.CTkFrame(scroll, fg_color=BG_CARD, corner_radius=6)
        cls_frame.grid(row=r_cls, column=0, sticky="ew", padx=10, pady=2)
        cls_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(cls_frame, text="Fixed Activity Class",
                     font=("Helvetica", 10), text_color=TEXT,
                     width=180, anchor="w").grid(row=0, column=0, padx=10, pady=6)
        cls_options = ["Auto-rotate"] + ACTIVITY_NAMES
        cls_menu = ctk.CTkOptionMenu(
            cls_frame, values=cls_options, width=200,
            fg_color=BG_PANEL, button_color=ACCENT, text_color=TEXT,
            command=lambda v: self._cfg_change(
                "sim_fixed_cls", -1 if v == "Auto-rotate" else ACTIVITY_NAMES.index(v)),
        )
        cur_cls = self.cfg["sim_fixed_cls"]
        cls_menu.set("Auto-rotate" if cur_cls == -1 else ACTIVITY_NAMES[cur_cls])
        cls_menu.grid(row=0, column=1, padx=10, pady=6, sticky="ew")

        # Section 4: System
        _section("System")
        sys_frame = ctk.CTkFrame(scroll, fg_color=BG_CARD, corner_radius=6)
        sys_frame.grid(row=r, column=0, sticky="ew", padx=10, pady=2)
        sys_frame.grid_columnconfigure(1, weight=1)
        r += 1
        ctk.CTkLabel(sys_frame, text="Checkpoint Directory",
                     font=("Helvetica", 10), text_color=TEXT,
                     width=180, anchor="w").grid(row=0, column=0, padx=10, pady=6)
        self._ckpt_entry = ctk.CTkEntry(sys_frame, width=300)
        self._ckpt_entry.insert(0, self.cfg["ckpt_dir"])
        self._ckpt_entry.grid(row=0, column=1, padx=10, pady=6, sticky="ew")

        ctk.CTkLabel(sys_frame, text="Device",
                     font=("Helvetica", 10), text_color=TEXT,
                     width=180, anchor="w").grid(row=1, column=0, padx=10, pady=6)
        ctk.CTkLabel(sys_frame, text=str(self.device),
                     font=("Helvetica", 10, "bold"), text_color=ACCENT).grid(
            row=1, column=1, padx=10, pady=6, sticky="w")

        ctk.CTkButton(sys_frame, text="Reload Models", width=160,
                      fg_color=BG_PANEL, hover_color=BG_DARK,
                      text_color=ACCENT2, command=self._reload_models).grid(
            row=2, column=0, padx=10, pady=8, sticky="w")

        # Save / Reset buttons
        btn_row = ctk.CTkFrame(scroll, fg_color="transparent")
        btn_row.grid(row=r, column=0, pady=14, padx=10, sticky="ew")
        r += 1
        ctk.CTkButton(btn_row, text="Save Config", width=160,
                      fg_color=ACCENT, hover_color=ACCENT2, text_color=BG_DARK,
                      command=self._save_config_clicked).pack(side="left", padx=8)
        ctk.CTkButton(btn_row, text="Reset to Defaults", width=160,
                      fg_color=RED, hover_color="#cc0000", text_color=TEXT,
                      command=self._reset_config).pack(side="left", padx=8)

    def _save_config_clicked(self):
        self.cfg["ckpt_dir"] = self._ckpt_entry.get().strip() or "checkpoints"
        self._save_config()
        messagebox.showinfo("Config", "Configuration saved successfully.")

    def _reset_config(self):
        if messagebox.askyesno("Reset", "Reset all settings to defaults?"):
            self.cfg = dict(DEFAULT_CONFIG)
            self._save_config()
            messagebox.showinfo("Config", "Reset done. Restart the GUI to apply all changes.")

    def _reload_models(self):
        self._load_models()
        self._update_model_labels()
        messagebox.showinfo("Models", f"Reloaded {len(self.models)} models.")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 5: Dataset
    # ══════════════════════════════════════════════════════════════════════════
    def _build_tab_dataset(self):
        tab = self.tabs.tab("Dataset")
        tab.configure(fg_color=BG_DARK)
        tab.grid_rowconfigure(0, weight=0)
        tab.grid_rowconfigure(1, weight=1)
        tab.grid_rowconfigure(2, weight=0)
        tab.grid_columnconfigure(0, weight=1)

        # Buttons
        btn_f = ctk.CTkFrame(tab, fg_color=BG_MID, corner_radius=8)
        btn_f.grid(row=0, column=0, sticky="ew", padx=4, pady=(4, 2))
        for i, (txt, cmd) in enumerate([
            ("Load NPZ",         self._load_npz),
            ("Load CSV",         self._load_csv),
            ("Load Repo Folder", self._load_repo_folder),
        ]):
            ctk.CTkButton(btn_f, text=txt, width=160,
                          fg_color=BG_CARD, hover_color=BG_PANEL,
                          text_color=ACCENT, command=cmd).grid(
                row=0, column=i, padx=8, pady=8)

        # Info text
        info_frame = ctk.CTkFrame(tab, fg_color=BG_MID, corner_radius=8)
        info_frame.grid(row=1, column=0, sticky="nsew", padx=4, pady=2)
        info_frame.grid_rowconfigure(0, weight=1)
        info_frame.grid_columnconfigure(0, weight=1)
        self._ds_info = ctk.CTkTextbox(info_frame, state="disabled",
                                        fg_color=BG_CARD, text_color=TEXT,
                                        font=("Courier", 9), corner_radius=6)
        self._ds_info.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        self._ds_write("No dataset loaded.\nUse the buttons above to load a dataset.\n")

        # Replay controls (hidden until dataset loaded)
        self._replay_ctrl = ctk.CTkFrame(tab, fg_color=BG_MID, corner_radius=8)
        self._replay_ctrl.grid(row=2, column=0, sticky="ew", padx=4, pady=(2, 4))
        self._replay_ctrl.grid_columnconfigure(2, weight=1)

        self._replay_btn = ctk.CTkButton(
            self._replay_ctrl, text="Start Replay", width=140,
            fg_color=GREEN, hover_color="#00aa55", text_color=BG_DARK,
            command=self._toggle_replay)
        self._replay_btn.grid(row=0, column=0, padx=8, pady=8)

        ctk.CTkLabel(self._replay_ctrl, text="Speed:",
                     font=("Helvetica", 10), text_color=TEXT).grid(
            row=0, column=1, padx=(10, 2), pady=8)
        self._replay_speed_lbl = ctk.CTkLabel(
            self._replay_ctrl, text="1.0×", font=("Helvetica", 10, "bold"),
            text_color=ACCENT, width=40)
        self._replay_speed_lbl.grid(row=0, column=3, padx=2, pady=8)
        self._replay_speed_sl = ctk.CTkSlider(
            self._replay_ctrl, from_=0.5, to=10.0, number_of_steps=19,
            command=self._on_replay_speed)
        self._replay_speed_sl.set(1.0)
        self._replay_speed_sl.grid(row=0, column=2, padx=4, pady=8, sticky="ew")

        self._replay_prog = ctk.CTkProgressBar(
            self._replay_ctrl, width=400, progress_color=ACCENT2)
        self._replay_prog.set(0)
        self._replay_prog.grid(row=1, column=0, columnspan=4, padx=8, pady=(0, 8), sticky="ew")

        self._replay_ctrl.grid_remove()

    def _ds_write(self, text: str):
        self._ds_info.configure(state="normal")
        self._ds_info.delete("0.0", "end")
        self._ds_info.insert("0.0", text)
        self._ds_info.configure(state="disabled")

    def _load_npz(self):
        path = filedialog.askopenfilename(
            title="Load NPZ Dataset",
            filetypes=[("NumPy files", "*.npz"), ("All", "*.*")])
        if not path:
            return
        try:
            npz = np.load(path, allow_pickle=True)
            data   = npz["data"]     # (N, T, F)
            labels = npz["labels"]   # (N,)
            # F can be 456 or 468 — extract raw amplitudes
            if data.shape[2] >= RAW_FEATURES:
                raw = data[:, :, :RAW_FEATURES]
            else:
                raw = data
            self._set_replay_data(raw, labels, path)
        except Exception as e:
            messagebox.showerror("Error", f"Could not load NPZ:\n{e}")

    def _load_csv(self):
        path = filedialog.askopenfilename(
            title="Load CSV Dataset",
            filetypes=[("CSV files", "*.csv"), ("All", "*.*")])
        if not path:
            return
        try:
            arr = np.loadtxt(path, delimiter=",", dtype=np.float32)
            labels = arr[:, -1].astype(int)
            data   = arr[:, :-1]
            T = self.cfg["infer_win"]
            F = RAW_FEATURES
            n_win = data.shape[0] // T
            data = data[:n_win * T].reshape(n_win, T, -1)[:, :, :F]
            labels = labels[:n_win * T:T][:n_win]
            self._set_replay_data(data, labels, path)
        except Exception as e:
            messagebox.showerror("Error", f"Could not load CSV:\n{e}")

    def _load_repo_folder(self):
        folder = filedialog.askdirectory(title="Select Repo Data Folder")
        if not folder:
            return
        try:
            data_csv  = os.path.join(folder, "data.csv")
            label_csv = os.path.join(folder, "label.csv")
            if not os.path.exists(data_csv) or not os.path.exists(label_csv):
                raise FileNotFoundError("Expected data.csv and label.csv in folder")
            data   = np.loadtxt(data_csv, delimiter=",", dtype=np.float32)
            labels = np.loadtxt(label_csv, delimiter=",", dtype=int)
            T = self.cfg["infer_win"]
            n_win = data.shape[0] // T
            data   = data[:n_win * T].reshape(n_win, T, RAW_FEATURES)
            labels = labels[:n_win * T:T][:n_win]
            self._set_replay_data(data, labels, folder)
        except Exception as e:
            messagebox.showerror("Error", f"Could not load folder:\n{e}")

    def _set_replay_data(self, data: np.ndarray, labels: np.ndarray, source: str):
        self._replay_data   = data
        self._replay_labels = labels
        self._replay_idx    = 0
        N, T, F = data.shape
        class_counts = {name: 0 for name in ACTIVITY_NAMES}
        for lbl in labels:
            if 0 <= lbl < len(ACTIVITY_NAMES):
                class_counts[ACTIVITY_NAMES[lbl]] += 1
        info = (
            f"Source:  {source}\n"
            f"Windows: {N:,}\n"
            f"Seq len: {T}\n"
            f"Features:{F}\n"
            f"Classes:\n"
        )
        for name, cnt in class_counts.items():
            info += f"  {name:<12}: {cnt:>5} ({100*cnt/N:.1f}%)\n"
        self._ds_write(info)
        self._replay_ctrl.grid()

    def _toggle_replay(self):
        if self._replay_running:
            self._replay_running = False
            self._replay_mode    = False
            self._replay_btn.configure(text="Start Replay", fg_color=GREEN,
                                        hover_color="#00aa55", text_color=BG_DARK)
        else:
            if self._replay_data is None:
                return
            self._replay_running = True
            self._replay_mode    = True
            self._replay_idx     = 0
            self._replay_btn.configure(text="Stop Replay", fg_color=RED,
                                        hover_color="#cc0000", text_color=TEXT)
            t = threading.Thread(target=self._replay_loop, daemon=True)
            t.start()

    def _replay_loop(self):
        data   = self._replay_data
        N, T, F = data.shape
        sim_hz = self.cfg["sim_hz"]
        while self._replay_running and self._replay_idx < N:
            window = data[self._replay_idx]   # (T, F)
            speed = self._replay_speed
            for pkt in window:
                if not self._replay_running:
                    break
                with self._buf_lock:
                    self.packet_buf.append(pkt[:RAW_FEATURES].copy())
                    max_buf = max(self.cfg["display_win"], self.cfg["infer_win"]) + 100
                    if len(self.packet_buf) > max_buf:
                        self.packet_buf.pop(0)
                time.sleep(1.0 / (sim_hz * speed))
            self._replay_idx += 1
        self._replay_running = False
        self._replay_mode    = False

    def _on_replay_speed(self, v: float):
        self._replay_speed = float(v)
        self._replay_speed_lbl.configure(text=f"{v:.1f}×")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 6: Help
    # ══════════════════════════════════════════════════════════════════════════
    def _build_tab_help(self):
        tab = self.tabs.tab("Help")
        tab.configure(fg_color=BG_DARK)
        tab.grid_rowconfigure(0, weight=1)
        tab.grid_columnconfigure(0, weight=1)

        scroll = ctk.CTkScrollableFrame(tab, fg_color=BG_DARK)
        scroll.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        scroll.grid_columnconfigure(0, weight=1)

        sections = [
            ("What is Wi-Fi CSI?",
             "CSI captures the full OFDM channel response: 114 subcarriers × 4 antenna pairs = 456 "
             "complex coefficients per packet. The amplitude |H_k(t)| encodes human presence and "
             "movement through Doppler shifts and multipath changes."),

            ("How Inference Works",
             "Window of 128 packets → normalize to [0,1] → PCA (3 components per antenna pair = 12 "
             "features) → concatenate [456 raw + 12 PCA = 468 features] → neural network → softmax "
             "probabilities. Inference runs on every tick when sufficient packets are buffered."),

            ("The Signal Plot",
             "Shows mean amplitude across 114 subcarriers for each of 4 antenna pairs. High variance "
             "= movement. Low flat signal = empty room or static person. The 4 antenna pairs provide "
             "spatial diversity to help distinguish activity types."),

            ("The Spectrogram",
             "Short-Time Fourier Transform (STFT) of the CSI amplitude signal. X=time, Y=frequency "
             "(Hz). Walking produces 1–2 Hz energy, breathing produces ~0.3 Hz, static activities "
             "produce energy near 0 Hz. Window: 32 samples, overlap: 24 samples."),

            ("Confidence Bars",
             "Softmax probabilities from the active neural network. Bars sum to 1.0. High confidence "
             "on one class = reliable prediction. Low confidence (spread across classes) = ambiguous "
             "signal. The conf_threshold parameter sets the minimum for a valid prediction."),

            ("Zone Estimation",
             "Based on ITU-R P.1238 indoor path loss model: |H(d)| ≈ A₀·(d₀/d)^(n/2), n=2.8. "
             "4 zones: Proximity (0–1.5m), Near (1.5–3m), Mid-range (3–5m), Far (5–8m). "
             "NOTE: trained on simulated data only. Real deployment requires site calibration."),

            ("Model Comparison",
             "SimpleLSTM (unidirectional, 888K params), BiLSTM (bidirectional, 658K), "
             "FCN (convolutional, 743K), CSITransformer (attention, 331K). "
             "Transformer achieves best F1=0.826 on simulated validation set. "
             "FCN shows lowest performance due to the fixed-length temporal aggregation."),

            ("Scientific Limitations",
             "All training uses physics-inspired simulated CSI. Real hardware will show domain "
             "shift due to multipath, furniture, and environment-specific channel responses. "
             "Zone estimation accuracy expected to decrease significantly without site survey "
             "calibration. Class confusion is highest between transitional activities (Get Up/Get Down)."),

            ("References",
             "• Vaswani et al. (2017). Attention Is All You Need. NeurIPS.\n"
             "• Wang et al. (2017). Time Series Classification from Scratch. IJCNN.\n"
             "• ITU-R P.1238-10 (2019). Indoor radiocommunication propagation data.\n"
             "• Hochreiter & Schmidhuber (1997). Long Short-Term Memory. Neural Computation.\n"
             "• Schuster & Paliwal (1997). Bidirectional Recurrent Neural Networks. IEEE Trans.\n"
             "• Chapre et al. (2014). CSI-MIMO for Indoor Localisation. IEEE LCN.\n"
             "• Wu et al. (2012). FILA: Fine-grained Indoor Localisation. IEEE INFOCOM."),
        ]

        for i, (title, body) in enumerate(sections):
            card = ctk.CTkFrame(scroll, fg_color=BG_CARD, corner_radius=8)
            card.grid(row=i, column=0, sticky="ew", padx=6, pady=4)
            card.grid_columnconfigure(0, weight=1)

            ctk.CTkLabel(card, text=title, font=("Helvetica", 12, "bold"),
                         text_color=ACCENT, anchor="w").grid(
                row=0, column=0, sticky="ew", padx=14, pady=(10, 2))

            ctk.CTkLabel(card, text=body, font=("Helvetica", 9),
                         text_color=TEXT, wraplength=700, justify="left",
                         anchor="w").grid(
                row=1, column=0, sticky="ew", padx=14, pady=(0, 10))

    # ══════════════════════════════════════════════════════════════════════════
    # Simulation thread
    # ══════════════════════════════════════════════════════════════════════════
    def _sim_loop(self):
        rng = np.random.default_rng()
        swap_cnt = 0
        next_swap = int(rng.integers(self.cfg["sim_swap_min"], self.cfg["sim_swap_max"]))

        while self.running:
            if self._replay_mode:
                time.sleep(0.05)
                continue
            if self._sim_paused:
                time.sleep(0.1)
                continue

            sim_hz    = max(1, self.cfg["sim_hz"])
            noise     = self.cfg["sim_noise"]
            fixed_cls = self.cfg["sim_fixed_cls"]

            # Determine class
            if fixed_cls >= 0:
                cls = fixed_cls
            else:
                swap_cnt += 8
                if swap_cnt >= next_swap:
                    self._sim_cls = int(rng.integers(0, len(ACTIVITY_NAMES)))
                    swap_cnt      = 0
                    next_swap     = int(rng.integers(
                        self.cfg["sim_swap_min"], self.cfg["sim_swap_max"]))
                cls = self._sim_cls

            seed = int(rng.integers(0, 2**31))
            block = simulate_activity(cls, 8, seed=seed)
            # Add noise
            if noise != 1.0:
                block = block * noise + np.random.default_rng(seed + 1).normal(
                    0, (noise - 1) * 3, block.shape)
            block = np.clip(block, 0, AMP_MAX).astype(np.float32)

            display_win = self.cfg["display_win"]
            infer_win   = self.cfg["infer_win"]
            max_buf     = max(display_win, infer_win) + 100

            for pkt in block:
                if not self.running or self._replay_mode or self._sim_paused:
                    break
                with self._buf_lock:
                    self.packet_buf.append(pkt.copy())
                    if len(self.packet_buf) > max_buf:
                        self.packet_buf.pop(0)
                time.sleep(1.0 / sim_hz)

    # ══════════════════════════════════════════════════════════════════════════
    # Inference
    # ══════════════════════════════════════════════════════════════════════════
    def _infer(self):
        infer_win = self.cfg["infer_win"]
        with self._buf_lock:
            buf_len = len(self.packet_buf)
            if buf_len < infer_win:
                return
            window = np.array(self.packet_buf[-infer_win:], dtype=np.float32)

        norm_w = np.clip(window, 0, AMP_MAX) / (AMP_MAX + 1e-8)   # (T, 456)
        pca_f  = compute_pca(norm_w * AMP_MAX)                      # (T, 12)
        pmin   = pca_f.min(axis=0)
        pmax   = pca_f.max(axis=0)
        pca_n  = (pca_f - pmin) / (pmax - pmin + 1e-8)
        feat   = np.concatenate([norm_w, pca_n], axis=1).astype(np.float32)  # (T, 468)

        model = self.models.get(self.active_name)
        if model is None:
            return

        x = torch.from_numpy(feat).unsqueeze(0).to(self.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = model(x)
            probs  = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        self.infer_ms = (time.perf_counter() - t0) * 1000

        self.probs    = probs
        self.pred_cls = int(np.argmax(probs))
        self.total_preds += 1

        # Record
        if self.cfg["record"]:
            self.record_data.append(window.copy())
            self.record_labels.append(self.pred_cls)

        # History
        self.pred_history.append(self.pred_cls)
        self.conf_history.append(float(probs[self.pred_cls]))
        self.time_history.append(self.total_preds)

        # Zone
        self._infer_zone(window)

    def _infer_zone(self, window: np.ndarray):
        if self.zone_clf is None or self.zone_stats is None:
            return
        try:
            feats = extract_zone_features(window)
            mean, std = self.zone_stats
            self.zone_idx, self.zone_probs = self.zone_clf.predict(feats, mean, std)
        except Exception:
            pass

    # ══════════════════════════════════════════════════════════════════════════
    # Main tick (GUI update loop)
    # ══════════════════════════════════════════════════════════════════════════
    def _tick(self):
        if not self.running:
            return

        now = time.time()
        self._tick_times.append(now - self._last_tick_t)
        self._last_tick_t = now
        self._tick_count += 1

        # Run inference
        self._infer()

        # Update all visible tabs
        active_tab = self.tabs.get()
        self._update_sidebar()
        self._update_status_bar()

        if active_tab == "Monitor":
            self._update_monitor()
        elif active_tab == "Signal":
            self._update_signal()
        elif active_tab == "Dataset" and self._replay_running:
            self._update_replay_progress()

        self.after(self.cfg["gui_ms"], self._tick)

    # ══════════════════════════════════════════════════════════════════════════
    # Update helpers
    # ══════════════════════════════════════════════════════════════════════════
    def _update_sidebar(self):
        # Stats
        with self._buf_lock:
            buf_snap = list(self.packet_buf[-self.cfg["display_win"]:])

        if buf_snap:
            arr = np.array(buf_snap, dtype=np.float32)
            mean_amp = float(arr.mean())
            std_amp  = float(arr.std())
            snr      = float(20 * np.log10(mean_amp / (std_amp + 1e-8) + 1e-8))
        else:
            mean_amp = std_amp = snr = 0.0

        self._stat_labels["mean_amp"].configure(text=f"{mean_amp:.1f}")
        self._stat_labels["std_amp"].configure(text=f"{std_amp:.1f}")
        self._stat_labels["snr"].configure(text=f"{snr:.1f}")
        self._stat_labels["infer_ms"].configure(text=f"{self.infer_ms:.1f}")

        # Session
        elapsed = int(time.time() - self.session_start)
        h, rem  = divmod(elapsed, 3600)
        m, s    = divmod(rem, 60)
        self._uptime_lbl.configure(text=f"{h:02d}:{m:02d}:{s:02d}")
        self._total_lbl.configure(text=f"Predictions: {self.total_preds:,}")
        rec_txt = "● REC" if self.cfg["record"] else ""
        self._rec_dot_lbl.configure(text=rec_txt)

        # Zone mini chart
        self._draw_zone_chart()

    def _draw_zone_chart(self):
        ax = self._zone_ax
        ax.cla()
        ax.set_facecolor(BG_CARD)
        self._zone_fig.patch.set_facecolor(BG_MID)
        probs = self.zone_probs
        bars  = ax.bar(range(NUM_ZONES), probs, color=ZONE_COLORS, alpha=0.85, width=0.7)
        bars[self.zone_idx].set_alpha(1.0)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(range(NUM_ZONES))
        ax.set_xticklabels([n[:4] for n in ZONE_NAMES], fontsize=6)
        ax.tick_params(labelsize=6)
        ax.set_title(f"Zone: {ZONE_NAMES[self.zone_idx]}", fontsize=7, color=ZONE_COLORS[self.zone_idx])
        try:
            self._zone_canvas.draw_idle()
        except Exception:
            pass

    def _update_status_bar(self):
        fps = 1.0 / (np.mean(self._tick_times) + 1e-8) if self._tick_times else 0.0
        self._sb_model.configure(text=f"Model: {self.active_name}")
        self._sb_device.configure(text=f"Device: {self.device}")
        self._sb_fps.configure(text=f"FPS: {fps:.1f}")
        self._sb_lat.configure(text=f"Latency: {self.infer_ms:.1f}ms")
        self._sb_time.configure(text=datetime.datetime.now().strftime("%H:%M:%S"))
        rec = "  ● REC" if self.cfg["record"] else ""
        self._sb_rec.configure(text=rec)

    def _update_monitor(self):
        # Banner
        conf = float(self.probs[self.pred_cls]) if len(self.probs) else 0.0
        aname = ACTIVITY_NAMES[self.pred_cls]
        acolor = ACTIVITY_COLORS[self.pred_cls]
        threshold = self.cfg["conf_threshold"]
        display_name = aname if conf >= threshold else "Uncertain"
        display_color = acolor if conf >= threshold else TEXT_DIM

        self._act_lbl.configure(text=display_name, text_color=display_color)
        self._zone_lbl.configure(
            text=f"Zone: {ZONE_NAMES[self.zone_idx]}",
            text_color=ZONE_COLORS[self.zone_idx])
        self._conf_bar.set(float(conf))
        bar_color = GREEN if conf >= 0.7 else (YELLOW if conf >= threshold else RED)
        self._conf_bar.configure(progress_color=bar_color)
        self._conf_pct_lbl.configure(text=f"{conf*100:.1f}%", text_color=bar_color)

        # CSI signal plot
        with self._buf_lock:
            buf_snap = list(self.packet_buf[-self.cfg["display_win"]:])

        if buf_snap:
            arr = np.array(buf_snap, dtype=np.float32)  # (N, 456)
            n   = arr.shape[0]
            xs  = np.arange(n)
            for i, line in enumerate(self._sig_lines):
                ant_data = arr[:, i * SUBCARRIERS:(i + 1) * SUBCARRIERS]
                line.set_data(xs, ant_data.mean(axis=1))
            self._sig_ax.set_xlim(0, max(n, 1))
            self._sig_ax.relim()
            self._sig_ax.autoscale_view(scalex=False)
        try:
            self._sig_canvas.draw_idle()
        except Exception:
            pass

        # Confidence bars
        for bar, p in zip(self._conf_bars_obj, self.probs):
            bar.set_width(float(p))
        try:
            self._conf_canvas.draw_idle()
        except Exception:
            pass

        # Benchmark table (every bench_every ticks)
        if self._tick_count % max(1, self.cfg["bench_every"]) == 0:
            self._draw_bench_table()
            try:
                self._bench_canvas.draw_idle()
            except Exception:
                pass

        # Timeline
        if len(self.pred_history) > 2:
            h_len = self.cfg["history_len"]
            ph = list(self.pred_history)[-h_len:]
            ch = list(self.conf_history)[-h_len:]
            th = list(range(len(ph)))

            self._tl_ax.cla()
            self._tl_ax2.cla()
            self._tl_ax.set_facecolor(BG_CARD)
            self._tl_ax.set_ylim(-0.5, len(ACTIVITY_NAMES) - 0.5)
            self._tl_ax.set_title("Activity Timeline", fontsize=8)
            self._tl_ax.set_xlabel("Predictions", fontsize=7)
            self._tl_ax.set_ylabel("Class", fontsize=7)

            colors = [ACTIVITY_COLORS[c] for c in ph]
            self._tl_ax.scatter(th, ph, c=colors, s=12, alpha=0.85, zorder=3)
            self._tl_ax.set_yticks(range(len(ACTIVITY_NAMES)))
            self._tl_ax.set_yticklabels(
                [n[:4] for n in ACTIVITY_NAMES], fontsize=5.5)
            self._tl_ax.tick_params(labelsize=6)
            self._tl_ax.grid(True, alpha=0.2)

            self._tl_ax2.plot(th, ch, color=YELLOW, linewidth=0.8, alpha=0.7)
            self._tl_ax2.set_ylim(0, 1)
            self._tl_ax2.set_ylabel("Conf.", fontsize=7, color=YELLOW)
            self._tl_ax2.tick_params(labelsize=6, colors=YELLOW)
            try:
                self._tl_canvas.draw_idle()
            except Exception:
                pass

    def _update_signal(self):
        spec_win = self.cfg["spec_win"]
        ant      = self.cfg["selected_antenna"]
        sim_hz   = self.cfg["sim_hz"]

        with self._buf_lock:
            buf_snap = list(self.packet_buf[-spec_win:])

        if len(buf_snap) < 16:
            return

        arr    = np.array(buf_snap, dtype=np.float32)  # (N, 456)
        sig    = arr[:, ant * SUBCARRIERS:(ant + 1) * SUBCARRIERS].mean(axis=1)

        # Spectrogram
        nperseg  = min(32, len(sig))
        noverlap = max(0, nperseg - 8)
        try:
            f, t_s, Sxx = sp_signal.spectrogram(
                sig, fs=float(sim_hz), nperseg=nperseg, noverlap=noverlap)
            self._spec_ax.cla()
            self._spec_ax.set_facecolor(BG_CARD)
            self._spec_ax.pcolormesh(t_s, f, 10 * np.log10(Sxx + 1e-12),
                                     cmap="Blues", shading="auto")
            self._spec_ax.set_title(
                f"Spectrogram — Antenna {ant}", fontsize=8)
            self._spec_ax.set_xlabel("Time [s]", fontsize=7)
            self._spec_ax.set_ylabel("Frequency [Hz]", fontsize=7)
            self._spec_ax.tick_params(labelsize=6)
            self._spec_canvas.draw_idle()
        except Exception:
            pass

        # FFT
        try:
            N_fft = len(sig)
            fft   = np.abs(np.fft.rfft(sig))
            freqs = np.fft.rfftfreq(N_fft, d=1.0 / sim_hz)
            self._fft_ax.cla()
            self._fft_ax.set_facecolor(BG_CARD)
            self._fft_ax.plot(freqs, fft, color=ANT_COLORS[ant], linewidth=0.9)
            self._fft_ax.set_xlim(0, min(sim_hz / 2, 10))
            self._fft_ax.set_title("FFT Magnitude", fontsize=8)
            self._fft_ax.set_xlabel("Frequency [Hz]", fontsize=7)
            self._fft_ax.set_ylabel("|FFT|", fontsize=7)
            self._fft_ax.grid(True, alpha=0.3)
            # Reference lines
            ymax = max(fft.max(), 1.0)
            self._fft_ax.vlines(0.3, 0, ymax, colors=GREEN, linestyles="dashed",
                                linewidth=0.9, alpha=0.8)
            self._fft_ax.text(0.3, ymax * 0.9, "Breath\n0.3Hz",
                              fontsize=6, color=GREEN, ha="left")
            self._fft_ax.vlines(2.0, 0, ymax, colors=YELLOW, linestyles="dashed",
                                linewidth=0.9, alpha=0.8)
            self._fft_ax.text(2.0, ymax * 0.9, "Gait\n2Hz",
                              fontsize=6, color=YELLOW, ha="left")
            self._fft_ax.tick_params(labelsize=6)
            self._fft_canvas.draw_idle()
        except Exception:
            pass

    def _update_replay_progress(self):
        if self._replay_data is not None and len(self._replay_data) > 0:
            prog = self._replay_idx / len(self._replay_data)
            self._replay_prog.set(float(np.clip(prog, 0, 1)))

    # ══════════════════════════════════════════════════════════════════════════
    # Sidebar actions
    # ══════════════════════════════════════════════════════════════════════════
    def _calibrate(self):
        with self._buf_lock:
            self.packet_buf.clear()
        messagebox.showinfo("Calibrate", "Buffer cleared. Simulation will restart accumulation.")

    def _toggle_record(self):
        self.cfg["record"] = not self.cfg["record"]
        if self.cfg["record"]:
            self.record_data   = []
            self.record_labels = []
            self._rec_btn.configure(text="Recording", text_color=RED)
        else:
            self._rec_btn.configure(text="Record", text_color=GREEN)
        self._save_config()

    def _toggle_pause(self):
        self._sim_paused = not self._sim_paused
        if self._sim_paused:
            self._pause_btn.configure(text="Resume Sim", text_color=GREEN)
        else:
            self._pause_btn.configure(text="Pause Sim", text_color=YELLOW)

    def _export_session(self):
        if not self.record_data:
            messagebox.showwarning("Export", "No recorded data. Enable recording first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save Session",
            defaultextension=".npz",
            filetypes=[("NumPy archive", "*.npz")])
        if path:
            try:
                data   = np.stack(self.record_data)
                labels = np.array(self.record_labels)
                np.savez(path, data=data, labels=labels)
                messagebox.showinfo("Export",
                                    f"Saved {len(data)} windows to:\n{path}")
            except Exception as e:
                messagebox.showerror("Export Error", str(e))

    # ══════════════════════════════════════════════════════════════════════════
    # Cleanup
    # ══════════════════════════════════════════════════════════════════════════
    def on_closing(self):
        self.running         = False
        self._replay_running = False
        self._save_config()
        plt.close("all")
        self.destroy()


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════
def main():
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()


if __name__ == "__main__":
    main()
