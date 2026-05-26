import os, csv
from pathlib import Path
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    "figure.dpi": 150,
})

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

MODEL_ORDER = ["deepgram-nova-2", "whisper-1", "sarvam-saarika-v2", "whisper-large-v3", "vakyansh"]
MODEL_LABELS = {
    "deepgram-nova-2"   : "Deepgram Nova-2",
    "whisper-1"         : "Whisper-1 (API)",
    "sarvam-saarika-v2" : "Sarvam v2.5",
    "whisper-large-v3"  : "Whisper LV3",
    "vakyansh"          : "Vakyansh",
}
MODEL_COLORS = {
    "deepgram-nova-2"   : "#378ADD",
    "whisper-1"         : "#1D9E75",
    "sarvam-saarika-v2" : "#D85A30",
    "whisper-large-v3"  : "#D4537E",
    "vakyansh"          : "#888780",
}

def load_rows():
    rows = []
    for f in ["results/results_raw.csv", "results/results_raw_colab.csv"]:
        if not Path(f).exists():
            print(f"  WARNING: {f} not found — skipping")
            continue
        with open(f, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                r["wer"]          = float(r["wer"])
                r["cer"]          = float(r["cer"])
                r["latency_s"]    = float(r["latency_s"])
                r["entity_exact"] = r["entity_exact"] == "True"
                r["entity_fuzzy"] = r["entity_fuzzy"] == "True"
                rows.append(r)
    return rows

def avg(lst): return sum(lst) / len(lst) if lst else 0.0

def save(fig, name):
    fig.savefig(f"{PLOTS_DIR}/{name}", bbox_inches="tight")
    plt.close()
    print(f"  Saved {name}")

# ── Plot 1: WER + CER overall ─────────────────────────────────
def plot_overall(rows):
    models = [m for m in MODEL_ORDER if any(r["model"] == m for r in rows)]
    wers = [avg([r["wer"] for r in rows if r["model"] == m]) for m in models]
    cers = [avg([r["cer"] for r in rows if r["model"] == m]) for m in models]
    x, w = np.arange(len(models)), 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - w/2, wers, w, color=[MODEL_COLORS[m] for m in models], alpha=0.9, label="WER")
    b2 = ax.bar(x + w/2, cers, w, color=[MODEL_COLORS[m] for m in models], alpha=0.45, label="CER")
    for b, v in list(zip(b1, wers)) + list(zip(b2, cers)):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([MODEL_LABELS[m] for m in models], rotation=15, ha="right")
    ax.set_ylabel("Score (lower is better)"); ax.set_title("WER and CER by model — all 24 samples"); ax.set_ylim(0, 1.4)
    ax.legend(handles=[mpatches.Patch(color="gray", alpha=0.9, label="WER"), mpatches.Patch(color="gray", alpha=0.45, label="CER")])
    fig.tight_layout(); save(fig, "01_wer_cer_overall.png")

# ── Plot 2: EMR ───────────────────────────────────────────────
def plot_emr(rows):
    models = [m for m in MODEL_ORDER if any(r["model"] == m for r in rows)]
    emr_e = [avg([r["entity_exact"] for r in rows if r["model"] == m]) * 100 for m in models]
    emr_f = [avg([r["entity_fuzzy"] for r in rows if r["model"] == m]) * 100 for m in models]
    x, w = np.arange(len(models)), 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - w/2, emr_e, w, color=[MODEL_COLORS[m] for m in models], alpha=0.9)
    b2 = ax.bar(x + w/2, emr_f, w, color=[MODEL_COLORS[m] for m in models], alpha=0.45)
    for b, v in list(zip(b1, emr_e)) + list(zip(b2, emr_f)):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([MODEL_LABELS[m] for m in models], rotation=15, ha="right")
    ax.set_ylabel("Entity Match Rate % (higher is better)"); ax.set_title("Entity Match Rate — the production metric"); ax.set_ylim(0, 30)
    ax.legend(handles=[mpatches.Patch(color="gray", alpha=0.9, label="Exact match"), mpatches.Patch(color="gray", alpha=0.45, label="Fuzzy match (≥0.75)")])
    fig.tight_layout(); save(fig, "02_emr.png")

# ── Plot 3: WER by condition ──────────────────────────────────
def plot_by_condition(rows):
    models = [m for m in MODEL_ORDER if any(r["model"] == m for r in rows)]
    conditions = sorted({r["condition"] for r in rows})
    x, w = np.arange(len(conditions)), 0.15
    fig, ax = plt.subplots(figsize=(13, 5))
    for i, m in enumerate(models):
        vals = [avg([r["wer"] for r in rows if r["model"] == m and r["condition"] == c]) for c in conditions]
        ax.bar(x + (i - len(models)/2 + 0.5) * w, vals, w, label=MODEL_LABELS[m], color=MODEL_COLORS[m], alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(conditions, rotation=20, ha="right")
    ax.set_ylabel("Avg WER (lower is better)"); ax.set_title("WER by audio condition"); ax.set_ylim(0, 1.5)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.legend(fontsize=9); fig.tight_layout(); save(fig, "03_wer_by_condition.png")

# ── Plot 4: WER by speaker ────────────────────────────────────
def plot_by_speaker(rows):
    models = [m for m in MODEL_ORDER if any(r["model"] == m for r in rows)]
    speakers = ["self", "friend", "self_modulated"]
    labels   = ["Self (n=21)", "Friend (n=2)", "Self-modulated (n=1)"]
    x, w = np.arange(len(speakers)), 0.15
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, m in enumerate(models):
        vals = [avg([r["wer"] for r in rows if r["model"] == m and r["speaker"] == s]) for s in speakers]
        ax.bar(x + (i - len(models)/2 + 0.5) * w, vals, w, label=MODEL_LABELS[m], color=MODEL_COLORS[m], alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Avg WER (lower is better)"); ax.set_title("WER by speaker — accent and voice variation"); ax.set_ylim(0, 1.5)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.legend(fontsize=9); fig.tight_layout(); save(fig, "04_wer_by_speaker.png")

# ── Plot 5: Hardest localities ────────────────────────────────
def plot_hard_localities(rows):
    loc_wer = defaultdict(list)
    for r in rows: loc_wer[r["canonical"]].append(r["wer"])
    sorted_locs = sorted(loc_wer.items(), key=lambda x: avg(x[1]), reverse=True)[:12]
    labels = [l for l, _ in sorted_locs]
    vals   = [avg(v) for _, v in sorted_locs]
    colors = ["#E24B4A" if v > 1.15 else "#BA7517" if v > 1.05 else "#378ADD" for v in vals]
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(labels[::-1], vals[::-1], color=colors[::-1], alpha=0.85)
    for bar, val in zip(bars, vals[::-1]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, f"{val:.3f}", va="center", fontsize=9)
    ax.set_xlabel("Avg WER across all models"); ax.set_title("Hardest localities — avg WER across all models")
    ax.axvline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(handles=[
        mpatches.Patch(color="#E24B4A", alpha=0.85, label="WER > 1.15 (severe)"),
        mpatches.Patch(color="#BA7517", alpha=0.85, label="WER 1.05–1.15 (hard)"),
        mpatches.Patch(color="#378ADD", alpha=0.85, label="WER < 1.05 (manageable)"),
    ], loc="lower right", fontsize=9)
    fig.tight_layout(); save(fig, "05_hardest_localities.png")

# ── Plot 6: Latency ───────────────────────────────────────────
def plot_latency(rows):
    models = [m for m in MODEL_ORDER if any(r["model"] == m for r in rows)]
    lats = [avg([r["latency_s"] for r in rows if r["model"] == m and r["latency_s"] > 0]) for m in models]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh([MODEL_LABELS[m] for m in models][::-1], lats[::-1],
                   color=[MODEL_COLORS[m] for m in models][::-1], alpha=0.85)
    for bar, val in zip(bars, lats[::-1]):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, f"{val:.2f}s", va="center", fontsize=10)
    ax.set_xlabel("Avg latency per sample (seconds)"); ax.set_title("Latency by model — lower is better for real-time telephony")
    ax.set_xlim(0, max(lats) * 1.3)
    fig.tight_layout(); save(fig, "06_latency.png")

# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    rows = load_rows()
    print(f"Loaded {len(rows)} rows | Models: {set(r['model'] for r in rows)}\n")
    plot_overall(rows)
    plot_emr(rows)
    plot_by_condition(rows)
    plot_by_speaker(rows)
    plot_hard_localities(rows)
    plot_latency(rows)
    print(f"\nAll 6 plots saved to ./{PLOTS_DIR}/")