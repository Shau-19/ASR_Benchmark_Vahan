# ============================================================
# run_benchmark.py — Main entry point
# ============================================================

import os
import json
import csv
from pathlib import Path
from config import AUDIO_DIR, RESULTS_DIR, AUDIO_FILES, GROUND_TRUTH, AUDIO_CONDITIONS, SPEAKER

from utils.transcribe_deepgram import transcribe_batch as deepgram_batch
from utils.transcribe_whisper   import transcribe_batch as whisper_batch
from utils.transcribe_sarvam    import transcribe_batch as sarvam_batch
from utils.metrics import wer, cer, entity_match, aggregate_metrics


def resolve_paths(audio_files, audio_dir):
    return {loc: str(Path(audio_dir) / fname) for loc, fname in audio_files.items()}


def evaluate(model_name, transcriptions):
    rows = []
    for locality, result in transcriptions.items():
        # strip _2 suffix to get the canonical locality name for entity matching
        canonical = locality.replace("_2", "").replace("_", " ").strip()
        gt    = GROUND_TRUTH.get(locality, "")
        hyp   = result.get("transcript", "")
        error = result.get("error")

        if error:
            rows.append({
                "model": model_name, "locality": locality,
                "canonical": canonical,
                "condition": AUDIO_CONDITIONS.get(locality, "unknown"),
                "speaker": SPEAKER.get(locality, "unknown"),
                "ground_truth": gt, "hypothesis": "",
                "wer": 1.0, "cer": 1.0,
                "entity_exact": False, "entity_fuzzy": False, "entity_ratio": 0.0,
                "latency_s": 0.0, "error": error,
            })
            continue

        em = entity_match(canonical, hyp)
        rows.append({
            "model": model_name, "locality": locality,
            "canonical": canonical,
            "condition": AUDIO_CONDITIONS.get(locality, "unknown"),
            "speaker": SPEAKER.get(locality, "unknown"),
            "ground_truth": gt, "hypothesis": hyp,
            "wer": round(wer(gt, hyp), 3),
            "cer": round(cer(gt, hyp), 3),
            "entity_exact": em["exact"],
            "entity_fuzzy": em["fuzzy"],
            "entity_ratio": em["ratio"],
            "latency_s": result.get("latency_s", 0.0),
            "error": None,
        })
    return rows


def save_raw_csv(all_rows, path):
    if not all_rows: return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n✓ Raw results saved → {path}")


def save_summary_csv(summary, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    rows = [{"model": model, **stats} for model, stats in summary.items()]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"✓ Summary saved      → {path}")


def print_summary_table(summary):
    print("\n" + "=" * 70)
    print(f"{'Model':<25} {'Avg WER':>8} {'Avg CER':>8} {'EMR Exact':>10} {'EMR Fuzzy':>10}")
    print("-" * 70)
    for model, s in summary.items():
        print(f"{model:<25} {s['avg_wer']:>8.3f} {s['avg_cer']:>8.3f} "
              f"{s['entity_exact_pct']:>9.1f}% {s['entity_fuzzy_pct']:>9.1f}%")
    print("=" * 70)


def print_condition_breakdown(all_rows):
    from collections import defaultdict
    data = defaultdict(lambda: defaultdict(list))
    for r in all_rows:
        data[r["model"]][r["condition"]].append(r["wer"])
    conditions = sorted({r["condition"] for r in all_rows})
    models = sorted(data.keys())
    print("\n── WER by Audio Condition ──")
    header = f"{'Condition':<20}" + "".join(f"{m[:15]:>16}" for m in models)
    print(header)
    print("-" * len(header))
    for cond in conditions:
        row = f"{cond:<20}"
        for model in models:
            vals = data[model][cond]
            avg = sum(vals) / len(vals) if vals else float("nan")
            row += f"{avg:>16.3f}"
        print(row)


def print_speaker_breakdown(all_rows):
    from collections import defaultdict
    data = defaultdict(lambda: defaultdict(list))
    for r in all_rows:
        data[r["model"]][r["speaker"]].append(r["wer"])
    speakers = sorted({r["speaker"] for r in all_rows})
    models = sorted(data.keys())
    print("\n── WER by Speaker ──")
    header = f"{'Speaker':<20}" + "".join(f"{m[:15]:>16}" for m in models)
    print(header)
    print("-" * len(header))
    for spk in speakers:
        row = f"{spk:<20}"
        for model in models:
            vals = data[model][spk]
            avg = sum(vals) / len(vals) if vals else float("nan")
            row += f"{avg:>16.3f}"
        print(row)


def main():
    print("=" * 60)
    print("  ASR Benchmark — Bangalore Locality Names")
    print("=" * 60)

    audio_paths = resolve_paths(AUDIO_FILES, AUDIO_DIR)

    print(f"\n[1/3] Running Deepgram Nova-2 ...")
    dg_rows  = evaluate("deepgram-nova-2",   deepgram_batch(audio_paths))

    print(f"\n[2/3] Running OpenAI Whisper-1 ...")
    wh_rows  = evaluate("whisper-1",         whisper_batch(audio_paths))

    print(f"\n[3/3] Running Sarvam saarika:v2.5 ...")
    sv_rows  = evaluate("sarvam-saarika-v2", sarvam_batch(audio_paths))

    all_rows = dg_rows + wh_rows + sv_rows

    summary = {
        "deepgram-nova-2"   : aggregate_metrics(dg_rows),
        "whisper-1"         : aggregate_metrics(wh_rows),
        "sarvam-saarika-v2" : aggregate_metrics(sv_rows),
    }

    print_summary_table(summary)
    print_condition_breakdown(all_rows)
    print_speaker_breakdown(all_rows)

    print("\n── Top Failures (highest WER per model) ──")
    for model, rows in [("deepgram-nova-2", dg_rows), ("whisper-1", wh_rows), ("sarvam-saarika-v2", sv_rows)]:
        worst = sorted(rows, key=lambda x: x["wer"], reverse=True)[:5]
        print(f"\n{model}:")
        for r in worst:
            print(f"  {r['locality']:<22} WER={r['wer']:.2f}  "
                  f"GT: '{r['ground_truth']}'  →  HYP: '{r['hypothesis']}'")

    save_raw_csv(all_rows, f"{RESULTS_DIR}/results_raw.csv")
    save_summary_csv(summary, f"{RESULTS_DIR}/summary.csv")

    raw_json_path = f"{RESULTS_DIR}/transcriptions_raw.json"
    dg_raw = {r["locality"]: r["hypothesis"] for r in dg_rows}
    wh_raw = {r["locality"]: r["hypothesis"] for r in wh_rows}
    sv_raw = {r["locality"]: r["hypothesis"] for r in sv_rows}
    with open(raw_json_path, "w", encoding="utf-8") as f:
        json.dump({"deepgram": dg_raw, "whisper": wh_raw, "sarvam": sv_raw}, f, indent=2, ensure_ascii=False)
    print(f"✓ Raw transcriptions → {raw_json_path}")
    print("\nDone ✓")


if __name__ == "__main__":
    main()