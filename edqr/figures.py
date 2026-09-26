"""Render the README figures from results files and training logs.

Usage:
    python -m edqr.figures --data-dir /path/to/XES3G5M
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import DataConfig
from .data import clean_test_sequences, load_question_modules


def plot_auc(models_dir: Path, out: Path):
    with open(models_dir / "adgkt_training_log.json", encoding="utf-8") as f:
        adgkt = json.load(f)
    with open(models_dir / "dkt_training_log.json", encoding="utf-8") as f:
        dkt = json.load(f)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot([e["epoch"] for e in adgkt["per_epoch"]], [e["val_auc"] for e in adgkt["per_epoch"]],
            "o-", label=f"ADGKT (best {adgkt['best_val_auc']:.3f})")
    ax.plot([e["epoch"] for e in dkt["per_epoch"]], [e["val_auc"] for e in dkt["per_epoch"]],
            "s--", label=f"DKT baseline (best {dkt['best_val_auc']:.3f})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation AUC")
    ax.set_title("Validation AUC — same split, causal-masked attention")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_topk(results_dir: Path, out: Path):
    with open(results_dir / "topk_evaluation_results.json", encoding="utf-8") as f:
        r = json.load(f)
    ks = sorted(int(k) for k in r["k"])
    recall = [r["k"][str(k)]["avg_recall_over_future_wrong"] for k in ks]
    precision = [r["k"][str(k)]["avg_precision"] for k in ks]
    coverage = [r["k"][str(k)]["user_coverage"] for k in ks]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ks, recall, "o-", label="recall over future wrong answers")
    ax.plot(ks, precision, "s--", label="precision@K")
    ax.plot(ks, coverage, "^:", label="user coverage")
    ax.set_xlabel("K (most-likely-wrong questions recommended)")
    ax.set_xticks(ks)
    ax.set_title(f"Error-driven Top-K (n={r['users_evaluated']} users)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_psv(results_dir: Path, out: Path, theta: str = "0.1"):
    with open(results_dir / "psv_results.json", encoding="utf-8") as f:
        r = json.load(f)
    t = r["thresholds"][theta]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    verified = t["verified_wrong"]
    missed = t["flagged_questions"] - verified
    axes[0].pie([verified, missed], labels=["answered wrong (true errors)", "answered right"],
                colors=["#ff9999", "#66b3ff"], autopct="%1.1f%%", startangle=90)
    axes[0].set_title(f"Flagged questions at θ={theta}: precision {t['precision']:.1%}")

    cov = [t["user_coverage"], 1 - t["user_coverage"]]
    axes[1].pie(cov, labels=["≥1 verified weak point", "no flag"],
                colors=["#99ff99", "#cccccc"], autopct="%1.1f%%", startangle=90)
    axes[1].set_title(f"User coverage (n={r['users_evaluated']})")

    fig.suptitle(f"Prediction–Select–Verification — lift {t['lift_over_natural']}× "
                 f"over natural error rate {r['natural_error_rate']:.1%}", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_sensitivity(results_dir: Path, out: Path):
    with open(results_dir / "psv_results.json", encoding="utf-8") as f:
        r = json.load(f)
    thetas = sorted(float(k) for k in r["thresholds"])
    precision = [r["thresholds"][str(t)]["precision"] for t in thetas]
    coverage = [r["thresholds"][str(t)]["user_coverage"] for t in thetas]
    flags = [r["thresholds"][str(t)]["flagged_questions"] for t in thetas]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax2 = ax1.twinx()
    ax1.plot(thetas, precision, "o-", color="tab:red", label="precision (share of flags truly wrong)")
    ax1.plot(thetas, coverage, "s--", color="tab:green", label="user coverage")
    ax2.plot(thetas, flags, "^:", color="tab:blue", label="flagged questions")
    ax1.set_xlabel("Threshold θ (flag when p(correct) ≤ θ)")
    ax1.set_ylabel("precision / coverage")
    ax2.set_ylabel("flagged questions")
    ax1.set_title("Precision–coverage trade-off across thresholds")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center",
               bbox_to_anchor=(0.5, -0.18), ncol=3)
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_position_effect(data_dir: Path, qid_map: dict, out: Path):
    cfg = DataConfig()
    tuo_qids, qid_to_module = load_question_modules(data_dir / "metadata" / "questions.json")
    test = clean_test_sequences(pd.read_csv(data_dir / "kc_level" / "test.csv"),
                                 set(qid_map.keys()), qid_to_module, cfg)
    error_by_pos = []
    for _, row in test.iterrows():
        rs = [int(r) for r in str(row["responses"]).split(",")]
        error_by_pos.append([1 - r for r in rs])
    n = min(len(s) for s in error_by_pos)
    rates = np.mean([s[:n] for s in error_by_pos], axis=0)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(range(1, n + 1), rates, ".-")
    ax.axhline(np.mean(rates), color="r", linestyle="--",
               label=f"mean {np.mean(rates):.1%}")
    ax.set_xlabel("Position in sequence")
    ax.set_ylabel("Error rate")
    ax.set_title("Error rate by sequence position (test users)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    plot_auc(args.models_dir, args.figures_dir / "01_validation_auc.png")
    with open(args.models_dir / "qid_map.json", encoding="utf-8") as f:
        qid_map = json.load(f)
    try:
        plot_topk(args.results_dir, args.figures_dir / "02_topk_metrics.png")
        plot_psv(args.results_dir, args.figures_dir / "03_psv_summary.png")
        plot_threshold_sensitivity(args.results_dir, args.figures_dir / "04_threshold_sensitivity.png")
    except FileNotFoundError as e:
        print(f"skipping result figures ({e}) — run edqr.evaluate first")
    plot_position_effect(args.data_dir, qid_map, args.figures_dir / "05_error_rate_by_position.png")
    print("figures written to", args.figures_dir)


if __name__ == "__main__":
    main()
