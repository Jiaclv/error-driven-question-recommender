"""Evaluation: Prediction-Select-Verification (PSV) and error-driven Top-K.

PSV (per user, sequential rolling):
    For every step t = 1..L-1 the model sees only interactions [:t] and
    scores the *next* question q_t. If p(correct) <= theta the question is
    flagged ("predicted error"). After the pass, flags are verified against
    the student's actual response at t.

    Implementation note: the original notebook looped step-by-step with a
    fresh forward per step. Here all steps of one user are batched into a
    single forward (each row is exactly the left-padded history that the
    step-by-step loop would have fed), so every threshold in the sweep is
    evaluated from the same cached probability vector.

Top-K (error-driven ranking):
    History = first ``history_ratio`` of the user's filtered sequence;
    candidates = every question not seen in history; rank by predicted
    correctness ASCENDING (most-likely-wrong first — the project's thesis)
    and check the top-K against the user's actual future answers.

Reported baselines:
    natural_error_rate  — share of actually-wrong answers over all evaluated
                          positions (the "blind pick" baseline for PSV)
    lift                — precision / natural_error_rate

Usage:
    python -m edqr.evaluate --mode psv   --data-dir ... --theta 0.1
    python -m edqr.evaluate --mode sweep --data-dir ...
    python -m edqr.evaluate --mode topk  --data-dir ...
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .config import DataConfig, EvalConfig, Paths, TrainConfig
from .data import clean_test_sequences, load_question_modules
from .models import ADGKT


def load_model(models_dir: Path, device) -> tuple[ADGKT, dict, dict, dict]:
    with open(models_dir / "qid_map.json", encoding="utf-8") as f:
        qid_map = json.load(f)
    with open(models_dir / "cid_map.json", encoding="utf-8") as f:
        cid_map = json.load(f)
    with open(models_dir / "training_config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    model = ADGKT(len(qid_map), len(cid_map), cfg["EMBED_SIZE"], cfg["HIDDEN_SIZE"],
                  cfg["NUM_HEADS"], cfg["DROPOUT"]).to(device)
    state = torch.load(models_dir / "adgkt_best.pth", map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, qid_map, cid_map, cfg


def prepare_eval_users(data_dir: Path, qid_map: dict, cfg: DataConfig) -> pd.DataFrame:
    tuo_qids, qid_to_module = load_question_modules(data_dir / "metadata" / "questions.json",
                                                    cfg.root_module)
    test_raw = pd.read_csv(data_dir / "kc_level" / "test.csv")
    test_df = clean_test_sequences(test_raw, set(qid_map.keys()), qid_to_module, cfg)
    print(f"test users: {len(test_raw)} raw -> {len(test_df)} retained")
    return test_df


def predict_next_probs(model, q_seq, c_seq, r_seq, qid_map, cid_map, device,
                       max_len: int, row_batch: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """p(correct) for every valid step t=1..L-1 of one user — single pass.

    One teacher-forced forward over the user's real interactions (no padding):
    thanks to the causal attention mask, the output at position t is exactly
    "history before t scores the question at t". This matches the training
    input format (all training sequences are full-length), which matters —
    a left-padded prefix is out-of-distribution for the GRU and poisons the
    predictions (measured: 44% of steps flagged at θ=0.1 with padding vs 3%
    without). Returns (probs, actual_responses) aligned to the same steps.
    """
    q_ids = [qid_map.get(q) for q in q_seq]
    c_ids = [cid_map.get(c) for c in c_seq]
    valid = [(q, c, r) for q, c, r in zip(q_ids, c_ids, r_seq) if q is not None and c is not None]
    if len(valid) < 2:
        return np.array([]), np.array([])
    q_ids = [v[0] for v in valid][-max_len:]
    c_ids = [v[1] for v in valid][-max_len:]
    r_ids = [v[2] for v in valid][-max_len:]

    iq = torch.LongTensor([q_ids[:-1]]).to(device)
    ic = torch.LongTensor([c_ids[:-1]]).to(device)
    ir = torch.LongTensor([r_ids[:-1]]).to(device)
    tq = torch.LongTensor([q_ids[1:]]).to(device)
    tc = torch.LongTensor([c_ids[1:]]).to(device)
    with torch.no_grad():
        probs = model(iq, ic, ir, tq, tc)[0].cpu().numpy()
    return probs, np.array(r_ids[1:], dtype=int)


def run_psv(model, test_df, qid_map, cid_map, device, eval_cfg: EvalConfig,
            max_len: int, thresholds: list, verbose: bool = True) -> dict:
    """Prediction-Select-Verification over all users and thresholds."""
    per_user = []          # cached (probs, actual_response) per user
    natural_wrong, natural_total = 0, 0

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="scoring", disable=not verbose):
        q_seq = str(row["questions"]).split(",")
        c_seq = str(row["concepts"]).split(",")
        r_seq = [int(r) for r in str(row["responses"]).split(",")]
        if len(q_seq) < 2:
            continue

        probs, actual = predict_next_probs(model, q_seq, c_seq, r_seq, qid_map, cid_map,
                                           device, max_len, eval_cfg.row_batch_size)
        if len(probs) == 0:
            continue
        per_user.append((probs, actual))
        natural_wrong += int((actual == 0).sum())
        natural_total += len(actual)

    natural_error_rate = natural_wrong / natural_total if natural_total else 0.0
    results = {
        "users_evaluated": len(per_user),
        "positions_evaluated": natural_total,
        "natural_error_rate": round(natural_error_rate, 4),
        "thresholds": {},
    }
    for theta in thresholds:
        flagged_total = verified_wrong = users_with_hits = 0
        hits_per_user = []
        for probs, actual in per_user:
            flags = probs <= theta
            n_flag = int(flags.sum())
            n_true = int((flags & (actual == 0)).sum())
            flagged_total += n_flag
            verified_wrong += n_true
            hits_per_user.append(n_flag)
            if n_flag > 0:
                users_with_hits += 1
        precision = verified_wrong / flagged_total if flagged_total else 0.0
        results["thresholds"][str(theta)] = {
            "flagged_questions": flagged_total,
            "verified_wrong": verified_wrong,
            "precision": round(precision, 4),
            "users_with_at_least_one_flag": users_with_hits,
            "user_coverage": round(users_with_hits / max(len(per_user), 1), 4),
            "avg_flags_per_user": round(float(np.mean(hits_per_user)), 2) if hits_per_user else 0,
            "lift_over_natural": round(precision / natural_error_rate, 2) if natural_error_rate else None,
        }
    return results


def run_topk(model, test_df, qid_map, cid_map, device, eval_cfg: EvalConfig,
             max_len: int) -> dict:
    """Error-driven Top-K ranking evaluation (candidates = unseen questions)."""
    inverse_qid = {v: k for k, v in qid_map.items()}
    all_q_indices = set(qid_map.values())

    q_to_c = {}
    for _, row in test_df.iterrows():
        for q, c in zip(str(row["questions"]).split(","), str(row["concepts"]).split(",")):
            q_to_c.setdefault(q, c)

    per_user_rows = []
    k_metrics = {k: {"hits": 0, "recall_sum": 0.0, "precision_sum": 0.0,
                     "covered_users": 0} for k in eval_cfg.topk_list}
    n_users = 0

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="topk"):
        q_seq = str(row["questions"]).split(",")
        r_seq = [int(r) for r in str(row["responses"]).split(",")]
        split = int(len(q_seq) * eval_cfg.history_ratio)
        if split == 0:
            continue
        hist = [(qid_map.get(q), int(r)) for q, r in zip(q_seq[:split], r_seq[:split])]
        hist = [(q, r) for q, r in hist if q is not None]
        if not hist:
            continue
        future = [(q, r) for q, r in zip(q_seq[split:], r_seq[split:])]
        future_wrong = {q for q, r in future if r == 0}
        if not future_wrong:
            continue

        hist_q = [q for q, _ in hist][-max_len - 1:]
        hist_c = [cid_map.get(q_to_c.get(inverse_qid[q]), 0) or 0 for q in hist_q]
        hist_r = [r for _, r in hist][-max_len - 1:]
        width = len(hist_q)  # no padding: keep the training-format real history

        answered = set(hist_q)
        candidates = sorted(all_q_indices - answered)

        bq = torch.LongTensor([hist_q]).to(device)
        bc = torch.LongTensor([hist_c]).to(device)
        br = torch.LongTensor([hist_r]).to(device)
        predictions = []
        for start in range(0, len(candidates), eval_cfg.pred_batch_size):
            batch = candidates[start:start + eval_cfg.pred_batch_size]
            orig = [inverse_qid[i] for i in batch]
            cons = [cid_map.get(q_to_c.get(o), 0) or 0 for o in orig]
            tq = torch.LongTensor(batch).unsqueeze(1).expand(-1, width).to(device)
            tc = torch.LongTensor(cons).unsqueeze(1).expand(-1, width).to(device)
            with torch.no_grad():
                p = model(bq.expand(len(batch), -1), bc.expand(len(batch), -1),
                          br.expand(len(batch), -1), tq, tc)
            predictions.extend(zip(p[:, -1].cpu().numpy().tolist(), orig))

        # most-likely-WRONG first: ascending predicted correctness
        predictions.sort(key=lambda x: x[0])
        n_users += 1
        for k in eval_cfg.topk_list:
            if len(predictions) < k:
                continue
            top = {qid for _, qid in predictions[:k]}
            hits = len(top & future_wrong)
            k_metrics[k]["hits"] += hits
            k_metrics[k]["recall_sum"] += hits / len(future_wrong)
            k_metrics[k]["precision_sum"] += hits / k
            if hits > 0:
                k_metrics[k]["covered_users"] += 1

    out = {"users_evaluated": n_users, "k": {}}
    for k, m in k_metrics.items():
        out["k"][str(k)] = {
            "avg_recall_over_future_wrong": round(m["recall_sum"] / max(n_users, 1), 4),
            "avg_precision": round(m["precision_sum"] / max(n_users, 1), 4),
            "user_coverage": round(m["covered_users"] / max(n_users, 1), 4),
            "total_hits": m["hits"],
        }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["psv", "sweep", "topk"], default="sweep")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--theta", type=float, default=0.1, help="flag threshold for --mode psv")
    args = parser.parse_args()

    paths = Paths()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, qid_map, cid_map, _ = load_model(args.models_dir, device)
    data_cfg, eval_cfg = DataConfig(), EvalConfig()
    test_df = prepare_eval_users(args.data_dir, qid_map, data_cfg)
    max_len = TrainConfig.max_seq_len

    args.results_dir.mkdir(parents=True, exist_ok=True)
    if args.mode in ("psv", "sweep"):
        thresholds = [args.theta] if args.mode == "psv" else eval_cfg.thresholds
        results = run_psv(model, test_df, qid_map, cid_map, device, eval_cfg, max_len, thresholds)
        out_file = args.results_dir / ("psv_results.json" if args.mode == "sweep" else "psv_single.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"saved -> {out_file}")
    else:
        results = run_topk(model, test_df, qid_map, cid_map, device, eval_cfg, max_len)
        out_file = args.results_dir / "topk_evaluation_results.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"saved -> {out_file}")


if __name__ == "__main__":
    main()
