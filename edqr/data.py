"""Data loading, filtering and dataset classes.

Pipeline (train):
    train_valid_sequences.csv
      -> keep only Extended-Thinking questions            (filter_to_module)
      -> drop users with < 120 such questions             (filter_to_module)
      -> truncate to first 120 interactions, drop users   (truncate_and_time_filter)
         whose span exceeds 90 days
      -> id maps (0 is reserved for padding)              (build_id_maps)
      -> per-user 80/20 temporal split                    (build_split_sequences)

Pipeline (evaluation):
    test.csv -> keep known questions, truncate to 120,
    drop >90-day users, re-derive module concepts         (clean_test_sequences)

ID convention: question and concept ids start at 1; 0 is padding everywhere.
This fixes the original ``mask = (target_q > 0)`` bug where the question
mapped to id 0 was silently dropped from the loss.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import DataConfig

SEQUENCE_COLS = ["questions", "concepts", "responses", "timestamps", "selectmasks", "is_repeat"]


def load_question_modules(questions_meta: Path, root_module: str = "拓展思维") -> tuple[set, dict]:
    """Map question id -> top-level module under the Extended-Thinking subtree.

    Returns (question_ids, qid_to_module) where qid_to_module[qid] is the
    first KC-tree level below ``root_module`` (e.g. "Geometry", "Counting").
    """
    with open(questions_meta, "r", encoding="utf-8") as f:
        questions = json.load(f)

    qids, qid_to_module = set(), {}
    for qid, details in questions.items():
        for route in details.get("kc_routes", []):
            parts = route.split("----")
            if parts[0] == root_module and len(parts) > 1:
                qids.add(qid)
                qid_to_module[qid] = parts[1]
                break
    return qids, qid_to_module


def _split_row(row, col: str) -> list:
    return str(row[col]).split(",")


def filter_to_module(df: pd.DataFrame, target_qids: set, min_questions: int) -> pd.DataFrame:
    """Keep only ``target_qids`` interactions; drop users below ``min_questions``."""
    target = {str(q) for q in target_qids}
    processed = []
    for _, row in df.iterrows():
        questions = _split_row(row, "questions")
        keep = [q in target for q in questions]
        if sum(keep) < min_questions:
            continue
        new_row = row.copy()
        for col in ("questions", "responses", "timestamps"):
            values = _split_row(row, col)
            if len(values) != len(questions):
                continue
            new_row[col] = ",".join(v for v, k in zip(values, keep) if k)
        processed.append(new_row)
    return pd.DataFrame(processed).reset_index(drop=True)


def truncate_and_time_filter(df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    """Truncate each user to the first ``max_length`` interactions and keep
    only users whose span fits within ``max_days``."""
    max_ms = cfg.max_days * 24 * 60 * 60 * 1000
    valid = []
    for _, row in df.iterrows():
        questions = _split_row(row, "questions")
        truncated = row.copy()
        for col in ("questions", "responses", "timestamps"):
            values = _split_row(row, col)
            truncated[col] = ",".join(values[: cfg.max_length])
        timestamps = [int(t) for t in _split_row(truncated, "timestamps") if t]
        if len(timestamps) > 1 and timestamps[-1] - timestamps[0] > max_ms:
            continue
        valid.append(truncated)
    return pd.DataFrame(valid).reset_index(drop=True)


def attach_module_concepts(df: pd.DataFrame, qid_to_module: dict) -> pd.DataFrame:
    """Replace KC-leaf concept ids with the top-level module of each question."""
    out = df.copy()
    out["concepts"] = out["questions"].apply(
        lambda qs: ",".join(qid_to_module.get(q, "Unknown") for q in str(qs).split(","))
    )
    return out


def build_id_maps(final_df: pd.DataFrame) -> tuple[dict, dict]:
    """Question/module -> contiguous ids starting at 1 (0 reserved for padding)."""
    qids = set()
    final_df["questions"].str.split(",").apply(qids.update)
    modules = set()
    final_df["concepts"].str.split(",").apply(modules.update)
    qid_map = {q: i + 1 for i, q in enumerate(sorted(qids, key=int))}
    cid_map = {c: i + 1 for i, c in enumerate(sorted(modules))}
    return qid_map, cid_map


def build_split_sequences(final_df: pd.DataFrame, qid_map: dict, cid_map: dict,
                          train_ratio: float) -> tuple[list, list]:
    """Per-user temporal split: first ``train_ratio`` of each sequence trains,
    the remainder validates. Mirrors the original pipeline."""
    train_seqs, val_seqs = [], []
    for _, row in final_df.iterrows():
        qs = _split_row(row, "questions")
        cs = _split_row(row, "concepts")
        rs = [int(r) for r in _split_row(row, "responses")]
        df = pd.DataFrame(
            {
                "question_id": [qid_map.get(q) for q in qs],
                "concept_id": [cid_map.get(c) for c in cs],
                "responses": rs,
            }
        ).dropna()
        if df.empty:
            continue
        df = df.astype(int)
        split = int(len(df) * train_ratio)
        if split > 1:
            train_seqs.append(df.iloc[:split])
        if len(df) - split > 1:
            val_seqs.append(df.iloc[split:])
    return train_seqs, val_seqs


def prepare_training_data(data_dir: Path, cfg: DataConfig) -> dict:
    """Run the full training-data pipeline and return everything training needs."""
    train_csv = Path(data_dir) / "kc_level" / "train_valid_sequences.csv"
    questions_meta = Path(data_dir) / "metadata" / "questions.json"

    tuo_qids, qid_to_module = load_question_modules(questions_meta, cfg.root_module)
    raw = pd.read_csv(train_csv)
    filtered = filter_to_module(raw, tuo_qids, cfg.min_questions)
    final_df = truncate_and_time_filter(filtered, cfg)
    final_df = attach_module_concepts(final_df, qid_to_module)
    qid_map, cid_map = build_id_maps(final_df)
    train_seqs, val_seqs = build_split_sequences(final_df, qid_map, cid_map, cfg.train_ratio)

    stats = {
        "raw_users": len(raw),
        "filtered_users": len(filtered),
        "final_users": len(final_df),
        "num_questions": len(qid_map),
        "num_modules": len(cid_map),
        "modules": sorted(cid_map),
        "train_sequences": len(train_seqs),
        "val_sequences": len(val_seqs),
    }
    return {"final_df": final_df, "qid_map": qid_map, "cid_map": cid_map,
            "train_seqs": train_seqs, "val_seqs": val_seqs, "stats": stats}


class KTDataset(Dataset):
    """ADGKT dataset: left-padded (question, concept, response) sequences.

    Item tensors all have length ``max_seq_len - 1``: inputs are the first
    len-1 interactions, targets/labels are shifted by one. ``mask`` marks the
    real (non-padded) target positions using the true sequence length.
    """

    def __init__(self, sequences: list, max_seq_len: int):
        self.sequences = sequences
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int):
        seq = self.sequences[index]
        length = min(len(seq), self.max_seq_len)
        q = np.zeros(self.max_seq_len, dtype=int)
        c = np.zeros(self.max_seq_len, dtype=int)
        r = np.zeros(self.max_seq_len, dtype=int)
        q[-length:] = seq["question_id"].values[-length:]
        c[-length:] = seq["concept_id"].values[-length:]
        r[-length:] = seq["responses"].values[-length:]

        # n_target real target positions sit at the tail after the shift-by-one
        n_target = length - 1
        mask = np.zeros(self.max_seq_len - 1, dtype=np.float32)
        mask[-n_target:] = 1.0

        return (
            torch.LongTensor(q[:-1]),
            torch.LongTensor(c[:-1]),
            torch.LongTensor(r[:-1]),
            torch.LongTensor(q[1:]),
            torch.LongTensor(c[1:]),
            torch.LongTensor(r[1:]),
            torch.FloatTensor(mask),
        )


class DKTDataset(Dataset):
    """DKT dataset: interaction id = question_id * 2 + response.

    Sequences are left-aligned (padding on the right), exactly as in the
    original implementation — the LSTM reads real interactions first.
    """

    def __init__(self, sequences: list, max_seq_len: int):
        self.sequences = sequences
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int):
        seq = self.sequences[index]
        q = seq["question_id"].values
        r = seq["responses"].values
        length = min(len(seq), self.max_seq_len)

        x = np.zeros(self.max_seq_len, dtype=int)
        target = np.zeros(self.max_seq_len, dtype=int)
        label = np.zeros(self.max_seq_len, dtype=np.float32)
        mask = np.zeros(self.max_seq_len, dtype=np.float32)
        for i in range(1, length):
            x[i - 1] = q[i - 1] * 2 + r[i - 1]
            target[i - 1] = q[i]
            label[i - 1] = r[i]
            mask[i - 1] = 1.0
        return (
            torch.LongTensor(x),
            torch.LongTensor(target),
            torch.FloatTensor(label),
            torch.FloatTensor(mask),
        )


def clean_test_sequences(test_df: pd.DataFrame, known_qids: set, qid_to_module: dict,
                         cfg: DataConfig) -> pd.DataFrame:
    """Evaluation-side cleaning, mirroring the original pipeline:

    keep only questions seen in training, truncate to ``max_length``,
    drop users whose span exceeds ``max_days``, re-derive module concepts.
    """
    max_ms = cfg.max_days * 24 * 60 * 60 * 1000
    processed = []
    for _, row in test_df.iterrows():
        questions = _split_row(row, "questions")
        keep = [q in known_qids for q in questions]
        if not any(keep):
            continue
        new_row = row.copy()
        for col in ("questions", "responses", "timestamps"):
            values = _split_row(row, col)
            if len(values) != len(questions):
                continue
            filtered = [v for v, k in zip(values, keep) if k]
            new_row[col] = ",".join(filtered[: cfg.max_length])
        timestamps = [int(t) for t in _split_row(new_row, "timestamps") if t]
        if len(timestamps) > 1 and timestamps[-1] - timestamps[0] > max_ms:
            continue
        new_row["concepts"] = ",".join(
            qid_to_module.get(q, "Unknown") for q in _split_row(new_row, "questions")
        )
        processed.append(new_row)
    return pd.DataFrame(processed).reset_index(drop=True)
