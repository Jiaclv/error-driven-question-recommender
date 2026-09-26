"""Train ADGKT or DKT.

Usage:
    python -m edqr.train --model adgkt --data-dir /path/to/XES3G5M
    python -m edqr.train --model dkt --data-dir /path/to/XES3G5M
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import DataConfig, ModelConfig, TrainConfig
from .data import DKTDataset, KTDataset, prepare_training_data
from .models import ADGKT, DKT


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate_auc(model, loader, device, model_name: str) -> float:
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            if model_name == "adgkt":
                iq, ic, ir, tq, tc, y, m = [b.to(device) for b in batch]
                p = model(iq, ic, ir, tq, tc)
            else:
                x, tgt, y, m = [b.to(device) for b in batch]
                logits = model(x)
                p = torch.sigmoid(logits.gather(2, tgt.unsqueeze(-1)).squeeze(-1))
            m_bool = m == 1
            preds.extend(p[m_bool].cpu().numpy())
            labels.extend(y[m_bool].cpu().numpy())
    return roc_auc_score(labels, preds) if preds else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["adgkt", "dkt"], default="adgkt")
    parser.add_argument("--data-dir", type=Path, required=True, help="XES3G5M dataset root")
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    args = parser.parse_args()

    set_seed(args.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg, model_cfg = DataConfig(), ModelConfig()
    print(f"Preparing data from {args.data_dir} ...")
    bundle = prepare_training_data(args.data_dir, data_cfg)
    stats = bundle["stats"]
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    qid_map, cid_map = bundle["qid_map"], bundle["cid_map"]
    num_q, num_c = len(qid_map), len(cid_map)

    max_seq_len = TrainConfig.max_seq_len
    if args.model == "adgkt":
        train_set = KTDataset(bundle["train_seqs"], max_seq_len)
        val_set = KTDataset(bundle["val_seqs"], max_seq_len)
        model = ADGKT(num_q, num_c, model_cfg.embed_size, model_cfg.hidden_size,
                      model_cfg.num_heads, model_cfg.dropout).to(device)
        criterion = nn.BCELoss()
    else:
        train_set = DKTDataset(bundle["train_seqs"], max_seq_len)
        val_set = DKTDataset(bundle["val_seqs"], max_seq_len)
        model = DKT(num_q).to(device)
        criterion = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=TrainConfig.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=TrainConfig.num_workers)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {"model": args.model, "epochs": args.epochs, "device": str(device),
               "seed": args.seed, "per_epoch": []}
    best_auc, best_state, best_epoch = 0.0, None, 1

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]"):
            optimizer.zero_grad()
            if args.model == "adgkt":
                iq, ic, ir, tq, tc, y, m = [b.to(device) for b in batch]
                p = model(iq, ic, ir, tq, tc)
            else:
                x, tgt, y, m = [b.to(device) for b in batch]
                logits = model(x)
                p = logits.gather(2, tgt.unsqueeze(-1)).squeeze(-1)
            m_bool = m == 1
            p, y = p[m_bool], y[m_bool].float()
            if p.nelement() == 0:
                continue
            loss = criterion(p, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        auc = evaluate_auc(model, val_loader, device, args.model)
        history["per_epoch"].append({"epoch": epoch,
                                     "train_loss": total_loss / max(n_batches, 1),
                                     "val_auc": auc})
        print(f"Epoch {epoch}: train_loss={total_loss / max(n_batches, 1):.4f}  val_auc={auc:.4f}")
        if auc > best_auc:
            best_auc, best_epoch = auc, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    history["best_val_auc"] = best_auc
    history["best_epoch"] = best_epoch
    history["data_stats"] = stats

    args.models_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.model
    torch.save(best_state, args.models_dir / f"{prefix}_best.pth")
    with open(args.models_dir / f"{prefix}_training_log.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    if args.model == "adgkt":
        with open(args.models_dir / "qid_map.json", "w", encoding="utf-8") as f:
            json.dump(qid_map, f)
        with open(args.models_dir / "cid_map.json", "w", encoding="utf-8") as f:
            json.dump(cid_map, f)
        with open(args.models_dir / "training_config.json", "w", encoding="utf-8") as f:
            json.dump({"TRAIN_RATIO": DataConfig.train_ratio, "MAX_SEQ_LEN": max_seq_len,
                       "EMBED_SIZE": model_cfg.embed_size, "HIDDEN_SIZE": model_cfg.hidden_size,
                       "NUM_HEADS": model_cfg.num_heads, "DROPOUT": model_cfg.dropout,
                       "EPOCHS": args.epochs, "BATCH_SIZE": args.batch_size,
                       "LEARNING_RATE": args.lr, "SEED": args.seed}, f, indent=2)

    print(f"\nBest validation AUC: {best_auc:.4f} (epoch {best_epoch})")
    print(f"Saved: {args.models_dir / f'{prefix}_best.pth'}")


if __name__ == "__main__":
    main()
