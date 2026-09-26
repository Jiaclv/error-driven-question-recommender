"""Configuration dataclasses.

Defaults reproduce the original training setup (models/training_config.json):
embed 128, hidden 128, 4 attention heads, dropout 0.2, 10 epochs, batch 32,
lr 1e-3, max sequence length 120, per-user 80/20 temporal split.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    """Dataset filtering rules (XES3G5M, Extended-Thinking subtree)."""

    root_module: str = "拓展思维"          # KC-tree subtree to keep
    min_questions: int = 120              # min Extended-Thinking questions per user (train)
    max_length: int = 120                 # truncate each user's sequence to this many
    max_days: int = 90                    # keep users whose span fits this window
    train_ratio: float = 0.8              # per-user temporal split (first 80% trains)


@dataclass
class ModelConfig:
    """ADGKT hyperparameters (identical to the original run)."""

    embed_size: int = 128
    hidden_size: int = 128
    num_heads: int = 4
    dropout: float = 0.2


@dataclass
class TrainConfig:
    model: str = "adgkt"                  # "adgkt" | "dkt"
    epochs: int = 10
    batch_size: int = 32
    lr: float = 1e-3
    seed: int = 42
    max_seq_len: int = 120                # padded sequence length (inputs are len-1)
    num_workers: int = 0


@dataclass
class EvalConfig:
    """Prediction-Select-Verification evaluation settings."""

    thresholds: list = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    topk_list: list = field(default_factory=lambda: [10, 20, 30, 40, 50])
    history_ratio: float = 0.5            # top-k eval: first half is history, second half ground truth
    pred_batch_size: int = 256            # candidate-question batch for top-k scoring
    row_batch_size: int = 128             # PSV: per-step history rows batched into one forward


@dataclass
class Paths:
    """Default repository layout (all overridable from the CLI)."""

    repo_root: Path = Path(__file__).resolve().parent.parent

    @property
    def data_dir(self) -> Path:
        return self.repo_root / "data" / "XES3G5M"

    @property
    def models_dir(self) -> Path:
        return self.repo_root / "models"

    @property
    def results_dir(self) -> Path:
        return self.repo_root / "results"

    @property
    def figures_dir(self) -> Path:
        return self.repo_root / "figures"

    @property
    def train_sequences(self) -> Path:
        return self.data_dir / "kc_level" / "train_valid_sequences.csv"

    @property
    def test_sequences(self) -> Path:
        return self.data_dir / "kc_level" / "test.csv"

    @property
    def questions_meta(self) -> Path:
        return self.data_dir / "metadata" / "questions.json"
