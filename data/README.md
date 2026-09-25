# Dataset — XES3G5M

This repository does **not** ship the dataset. XES3G5M is a large-scale,
publicly available knowledge-tracing dataset published by the AI4ED group:

> Liu *et al.* **[XES3G5M: A Knowledge Tracing Benchmark Dataset with Auxiliary Information](https://github.com/ai4ed/XES3G5M)** (NeurIPS 2023 Datasets & Benchmarks)

## Download

Clone or download the dataset from the official repository:

```bash
git clone https://github.com/ai4ed/XES3G5M.git
```

(or download the archive from the same page — the full dataset is ~8 GB).

## Required layout

Only two subsets are needed by this project. Place them under `data/XES3G5M/`
relative to the repository root, so the notebook's relative paths resolve:

```
data/XES3G5M/
├── kc_level/
│   ├── train_valid_sequences.csv      # KC-level interaction sequences (training/valid)
│   └── test.csv                       # KC-level interaction sequences (test, used for cold start)
└── metadata/
    ├── questions.json                 # question metadata incl. KC routes
    └── kc_routes_map.json             # KC hierarchy routes
```

The remaining files of the official release (`question_level/`, embeddings,
images, `test_question_window_sequences.csv`) are not used by the pipeline and
can be skipped to save space.

`data/XES3G5M/metadata/kc_tree_with_qids.json` is **generated automatically**
by the notebook (section 1.1.4, *Build KC Tree Structure*) the first time you
run it — you do not need to obtain it separately.

## Preprocessing summary

The notebook applies the following filtering to the KC-level data (all numbers
reproducible from the notebook outputs):

| Step | Effect |
|---|---|
| Restrict to the “Extended Thinking” (思维拓展) top-level module | 33,397 → 33,392 user rows; 6,940 unique questions |
| Keep users with ≥ 120 answered questions | drops short/inactive users |
| Keep users whose answering window spans ≤ 90 days | **final: 10,382 users** |
| Aggregate fine-grained KCs into 7 top-level modules | 5,772 questions kept after mapping |

Mean historical accuracy of the retained users is 80.4%, i.e. a base error
rate of ≈ 20% — the reference point the recommender is evaluated against.
