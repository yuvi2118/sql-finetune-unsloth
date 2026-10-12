# Text-to-SQL with QLoRA: fine-tuning Llama-3.2-3B-Instruct (Unsloth)

Fine-tuning a 3B instruction model to turn a schema and a question into one SQL query, with a small, honest evaluation harness.
On 900 held-out questions, normalized-SQL exact match goes from **57.6% (zero-shot) to 84.0%** after LoRA fine-tuning on 8,000 examples
(WikiSQL-style 65.5 → 89.2, Spider-style 41.7 → 73.7). Training the full run takes about 15 minutes on a single RTX A6000.

## Results

Test set: 900 questions never used for training or model selection (600 WikiSQL-style, 300 Spider-style). One run per row (seed 42).

| model | slice | n | format_ok | valid | match |
|---|---|---|---|---|---|
| base Llama-3.2-3B, zero-shot | all | 900 | 100.0 | 89.6 | 57.6 |
| base Llama-3.2-3B, zero-shot | wikisql | 600 | 100.0 | 87.5 | 65.5 |
| base Llama-3.2-3B, zero-shot | spider | 300 | 100.0 | 93.7 | 41.7 |
| LoRA SFT, 500 examples, 1 epoch, r=16 | all | 900 | 100.0 | 97.7 | 71.2 |
| LoRA SFT, 500 examples, 1 epoch, r=16 | wikisql | 600 | 100.0 | 98.5 | 78.8 |
| LoRA SFT, 500 examples, 1 epoch, r=16 | spider | 300 | 100.0 | 96.0 | 56.0 |
| LoRA SFT, 8000 examples, 2 epochs, r=16 | all | 900 | 100.0 | 99.6 | 84.0 |
| LoRA SFT, 8000 examples, 2 epochs, r=16 | wikisql | 600 | 100.0 | 100.0 | 89.2 |
| LoRA SFT, 8000 examples, 2 epochs, r=16 | spider | 300 | 100.0 | 98.7 | 73.7 |

Ablations (2,000 training examples, QLoRA unless noted, all 900 test questions):

| run | match | valid | peak VRAM (GB) | train time (s) |
|---|---|---|---|---|
| control: r=16, lr 2e-4, 1 epoch | 75.3 | 99.1 | 3.5 | 117 |
| rank 64 | 79.2 | 99.1 | 4.1 | 123 |
| 16-bit LoRA (no 4-bit base) | 75.4 | 98.1 | 7.6 | 108 |
| lr 2e-5 | 67.7 | 97.1 | 3.5 | 117 |
| 2 epochs | 78.4 | 98.9 | 3.5 | 230 |

How much to trust these numbers: with n = 900 the 95% interval is roughly ±2.5 to ±3 points around a score of 75-85%, and each row is a single seed.
So the large effects (zero-shot vs fine-tuned, the lr 2e-5 drop, more data) are clear; differences of a couple of points (for example control vs 16-bit LoRA) are not.
`reports/results.md` holds the same tables; per-example predictions are not committed.

## What the metrics mean

| metric | definition |
|---|---|
| `format_ok` | the reply is a bare `SELECT`/`WITH` query: no code fence, no blank line, nothing else |
| `valid` | the extracted SQL parses and plans in SQLite (`EXPLAIN`) against the question's schema, with empty tables |
| `match` | the prediction and the gold query are equal after normalization with `sqlglot` (SQLite dialect, double quotes read as string literals, quoted numbers read as numbers, lowercased) |

`match` is a normalized string comparison. It is **not** execution accuracy and **not** the official Spider evaluation, so it cannot be compared with leaderboard numbers.
Two consequences: equivalent queries written differently count as misses, and because the whole query is lowercased, string literals compare case-insensitively.
`valid` only says the query is well-formed for the schema, not that it answers the question.

## Data and splits

Source: [`b-mc2/sql-create-context`](https://huggingface.co/datasets/b-mc2/sql-create-context) (question, `CREATE TABLE` context, SQL answer; derived from WikiSQL and Spider; CC BY 4.0 per its dataset card).

1. Clean: drop rows with an empty field, rows whose gold SQL fails to plan against its own schema, and duplicate (question, schema) pairs.
2. Label each row `wikisql` or `spider` from its schema text.
3. Split with a fixed seed (42). WikiSQL-style rows: random split. Spider-style rows: **grouped by exact schema text**, so no test or validation schema appears in training.

| split | wikisql | spider | total |
|---|---|---|---|
| train | 5,500 | 2,500 | 8,000 |
| val | 200 | 100 | 300 |
| test | 600 | 300 | 900 |

Checks printed by `src/data.py` and re-verified: 0 exact (question, schema) overlaps between train and val/test, and 0 of 300 Spider test schemas present in train.
Limitation: grouping is by exact schema string, not by source database, so the same database could appear with different table subsets in different splits. The dataset has no database id to rule this out.
The validation split is used only to monitor loss during training; no checkpoint selection or tuning is done on the test split.

## Method

- Base model: `unsloth/Llama-3.2-3B-Instruct`, 4-bit (QLoRA) via Unsloth.
- LoRA on all attention and MLP projections (q, k, v, o, gate, up, down); r=16, alpha=r, dropout 0; Unsloth gradient checkpointing.
- Chat-format prompt (system instruction, then schema and question); the loss is computed on the assistant reply only (`train_on_responses_only`).
- AdamW 8-bit, lr 2e-4, linear decay, 5 warmup steps, weight decay 0.01, batch 8 × 2 gradient accumulation, max length 2048, seed 42.
- Decoding: greedy, up to 256 new tokens, batch inference sorted by prompt length.
- Hardware and versions of the reported runs: one NVIDIA RTX A6000 (48 GB), Python 3.11, torch 2.14.1 (CUDA 13), Unsloth 2026.10.3, transformers 5.17.0. The full run (8,000 examples, 2 epochs) took 884 s; final validation loss 0.041.

## Repository layout

```
src/data.py         download, clean and split the dataset
src/prompts.py      the prompt template shared by training and inference
src/train_sft.py    QLoRA fine-tuning
src/generate.py     batch generation from the base model or an adapter
src/evaluate.py     format / validity / match scoring and per-slice report
src/export.py       merge an adapter into a 16-bit model
scripts/ablate.sh   the five ablation runs
reports/results.md  results tables
tests/              unit tests for the scorer
env.sh              cache locations and environment setup
requirements.lock   pinned dependencies
```

## Reproduce

Create a Python 3.11 environment with the pinned dependencies from `requirements.lock`, then run everything from the repository root.

```bash
source env.sh                      # cache dirs inside the project, activates .venv if present
python src/data.py                 # writes data/processed/{train,val,test}.jsonl

# zero-shot baseline
python src/generate.py --out outputs/baseline/preds.jsonl
python src/evaluate.py --preds outputs/baseline/preds.jsonl --out outputs/baseline/rows.jsonl

# full fine-tune, then evaluate the adapter
python src/train_sft.py --n-train 8000 --epochs 2 --out outputs/run_full
python src/generate.py --adapter outputs/run_full --out outputs/run_full/preds.jsonl
python src/evaluate.py --preds outputs/run_full/preds.jsonl --out outputs/run_full/rows.jsonl

bash scripts/ablate.sh             # the five ablations (training alone takes 2 to 4 minutes each on an A6000)
python tests/test_evaluate.py      # scorer unit tests
```

## Known limitations

- Single seed per configuration, no confidence intervals beyond the rough estimate above.
- Normalized-SQL match instead of execution accuracy; the Spider-style slice is a subset derived through `sql-create-context`, not the official Spider dev set.
- `src/generate.py` always loads the model in 4-bit, so the 16-bit-LoRA ablation was scored with 4-bit inference. The effect of that mismatch was not measured.
- No hyperparameter search beyond the five ablations; the test set was used only for the reported evaluations.
- The `src/*.py` scripts import each other as top-level modules (`from prompts import ...`) and are meant to be run as scripts. `src/evaluate.py` shares its name with the Hugging Face `evaluate` package.

## Licenses and attribution

- The code in this repository is released under the [MIT License](LICENSE). The base model and the dataset are covered by their own licenses, below.
- The base model, Llama 3.2, is released by Meta under the [Llama 3.2 Community License](https://www.llama.com/llama3_2/license/). It is **not included in this repository**, and neither are fine-tuned weights; downloading and using the model is subject to that license.
- The dataset is not redistributed here. `sql-create-context` is listed as CC BY 4.0 on its dataset card and is built from WikiSQL and Spider, which have their own licenses.
