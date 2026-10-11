import json
import random
from collections import Counter
from pathlib import Path

from datasets import load_dataset

SEED = 42
N_TRAIN, N_VAL, N_TEST = 8000, 500, 1000

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"


def clean(rows):
    seen, kept, dropped = set(), [], Counter()
    for r in rows:
        q, c, a = (r[k].strip() for k in ("question", "context", "answer"))
        if not (q and c and a):
            dropped["empty_field"] += 1
            continue
        key = (q.lower(), c)
        if key in seen:
            dropped["duplicate"] += 1
            continue
        seen.add(key)
        kept.append({"question": q, "context": c, "answer": a})
    return kept, dropped


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({"id": f"{path.stem}-{i}", **r}) + "\n")


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("b-mc2/sql-create-context", split="train")
    ds.to_json(str(RAW / "sql_create_context.jsonl"))
    print(f"raw rows: {len(ds)}")

    rows, dropped = clean(ds)
    print(f"after cleaning: {len(rows)}  dropped: {dict(dropped)}")

    random.Random(SEED).shuffle(rows)
    test = rows[:N_TEST]
    val = rows[N_TEST:N_TEST + N_VAL]
    train = rows[N_TEST + N_VAL:N_TEST + N_VAL + N_TRAIN]

    write_jsonl(OUT / "train.jsonl", train)
    write_jsonl(OUT / "val.jsonl", val)
    write_jsonl(OUT / "test.jsonl", test)

    train_schemas = {r["context"] for r in train}
    overlap = sum(r["context"] in train_schemas for r in test)
    print(f"train/val/test: {len(train)}/{len(val)}/{len(test)}")
    print(f"test rows whose schema also appears in train: {overlap}/{len(test)}")

    print("\n--- 15 random train examples ---")
    for r in random.Random(0).sample(train, 15):
        print(f"Q: {r['question']}\nSCHEMA: {r['context']}\nSQL: {r['answer']}\n")


if __name__ == "__main__":
    main()
