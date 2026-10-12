import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset
from evaluate import is_valid

SEED = 42
SIZES = {
    "wikisql": {"test": 600, "val": 200, "train": 5500},
    "spider": {"test": 300, "val": 100, "train": 2500},
}

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"


def source_of(context):
    return "wikisql" if context.startswith("CREATE TABLE table_") else "spider"


def clean(rows):
    seen, kept, dropped = set(), [], Counter()
    for r in rows:
        q, c, a = (r[k].strip() for k in ("question", "context", "answer"))
        if not (q and c and a):
            dropped["empty_field"] += 1
            continue
        if not is_valid(c, a):
            dropped["invalid_gold"] += 1
            continue
        key = (q.lower(), c)
        if key in seen:
            dropped["duplicate"] += 1
            continue
        seen.add(key)
        kept.append({"question": q, "context": c, "answer": a, "source": source_of(c)})
    return kept, dropped


def split_random(rows, sizes, rng):
    rng.shuffle(rows)
    n_test, n_val = sizes["test"], sizes["val"]
    return {
        "test": rows[:n_test],
        "val": rows[n_test:n_test + n_val],
        "train": rows[n_test + n_val:n_test + n_val + sizes["train"]],
    }


def split_by_schema(rows, sizes, rng):
    groups = defaultdict(list)
    for r in rows:
        groups[r["context"]].append(r)
    keys = sorted(groups)
    rng.shuffle(keys)

    out = {"test": [], "val": [], "train": []}
    pool = []
    for k in keys:
        if len(out["test"]) < sizes["test"]:
            out["test"].extend(groups[k])
        elif len(out["val"]) < sizes["val"]:
            out["val"].extend(groups[k])
        else:
            pool.extend(groups[k])
    out["test"] = out["test"][:sizes["test"]]
    out["val"] = out["val"][:sizes["val"]]
    rng.shuffle(pool)
    out["train"] = pool[:sizes["train"]]
    return out


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({"id": f"{path.stem}-{i}", **r}) + "\n")


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("b-mc2/sql-create-context", split="train")
    ds.to_json(str(RAW / "sql_create_context.jsonl"))
    rows, dropped = clean(ds)
    print(f"raw: {len(ds)}  cleaned: {len(rows)}  dropped: {dict(dropped)}")

    rng = random.Random(SEED)
    by_source = {s: [r for r in rows if r["source"] == s] for s in SIZES}
    print({s: len(v) for s, v in by_source.items()})

    parts = {"test": [], "val": [], "train": []}
    parts_by_source = {}
    for s, sizes in SIZES.items():
        fn = split_random if s == "wikisql" else split_by_schema
        parts_by_source[s] = fn(by_source[s], sizes, rng)
        for name in parts:
            parts[name].extend(parts_by_source[s][name])
    for name in parts:
        rng.shuffle(parts[name])
        write_jsonl(OUT / f"{name}.jsonl", parts[name])

    for s in SIZES:
        p = parts_by_source[s]
        train_schemas = {r["context"] for r in p["train"]}
        overlap = sum(r["context"] in train_schemas for r in p["test"])
        print(f"{s}: " + "  ".join(f"{n}={len(p[n])}" for n in ("train", "val", "test"))
              + f"  | test rows with schema seen in train: {overlap}")


if __name__ == "__main__":
    main()
