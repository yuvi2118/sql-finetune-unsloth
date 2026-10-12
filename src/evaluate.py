import argparse
from pathlib import Path
import json
import re
import sqlite3
from collections import defaultdict

import sqlglot


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_sql(text):
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, re.S | re.I)
    return (m.group(1) if m else text).strip().rstrip(";").strip()


def single_quotes(sql):
    return re.sub(r'"([^"]*)"', lambda m: "'" + m.group(1).replace("'", "''") + "'", sql)


def unquote_numbers(sql):
    return re.sub(r"'(\d+(?:\.\d+)?)'", r"\1", sql)


def normalize(sql):
    try:
        return sqlglot.parse_one(unquote_numbers(single_quotes(sql)), read="sqlite").sql(dialect="sqlite").lower()
    except Exception:
        return None


def is_format_ok(text):
    t = text.strip()
    return "```" not in t and "\n\n" not in t and re.match(r"(?i)^(select|with)\b", t) is not None


def is_valid(context, sql):
    con = sqlite3.connect(":memory:")
    try:
        con.executescript(context.lower().replace("create table", "create table if not exists"))
        con.execute("explain " + single_quotes(sql).lower())
        return True
    except Exception:
        return False
    finally:
        con.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--data", default="data/processed/test.jsonl")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    preds = {r["id"]: r["prediction"] for r in load_jsonl(args.preds)}
    rows = []
    for g in load_jsonl(args.data):
        raw = preds.get(g["id"], "")
        sql = extract_sql(raw)
        gold_norm = normalize(g["answer"])
        rows.append({
            "id": g["id"], "source": g["source"], "question": g["question"],
            "gold": g["answer"], "prediction": raw,
            "format_ok": is_format_ok(raw),
            "valid": is_valid(g["context"], sql),
            "match": gold_norm is not None and normalize(sql) == gold_norm,
        })

    groups = defaultdict(list)
    for r in rows:
        groups["all"].append(r)
        groups[r["source"]].append(r)
    print(f"{'slice':<9}{'n':>6}{'format_ok':>11}{'valid':>8}{'match':>8}")
    for name in ("all", "wikisql", "spider"):
        rs = groups[name]
        pct = lambda k: 100 * sum(r[k] for r in rs) / len(rs)
        print(f"{name:<9}{len(rs):>6}{pct('format_ok'):>10.1f}%{pct('valid'):>7.1f}%{pct('match'):>7.1f}%")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
