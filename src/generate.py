import argparse
import json
from pathlib import Path

from unsloth import FastLanguageModel

import torch
from prompts import build_messages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Llama-3.2-3B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--data", default="data/processed/test.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    model, tok = FastLanguageModel.from_pretrained(
        model_name=args.adapter or args.model, max_seq_length=2048, load_in_4bit=True
    )
    FastLanguageModel.for_inference(model)
    tok.padding_side = "left"

    with open(args.data) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    rows = rows[: args.limit]

    prompts = [
        tok.apply_chat_template(
            build_messages(r["context"], r["question"]), tokenize=False, add_generation_prompt=True
        )
        for r in rows
    ]
    order = sorted(range(len(rows)), key=lambda i: len(prompts[i]))

    preds = {}
    for start in range(0, len(order), args.batch_size):
        idx = order[start : start + args.batch_size]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True,
                  add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, use_cache=True)
        texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for i, t in zip(idx, texts):
            preds[rows[i]["id"]] = t.strip()
        print(f"{len(preds)}/{len(rows)}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps({"id": r["id"], "prediction": preds[r["id"]]}) + "\n")


if __name__ == "__main__":
    main()
