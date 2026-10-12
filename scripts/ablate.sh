#!/usr/bin/env bash
cd "$(dirname "$0")/.." && source env.sh

run() {
  name=$1; shift
  echo "== $name  ($*)"
  python src/train_sft.py --n-train 2000 --out outputs/$name "$@" > outputs/$name.log 2>&1
  grep -oE "peak_vram_gb=[0-9.]+|'train_runtime': '[0-9.]+'" outputs/$name.log
  python src/generate.py --adapter outputs/$name --out outputs/$name/preds.jsonl > /dev/null 2>&1
  python src/evaluate.py --preds outputs/$name/preds.jsonl --out outputs/$name/rows.jsonl | grep -E "^(slice|all)"
}

run abl_control --epochs 1
run abl_rank64 --epochs 1 --r 64
run abl_lora16bit --epochs 1 --no-4bit
run abl_lr2e5 --epochs 1 --lr 2e-5
run abl_2epochs --epochs 2
