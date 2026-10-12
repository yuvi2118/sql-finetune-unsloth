import argparse

from unsloth import FastLanguageModel

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

model, tok = FastLanguageModel.from_pretrained(
    model_name=args.adapter, max_seq_length=2048, load_in_4bit=True
)
model.save_pretrained_merged(args.out, tok, save_method="merged_16bit")
