import argparse
import json

from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only

from datasets import Dataset
from trl import SFTConfig, SFTTrainer

from prompts import build_messages


def load_rows(path, n=None):
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return rows[:n]


def to_dataset(rows, tok):
    texts = [
        tok.apply_chat_template(
            build_messages(r["context"], r["question"]) + [{"role": "assistant", "content": r["answer"]}],
            tokenize=False,
        )
        for r in rows
    ]
    return Dataset.from_list([{"text": t} for t in texts])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Llama-3.2-3B-Instruct")
    ap.add_argument("--n-train", type=int, default=500)
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model, tok = FastLanguageModel.from_pretrained(
        model_name=args.model, max_seq_length=2048, load_in_4bit=True
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.r,
        lora_alpha=args.r,
        lora_dropout=0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    train_ds = to_dataset(load_rows("data/processed/train.jsonl", args.n_train), tok)
    val_ds = to_dataset(load_rows("data/processed/val.jsonl"), tok)

    config = SFTConfig(
        output_dir=args.out,
        dataset_text_field="text",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="linear",
        warmup_steps=5,
        optim="adamw_8bit",
        weight_decay=0.01,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=20,
        save_strategy="no",
        report_to="none",
        seed=42,
    )
    try:
        trainer = SFTTrainer(model=model, processing_class=tok, train_dataset=train_ds,
                             eval_dataset=val_ds, args=config)
    except TypeError:
        trainer = SFTTrainer(model=model, tokenizer=tok, train_dataset=train_ds,
                             eval_dataset=val_ds, args=config)

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    )
    trainer.train()

    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    with open(f"{args.out}/log_history.json", "w") as f:
        json.dump(trainer.state.log_history, f, indent=1)


if __name__ == "__main__":
    main()
