# Results

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
