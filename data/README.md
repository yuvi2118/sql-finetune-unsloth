# data/

Nothing in this folder is committed except this file. Generate the data with:

```bash
python src/data.py
```

| path | contents |
|---|---|
| `data/raw/sql_create_context.jsonl` | the raw `b-mc2/sql-create-context` download |
| `data/processed/{train,val,test}.jsonl` | the cleaned splits (8,000 / 300 / 900 rows, seed 42) |

Each processed row is one JSON object per line:

```json
{"id": "train-0", "question": "...", "context": "CREATE TABLE ...", "answer": "SELECT ...", "source": "wikisql"}
```

`source` is `wikisql` or `spider`, inferred from the schema text. See the main README for the cleaning and splitting rules.
