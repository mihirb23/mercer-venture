import os
import json
import random


folder = "processed_documents"
all_files = [f for f in os.listdir(folder) if f.endswith("_cleaned.txt")]

random.shuffle(all_files)
train_files = all_files[:8]
valid_files = all_files[8:]  

def make_jsonl(txt_files, out_file, label=0):
    with open(out_file, "w", encoding="utf-8") as fout:
        for fname in txt_files:
            with open(os.path.join(folder, fname), encoding="utf-8") as f:
                text = f.read().strip()
            fout.write(json.dumps({"text": text, "label": label}) + "\n")

make_jsonl(train_files, "train.jsonl", label=0)
make_jsonl(valid_files, "valid.jsonl", label=0)
print(f"Wrote {len(train_files)} to train.jsonl and {len(valid_files)} to valid.jsonl")
