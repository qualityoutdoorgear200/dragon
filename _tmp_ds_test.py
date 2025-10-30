from datasets import load_dataset

ds = load_dataset("HuggingFaceFW/fineweb", streaming=True, split="train")
row = next(iter(ds))
print(row.keys())
print((row.get("text", "")[:200]).replace("\n"," "))
