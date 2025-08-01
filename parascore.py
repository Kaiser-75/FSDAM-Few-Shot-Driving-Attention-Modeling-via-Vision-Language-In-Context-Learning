import os
import json
import torch
from parascore import ParaScorer

model_files = [
    "/results/blip2_predictions_zero_shot.json",
    "/results/predictions_few_shot.json",
    "/results/predictions_zero_shot.json",
    "/results/predictions_one_shot.json"
]

scorer = ParaScorer(lang="en", model_type="bert-base-uncased")

for path in model_files:
    if not os.path.exists(path):
        print(f"[!] File not found: {path}")
        continue

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    preds   = [item["prediction"]  for item in data]
    refs    = [item["ground_truth"] for item in data]
    sources = refs  

    result = scorer.score(preds, refs, sources)
    
    if isinstance(result, tuple):
        scores = result[0]
    else:
        scores = result["parascore"]

   
    if isinstance(scores, torch.Tensor):
        avg = scores.mean().item()
    else:
        avg = sum(scores) / len(scores) if scores else 0.0

    print(f"==== {os.path.basename(path)} ====")
    print(f"ParaScore: {avg:.4f}")

