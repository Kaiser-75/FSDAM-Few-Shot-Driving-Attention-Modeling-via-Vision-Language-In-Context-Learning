import os
import json
import torch
from sentence_transformers import SentenceTransformer, util

model_files = [
    "/results/blip2_predictions_zero_shot.json",
    "/results/predictions_few_shot.json",
    "/results/predictions_zero_shot.json",
    "/results/predictions_one_shot.json"
]

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("all-mpnet-base-v2", device=device)

for path in model_files:
    if not os.path.exists(path):
        continue

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    preds = [item["prediction"] for item in data]
    gts   = [item["ground_truth"] for item in data]

    emb_pred = model.encode(preds, batch_size=32, convert_to_tensor=True)
    emb_gt   = model.encode(gts,   batch_size=32, convert_to_tensor=True)

    sims = util.cos_sim(emb_pred, emb_gt).diagonal().tolist()
    avg_sim = sum(sims) / len(sims)

    print(f"{os.path.basename(path)}: SIM = {avg_sim:.4f}")
