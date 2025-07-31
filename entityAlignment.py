import os
import json
import spacy
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# ----- Configuration -----
JSON_PATH = "/content/blip2_predictions_zero_shot.json"  
EMBED_MODEL = "all-mpnet-base-v2"
COSINE_THRESHOLD = 0.75
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----- Load NLP & Embedding Models -----
nlp = spacy.load("en_core_web_sm")
embed = SentenceTransformer(EMBED_MODEL, device=DEVICE)

def extract_noun_chunks(text):
    """Lowercase & strip determiners from each noun-chunk."""
    doc = nlp(text)
    chunks = []
    for np in doc.noun_chunks:
        span = np.text.lower().strip()
        if span.split()[0] in {"the", "a", "an"}:
            span = " ".join(span.split()[1:])
        if span:
            chunks.append(span)
    return chunks

def soft_entity_alignment(gt_text, pred_text, thr=COSINE_THRESHOLD):
    gt_chunks   = extract_noun_chunks(gt_text)
    pred_chunks = extract_noun_chunks(pred_text)

    if not gt_chunks or not pred_chunks:
        return 0.0, 0.0, 0.0

    # compute embeddings
    gt_emb = embed.encode(gt_chunks, convert_to_tensor=True, show_progress_bar=False)
    pr_emb = embed.encode(pred_chunks, convert_to_tensor=True, show_progress_bar=False)

    # cosine similarity matrix [pred × gt]
    cosmat = util.cos_sim(pr_emb, gt_emb)

    # count matches above threshold
    pred_matches = (cosmat.max(dim=1).values >= thr).sum().item()
    gt_matches   = (cosmat.max(dim=0).values >= thr).sum().item()

    prec = pred_matches / len(pred_chunks)
    rec  = gt_matches   / len(gt_chunks)
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return prec, rec, f1

# ----- Read JSON & Compute Metrics -----
with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

precisions, recalls, f1s = [], [], []

for item in tqdm(data, desc="Computing alignment"):
    gt  = item["ground_truth"]
    pred = item["prediction"]
    p, r, f1 = soft_entity_alignment(gt, pred)
    precisions.append(p)
    recalls.append(r)
    f1s.append(f1)

# ----- Aggregate & Print -----
avg_p = sum(precisions) / len(precisions)
avg_r = sum(recalls)    / len(recalls)
avg_f1= sum(f1s)        / len(f1s)

print(f"\nSoft Entity Alignment @ thr={COSINE_THRESHOLD}")
print(f"  Precision: {avg_p:.3f}")
print(f"  Recall   : {avg_r:.3f}")
print(f"  F1-score : {avg_f1:.3f}")

