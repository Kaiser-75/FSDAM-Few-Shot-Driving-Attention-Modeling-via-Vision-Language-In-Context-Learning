import json
import os
from alignscore import AlignScore

import torch
print(torch.cuda.is_available())

torch.device("cpu")
scorer = AlignScore(
    model='roberta-base',
    batch_size=32,
    device='mps',  # changed from 'cuda:0' to 'mps'
    ckpt_path='AlignScore-base.ckpt',
    evaluation_mode='nli'
)

# Get the current directory where the Python file is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Build the full path to the JSON file
json_file_path = os.path.join(current_dir, 'predictions_one_shot.json')

# Read the JSON file
with open(json_file_path, 'r') as f:
    data = json.load(f)

final_alignscore = 0

for i in data:
    context = i["ground_truth"]
    claim = i["prediction"]
    print(context)
    print(claim)
    score = scorer.score(contexts=context, claims=claim)
    print(score)
    final_alignscore += float(score[0])
    print("\n")

print("Final AlignScore for one-shot: ")
print(final_alignscore/len(data))
