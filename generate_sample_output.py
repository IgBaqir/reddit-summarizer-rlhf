"""
Generate a sample output file to demonstrate what goes in the outputs folder
"""

import json
import os

# Create a sample evaluation results file
sample_results = {
    "rouge1": {
        "mean": 0.4523,
        "scores": [0.45, 0.46, 0.44, 0.45, 0.46]
    },
    "rouge2": {
        "mean": 0.3124,
        "scores": [0.31, 0.32, 0.30, 0.31, 0.32]
    },
    "rougeL": {
        "mean": 0.4231,
        "scores": [0.42, 0.43, 0.41, 0.42, 0.43]
    },
    "reward": {
        "mean": 0.0,
        "scores": [0.0, 0.0, 0.0, 0.0, 0.0]
    },
    "num_samples": 5,
    "note": "This is a sample output file. Run evaluation/evaluate.py to generate real results."
}

# Save to outputs folder
os.makedirs("outputs", exist_ok=True)
output_path = "outputs/evaluation_results.json"

with open(output_path, 'w') as f:
    json.dump(sample_results, f, indent=2)

print(f"✓ Created sample output file: {output_path}")
print("\nTo generate real evaluation results, run:")
print("  python3 evaluation/evaluate.py --model <model_path> --test-data ./data/splits/test.json")
print("\nNote: You need trained models first. Run training scripts to create them.")

