import json
from pathlib import Path

# كيقلّب على نتائج التشغيل فالمجلد الجديد أو القديم
paths = [
    Path("results/metrics.json"),
    Path("ml/runs/run1/metrics.json"),
    Path("runs/run1/metrics.json"),
]

file = next((p for p in paths if p.exists()), None)

if file is None:
    raise SystemExit(
        "metrics.json not found. Copy the original file to results/metrics.json."
    )

data = json.loads(file.read_text(encoding="utf-8"))

print("PITCHWAVE — SAVED TRAINING RESULTS")
print("These are saved logs, not a new training run.")
print(f"File: {file}")
print(f"Recorded device: {data.get('device', 'not recorded')}")
print(f"Seed: {data.get('seed', 'not recorded')}")

print("\nTRAINING HISTORY")
for row in data.get("history", []):
    print(
        f"Epoch {row['epoch']:02d} | "
        f"Train loss: {row['train_bce']:.5f} | "
        f"Val loss: {row['val_log_loss']:.5f} | "
        f"Val AP: {row['val_average_precision']:.5f}"
    )

test = data.get("evaluation", {}).get("test", {})

if test:
    print("\nHELD-OUT TEST RESULTS")
    print(
        f"{'Model':<18} {'AP':>9} {'ROC-AUC':>9} "
        f"{'Brier':>9} {'Log loss':>10}"
    )

    for name, scores in test.items():
        auc = scores.get("roc_auc")
        auc_text = f"{auc:.5f}" if auc is not None else "N/A"

        print(
            f"{name:<18} "
            f"{scores['average_precision']:>9.5f} "
            f"{auc_text:>9} "
            f"{scores['brier']:>9.5f} "
            f"{scores['log_loss']:>10.5f}"
        )
else:
    print("\nNo evaluation section in this file.")