import json
import os
import pandas as pd


BASELINE_METRICS = "outputs/metrics/resnet18_cifar10_sgd_metrics.json"
SWA_METRICS = "outputs/metrics/resnet18_cifar10_swa_metrics.json"
SAVE_PATH = "outputs/metrics/baseline_vs_swa.csv"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    baseline = load_json(BASELINE_METRICS)
    swa = load_json(SWA_METRICS)

    rows = [
        {
            "method": "SGD",
            "accuracy": baseline["test_acc"][-1],
            "nll": baseline["test_nll"][-1],
            "ece": baseline["test_ece"][-1],
        },
        {
            "method": "SWA",
            "accuracy": swa["final_swa_test_acc"],
            "nll": swa["final_swa_test_nll"],
            "ece": swa["final_swa_test_ece"],
        },
    ]

    df = pd.DataFrame(rows)
    os.makedirs("outputs/metrics", exist_ok=True)
    df.to_csv(SAVE_PATH, index=False)

    print(df)
    print(f"\nSaved comparison table to: {SAVE_PATH}")


if __name__ == "__main__":
    main()