import json
import os

import pandas as pd


SGD_METRICS = "outputs/metrics/resnet18_cifar10_sgd_metrics.json"
SWA_METRICS = "outputs/metrics/resnet18_cifar10_swa_metrics.json"
SWAG_METRICS = "outputs/metrics/resnet18_cifar10_swag_metrics.json"

SAVE_PATH = "outputs/metrics/sgd_swa_swag_comparison.csv"


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    sgd = load_json(SGD_METRICS)
    swa = load_json(SWA_METRICS)
    swag = load_json(SWAG_METRICS)

    rows = [
        {
            "method": "SGD",
            "accuracy": sgd["test_acc"][-1],
            "nll": sgd["test_nll"][-1],
            "ece": sgd["test_ece"][-1],
        },
        {
            "method": "SWA",
            "accuracy": swa["final_swa_test_acc"],
            "nll": swa["final_swa_test_nll"],
            "ece": swa["final_swa_test_ece"],
        },
        {
            "method": "SWAG",
            "accuracy": swag["final_swag_test_acc"],
            "nll": swag["final_swag_test_nll"],
            "ece": swag["final_swag_test_ece"],
        },
    ]

    df = pd.DataFrame(rows)

    os.makedirs("outputs/metrics", exist_ok=True)
    df.to_csv(SAVE_PATH, index=False)

    print("\nSGD vs SWA vs SWAG comparison:")
    print(df.to_string(index=False))
    print(f"\nSaved comparison table to: {SAVE_PATH}")


if __name__ == "__main__":
    main()