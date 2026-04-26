import argparse
import json
import os

import pandas as pd


RUN_PATHS = {
    "debug": {
        "sgd": "outputs/metrics/resnet18_cifar10_sgd_metrics.json",
        "swa": "outputs/metrics/resnet18_cifar10_swa_metrics.json",
        "swag": "outputs/metrics/resnet18_cifar10_swag_metrics.json",
        "save": "outputs/metrics/sgd_swa_swag_comparison_debug.csv",
    },
    "long": {
        "sgd": "outputs/metrics/resnet18_cifar10_sgd_long_metrics.json",
        "swa": "outputs/metrics/resnet18_cifar10_swa_long_metrics.json",
        "swag": "outputs/metrics/resnet18_cifar10_swag_long_metrics.json",
        "save": "outputs/metrics/sgd_swa_swag_comparison_long.csv",
    },
}


def load_json(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing metrics file: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_comparison(run_name: str):
    paths = RUN_PATHS[run_name]

    sgd = load_json(paths["sgd"])
    swa = load_json(paths["swa"])
    swag = load_json(paths["swag"])

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
    df.to_csv(paths["save"], index=False)

    print(f"\nSGD vs SWA vs SWAG comparison ({run_name}):")
    print(df.to_string(index=False))
    print(f"\nSaved comparison table to: {paths['save']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        type=str,
        default="long",
        choices=["debug", "long"],
        help="Which experiment group to compare",
    )
    args = parser.parse_args()

    build_comparison(args.run)


if __name__ == "__main__":
    main()