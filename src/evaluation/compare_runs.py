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
        "swa_from_swag": False,
    },
    "long": {
        "sgd": "outputs/metrics/resnet18_cifar10_sgd_long_metrics.json",
        "swa": "outputs/metrics/resnet18_cifar10_swag_long_metrics.json",
        "swag": "outputs/metrics/resnet18_cifar10_swag_long_metrics.json",
        "save": "outputs/metrics/sgd_swa_swag_comparison_long.csv",
        "swa_from_swag": True,
    },
}


def load_json(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing metrics file: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_last_metric(metrics: dict, key: str):
    if key not in metrics:
        raise KeyError(f"Missing key '{key}' in metrics file.")

    value = metrics[key]

    if isinstance(value, list):
        if len(value) == 0:
            raise ValueError(f"Metric list '{key}' is empty.")
        return value[-1]

    return value


def get_swa_metrics(swa_metrics: dict):
    required = [
        "final_swa_test_acc",
        "final_swa_test_nll",
        "final_swa_test_ece",
    ]

    missing = [key for key in required if key not in swa_metrics]

    if missing:
        raise KeyError(
            "Missing SWA metrics in file. "
            f"Missing keys: {missing}. "
            "If this is the long run, make sure train_swag.py saves "
            "final_swa_test_acc, final_swa_test_nll, final_swa_test_ece "
            "before save_metrics(...) is called."
        )

    return {
        "accuracy": swa_metrics["final_swa_test_acc"],
        "nll": swa_metrics["final_swa_test_nll"],
        "ece": swa_metrics["final_swa_test_ece"],
    }


def get_swag_metrics(swag_metrics: dict):
    required = [
        "final_swag_test_acc",
        "final_swag_test_nll",
        "final_swag_test_ece",
    ]

    missing = [key for key in required if key not in swag_metrics]

    if missing:
        raise KeyError(
            "Missing SWAG metrics in file. "
            f"Missing keys: {missing}."
        )

    return {
        "accuracy": swag_metrics["final_swag_test_acc"],
        "nll": swag_metrics["final_swag_test_nll"],
        "ece": swag_metrics["final_swag_test_ece"],
    }


def build_comparison(run_name: str):
    paths = RUN_PATHS[run_name]

    sgd_metrics = load_json(paths["sgd"])
    swa_metrics_source = load_json(paths["swa"])
    swag_metrics = load_json(paths["swag"])

    sgd_row = {
        "method": "SGD",
        "accuracy": get_last_metric(sgd_metrics, "test_acc"),
        "nll": get_last_metric(sgd_metrics, "test_nll"),
        "ece": get_last_metric(sgd_metrics, "test_ece"),
    }

    swa_values = get_swa_metrics(swa_metrics_source)
    swa_row = {
        "method": "SWA",
        **swa_values,
    }

    swag_values = get_swag_metrics(swag_metrics)
    swag_row = {
        "method": "SWAG",
        **swag_values,
    }

    rows = [sgd_row, swa_row, swag_row]

    df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(paths["save"]), exist_ok=True)
    df.to_csv(paths["save"], index=False)

    print(f"\nSGD vs SWA vs SWAG comparison ({run_name}):")

    if paths["swa_from_swag"]:
        print("Note: SWA and SWAG are evaluated from the same SWAG trajectory.")

    print(df.to_string(index=False))
    print(f"\nSaved comparison table to: {paths['save']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        type=str,
        default="long",
        choices=["debug", "long"],
        help="Which experiment group to compare.",
    )
    args = parser.parse_args()

    build_comparison(args.run)


if __name__ == "__main__":
    main()