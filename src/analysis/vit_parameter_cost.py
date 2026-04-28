import argparse
import math

import torch

from src.models.model_factory import get_model


def count_params(named_params, keyword_filter=None):
    total = 0
    selected = []

    for name, param in named_params:
        if keyword_filter is None or any(k in name for k in keyword_filter):
            n = param.numel()
            total += n
            selected.append((name, n, tuple(param.shape)))

    return total, selected


def bytes_to_mb(num_bytes):
    return num_bytes / (1024 ** 2)


def swag_memory_mb(num_params, max_rank=20, bytes_per_float=4):
    """
    SWAG stores approximately:
    - mean: d
    - second moment: d
    - deviations: Kd

    Total = (K + 2)d floats
    """
    num_floats = (max_rank + 2) * num_params
    return bytes_to_mb(num_floats * bytes_per_float)


def adamw_memory_mb(num_params, bytes_per_float=4):
    """
    AdamW stores:
    - parameters: d
    - gradients: d
    - first moment: d
    - second moment: d

    Approx = 4d floats during training.
    This ignores activations and CUDA overhead.
    """
    num_floats = 4 * num_params
    return bytes_to_mb(num_floats * bytes_per_float)


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="vit_tiny_patch16_224")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--max-rank", type=int, default=20)
    parser.add_argument("--show-selected", action="store_true")
    args = parser.parse_args()

    model = get_model(
        name=args.model,
        num_classes=args.num_classes,
        pretrained=False,
    )

    named_params = list(model.named_parameters())

    total_params, _ = count_params(named_params)

    head_params, head_selected = count_params(
        named_params,
        keyword_filter=["head"],
    )

    final_block_params, final_block_selected = count_params(
        named_params,
        keyword_filter=["blocks.11"],
    )

    final_block_head_params, final_block_head_selected = count_params(
        named_params,
        keyword_filter=["blocks.11", "head"],
    )

    print_section("ViT-Tiny Parameter Count and SWAG Storage Cost")

    print(f"Model: {args.model}")
    print(f"Number of classes: {args.num_classes}")
    print(f"SWAG max rank K: {args.max_rank}")

    print("\nParameter counts:")
    print(f"Full ViT parameters:              {total_params:,}")
    print(f"Classifier head parameters:       {head_params:,}")
    print(f"Final block parameters:           {final_block_params:,}")
    print(f"Final block + head parameters:    {final_block_head_params:,}")

    print_section("Approximate SWAG Statistics Memory")

    print(
        f"Full ViT SWAG memory:              "
        f"{swag_memory_mb(total_params, args.max_rank):.2f} MB"
    )
    print(
        f"Head-only SWAG memory:             "
        f"{swag_memory_mb(head_params, args.max_rank):.2f} MB"
    )
    print(
        f"Final block + head SWAG memory:    "
        f"{swag_memory_mb(final_block_head_params, args.max_rank):.2f} MB"
    )

    print_section("Approximate AdamW Training State Memory")

    print(
        f"Full ViT AdamW state memory:       "
        f"{adamw_memory_mb(total_params):.2f} MB"
    )
    print(
        f"Final block + head AdamW memory:   "
        f"{adamw_memory_mb(final_block_head_params):.2f} MB"
    )

    print_section("Mathematical Summary")

    print(
        "For d selected parameters and SWAG rank K:\n"
        "  mean vector         = d floats\n"
        "  second moment       = d floats\n"
        "  low-rank deviations = Kd floats\n"
        "  total SWAG storage  = (K + 2)d floats\n"
    )

    print(
        f"With K = {args.max_rank}, SWAG stores approximately "
        f"{args.max_rank + 2}x the selected parameter count."
    )

    print("\nThis motivates subspace SWAG:")
    print("  Full ViT SWAG:              p(theta_all | D)")
    print("  Final-block + head SWAG:    p(theta_lastblock, theta_head | D)")
    print("  LoRA + head SWAG:           p(theta_LoRA, theta_head | D)")

    if args.show_selected:
        print_section("Selected Final Block + Head Parameters")
        for name, n, shape in final_block_head_selected:
            print(f"{name:<60} {n:>12,} {shape}")


if __name__ == "__main__":
    main()