"""Offline, weights-only memory estimates for ModernLLM model presets.

No model is instantiated and no GPU/PyTorch dependency is required.
"""

import argparse
import json

from modern_llm.config import PRESET_CONFIGS

BYTES_PER_PARAMETER = {"fp32": 4, "fp16": 2, "bf16": 2}
GIB = 1024 ** 3


def estimate_preset(preset: str, dtype: str = "bf16") -> dict:
    """Return the estimated parameter footprint (not total training VRAM)."""
    if preset not in PRESET_CONFIGS:
        raise ValueError(f"Unknown preset: {preset}")
    if dtype not in BYTES_PER_PARAMETER:
        raise ValueError(f"Unsupported dtype: {dtype}")

    config = PRESET_CONFIGS[preset]
    parameters = config.estimated_params
    return {
        "preset": preset,
        "model": config.model_name,
        "dtype": dtype,
        "parameters": parameters,
        "weight_bytes": parameters * BYTES_PER_PARAMETER[dtype],
        "weight_gib": round(parameters * BYTES_PER_PARAMETER[dtype] / GIB, 3),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Estimate ModernLLM preset weight memory without loading a model."
    )
    parser.add_argument("--preset", choices=sorted(PRESET_CONFIGS), help="Default: all presets")
    parser.add_argument("--dtype", choices=sorted(BYTES_PER_PARAMETER), default="bf16")
    parser.add_argument("--json", action="store_true", help="Machine-readable output")
    args = parser.parse_args(argv)

    presets = [args.preset] if args.preset else sorted(PRESET_CONFIGS)
    rows = [estimate_preset(name, args.dtype) for name in presets]
    if args.json:
        print(json.dumps({"estimates": rows, "scope": "weights_only"}, ensure_ascii=False, indent=2))
    else:
        for row in rows:
            print(f"{row['preset']:7} {row['parameters']:,} params | {row['weight_gib']:.3f} GiB ({row['dtype']})")
        print("Weights only: excludes gradients, optimizer, activations, KV cache and runtime overhead.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
