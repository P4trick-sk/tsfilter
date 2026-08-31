from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from tsfilter.config.param_spaces import load_param_spaces, GENERATOR_REGISTRY, NOISE_REGISTRY
from tsfilter.generators.factory import RandomGeneratorFactory
from tsfilter.noise.factory import RandomNoiseFactory
from tsfilter.dataset.builder import DatasetBuilder
from tsfilter.dataset import storage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a synthetic noisy time series dataset.")

    parser.add_argument("--length", type=int, required=True, help="Length of each generated series")
    parser.add_argument("--n-samples", type=int, default=100, help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=None, help="Master seed for the whole run")

    parser.add_argument(
        "--generators", nargs="+", default=list(GENERATOR_REGISTRY.keys()),
        help=f"Generator class names to sample from. Available: {list(GENERATOR_REGISTRY.keys())}",
    )
    parser.add_argument(
        "--noise", nargs="+", default=list(NOISE_REGISTRY.keys()),
        help=f"Noise class names to sample from. Available: {list(NOISE_REGISTRY.keys())}",
    )

    parser.add_argument(
        "--generator-config", type=Path, default=Path("src/tsfilter/config/generator_param_spaces.yaml"),
    )
    parser.add_argument(
        "--noise-config", type=Path, default=Path("src/tsfilter/config/noise_param_spaces.yaml"),
    )

    parser.add_argument("--output", type=Path, required=True, help="Output file path")
    parser.add_argument("--format", choices=["json", "csv"], default="json")

    return parser.parse_args()


def resolve_classes(names: list[str], registry: dict, kind: str) -> list[type]:
    unknown = [n for n in names if n not in registry]
    if unknown:
        raise ValueError(f"Unknown {kind} name(s): {unknown}. Available: {list(registry.keys())}")
    return [registry[n] for n in names]


def main() -> None:
    args = parse_args()

    seed_seq = np.random.SeedSequence(args.seed)
    gen_seed, noise_seed, builder_seed = (int(s.generate_state(1)[0]) for s in seed_seq.spawn(3))

    generator_classes = resolve_classes(args.generators, GENERATOR_REGISTRY, "generator")
    noise_classes = resolve_classes(args.noise, NOISE_REGISTRY, "noise")

    generator_spaces = load_param_spaces(args.generator_config, GENERATOR_REGISTRY)
    noise_spaces = load_param_spaces(args.noise_config, NOISE_REGISTRY)

    generator_factory = RandomGeneratorFactory(
        length=args.length, param_spaces=generator_spaces, seed=gen_seed,
    )
    noise_factory = RandomNoiseFactory(param_spaces=noise_spaces, seed=noise_seed)

    builder = DatasetBuilder(
        generator_factory=generator_factory,
        noise_factory=noise_factory,
        generator_classes=generator_classes,
        noise_classes=noise_classes,
        seed=builder_seed,
    )

    print(f"Generating {args.n_samples} samples of length {args.length}...")
    samples = builder.build(args.n_samples)

    if args.format == "json":
        storage.to_json(samples, args.output)
    else:
        storage.to_csv(samples, args.output)

    print(f"Wrote {len(samples)} samples to {args.output}")


if __name__ == "__main__":
    main()