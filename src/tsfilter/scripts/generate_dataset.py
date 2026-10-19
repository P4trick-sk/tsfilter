from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tsfilter.config.param_spaces import load_param_spaces, GENERATOR_REGISTRY, NOISE_REGISTRY, \
    load_generator_selection, load_noise_selection
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
        "--generator-selection", type=Path,
        help="YAML: which generators are allowed, plus optional multi/mixed composite entries",
        default=Path("config/generator_selection.yaml")
    )
    parser.add_argument(
        "--noise-selection", type=Path,
        help="YAML: which noises are allowed, plus optional multi/mixed composite entries",
        default=Path("config/noise_selection.yaml")
    )

    parser.add_argument(
        "--generator-config", type=Path, default=Path("config/generator_param_spaces.yaml"),
    )
    parser.add_argument(
        "--noise-config", type=Path, default=Path("config/noise_param_spaces.yaml"),
    )

    parser.add_argument("--output", type=Path, required=True, help="File path for json, directory path for csv")
    parser.add_argument("--format", choices=["json", "csv"], default="json")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    seed_seq = np.random.SeedSequence(args.seed)
    gen_seed, noise_seed, builder_seed = (int(s.generate_state(1)[0]) for s in seed_seq.spawn(3))

    generator_classes = load_generator_selection(args.generator_selection)
    noise_classes = load_noise_selection(args.noise_selection)

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