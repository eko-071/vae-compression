import os
import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import argparse
import yaml

from training.train import train
from evaluation.benchmark_model import benchmark_model


def sweep_latent_dims(config):

    latent_dims = [4, 8, 16, 32, 64]

    # Optional: add 128 for CelebA
    if config['dataset'].lower() == 'celeba':
        latent_dims.append(128)

    print("\nStarting latent dimension sweep...\n")

    for latent_dim in latent_dims:

        config['latent_dim'] = latent_dim

        print("\n" + "=" * 60)
        print(f"Running latent_dim = {latent_dim}")
        print("=" * 60 + "\n")

        # Train model
        train(config)

        # Benchmark trained model
        benchmark_model(config)

    print("\nSweep complete.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config',
        type=str,
        required=True
    )

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    sweep_latent_dims(config)