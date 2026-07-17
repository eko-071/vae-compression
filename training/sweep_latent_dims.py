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

    latent_dims = [2, 4, 8, 16, 32, 64, 128, 256, 512]

    print("\nStarting latent dimension sweep...\n")

    for latent_dim in latent_dims:
        config['latent_dim'] = latent_dim

        checkpoint_path = os.path.join(
            config['save_dir'],
            f"{config['dataset']}_ld{latent_dim}.pt",
        )
        if os.path.exists(checkpoint_path):
            print(f"  Skipping ld={latent_dim} — checkpoint exists")
            continue

        print("\n" + "=" * 60)
        print(f"Training latent_dim = {latent_dim}")
        print("=" * 60 + "\n")

        train(config)

    print("\n--- Benchmarking all latent dims ---\n")
    config['latent_dims'] = latent_dims
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