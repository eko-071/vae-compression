import os
import glob
import argparse

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_results(dataset):
    csv_files = glob.glob('./results/*/benchmarks/*.csv')
    dfs = []

    for file in csv_files:
        df = pd.read_csv(file)
        if dataset in df['dataset'].values:
            dfs.append(df)

    if not dfs:
        print(f"No benchmark CSVs found for dataset: {dataset}")
        return

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined[combined['dataset'] == dataset]

    os.makedirs('./evaluation/plots', exist_ok=True)

    for metric, ylabel, filename in [
        ('avg_psnr', 'Average PSNR (dB)', f'{dataset}_size_vs_psnr.png'),
        ('avg_ssim', 'Average SSIM',       f'{dataset}_size_vs_ssim.png'),
    ]:
        plt.figure(figsize=(8, 6))

        for model_name, group in combined.groupby('model'):
            group = group.sort_values('avg_size_bytes')
            plt.plot(group['avg_size_bytes'], group[metric],
                     marker='o', label=model_name)

        plt.xlabel('Average Compressed Size (bytes)')
        plt.ylabel(ylabel)
        plt.title(f'{ylabel} vs Compressed Size — {dataset}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'./evaluation/plots/{filename}')
        plt.close()

    print(f"Plots saved to ./evaluation/plots/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()
    plot_results(args.dataset)