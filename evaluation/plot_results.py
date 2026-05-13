import os
import glob

import pandas as pd
import matplotlib.pyplot as plt

def plot_results(dataset):
    benchmark_dirs = [
        './results/v1_linear_ae/benchmarks',
        './results/v2_convolutional_ae/benchmarks',
        './results/v3_vae/benchmarks',
    ]
    dfs = []
    for benchmark_dir in benchmark_dirs:

        csv_files = glob.glob(
            os.path.join(benchmark_dir, '*.csv')
        )

        for file in csv_files:

            df = pd.read_csv(file)
            if dataset in file:
                dfs.append(df)
    jpeg_df = pd.read_csv(
    f'./results/jpeg/benchmarks/jpeg_{dataset}.csv'
    )
    jpeg_df['model'] = 'JPEG'
    dfs.append(jpeg_df)
    combined_df = pd.concat(
        dfs,
        ignore_index=True
    )
    combined_df = combined_df.dropna(subset=['model'])
    os.makedirs(
        './evaluation/plots',
        exist_ok=True
    )
    plt.figure(figsize=(8, 6))
    for model_name in combined_df['model'].unique():

        model_df = combined_df[
            combined_df['model'] == model_name
        ]

        model_df = model_df.sort_values(
            by='avg_size_bytes'
        )

        plt.plot(   
            model_df['avg_size_bytes'],
            model_df['avg_psnr'],
            marker='o',
            label=model_name
        )
        plt.xlabel('Average Compressed Size (bytes)')

    plt.xlabel('Average Compressed Size (bytes)')

    plt.ylabel('Average PSNR')

    plt.title('Compression Size vs PSNR')

    plt.legend()

    plt.grid(True)

    plt.tight_layout()
    plt.savefig(
        f'./evaluation/plots/{dataset}_size_vs_psnr.png'
    )

    plt.close()
    plt.figure(figsize=(8, 6))

    for model_name in combined_df['model'].unique():

        model_df = combined_df[
            combined_df['model'] == model_name
        ]

        model_df = model_df.sort_values(
            by='avg_size_bytes'
        )

        plt.plot(
            model_df['avg_size_bytes'],
            model_df['avg_ssim'],
            marker='o',
            label=model_name
        )
    plt.xlabel('Average Compressed Size (bytes)')

    plt.ylabel('Average SSIM')

    plt.title('Compression Size vs SSIM')

    plt.legend()

    plt.grid(True)

    plt.tight_layout()
    plt.savefig(
        f'./evaluation/plots/{dataset}_size_vs_ssim.png'
    )

    plt.close()
    print("Plots saved to ./evaluation/plots")

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset',
        type=str,
        required=True
    )

    args = parser.parse_args()

    plot_results(args.dataset)