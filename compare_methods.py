#!/usr/bin/env python3
"""
Compare multiple methods' results on OpenMaterial dataset
Usage: python compare_methods.py --methods neus2 2dgs pgsr instant-nsr-pl-wmask
"""

import os
import json
import argparse
import pandas as pd
import glob

bsdf_names = [
    'diffuse',
    'dielectric',
    'roughdielectric',
    'conductor',
    'roughconductor',
    'plastic',
    'roughplastic'
]

def collect_results(method_name, output_dir):
    """Collect PSNR, SSIM, LPIPS, Chamfer Distance for a method"""
    results = {
        'psnr': {name: [] for name in bsdf_names},
        'ssim': {name: [] for name in bsdf_names},
        'lpips': {name: [] for name in bsdf_names},
        'chamfer': {name: [] for name in bsdf_names}
    }

    if not os.path.exists(output_dir):
        print(f"[WARNING] Output directory not found for {method_name}: {output_dir}")
        return None

    # Collect rendering metrics (PSNR, SSIM, LPIPS)
    for obj_dir in glob.glob(os.path.join(output_dir, '*')):
        if not os.path.isdir(obj_dir):
            continue

        txt_file = os.path.join(obj_dir, f'{method_name}.txt')
        if os.path.exists(txt_file):
            with open(txt_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(':')
                    if len(parts) >= 4:
                        material = parts[2]
                        metrics = parts[3]

                        # Extract PSNR and SSIM
                        if '-' in metrics:
                            psnr_ssim = metrics.split('-')
                            if len(psnr_ssim) >= 2:
                                try:
                                    psnr = float(psnr_ssim[0])
                                    ssim = float(psnr_ssim[1])

                                    for name in bsdf_names:
                                        if material.startswith(name):
                                            results['psnr'][name].append(psnr)
                                            results['ssim'][name].append(ssim)
                                except ValueError:
                                    pass

    # Collect mesh evaluation metrics (Chamfer Distance)
    mesh_output_files = glob.glob(os.path.join(output_dir, '*', f'{method_name}-mesh-output.txt'))
    for mesh_file in mesh_output_files:
        with open(mesh_file, 'r') as f:
            for line in f:
                parts = line.strip().split(':')
                if len(parts) >= 4:
                    material = parts[2]
                    chamfer = float(parts[3])

                    for name in bsdf_names:
                        if material.startswith(name):
                            results['chamfer'][name].append(chamfer)

    # Calculate averages
    avg_results = {}
    for metric in ['psnr', 'ssim', 'lpips', 'chamfer']:
        avg_results[metric] = {}
        for name in bsdf_names:
            if len(results[metric][name]) > 0:
                avg_results[metric][name] = sum(results[metric][name]) / len(results[metric][name])
            else:
                avg_results[metric][name] = 0.0

    return avg_results


def main():
    parser = argparse.ArgumentParser(description='Compare methods on OpenMaterial dataset')
    parser.add_argument('--methods', nargs='+',
                        default=['instant-nsr-pl-wmask', 'neus2', '2dgs', 'pgsr'],
                        help='Methods to compare')
    parser.add_argument('--output_dir_pattern', type=str, default='{}_output',
                        help='Output directory pattern (default: {}_output)')

    args = parser.parse_args()

    print("=" * 60)
    print(" OpenMaterial Benchmark - Methods Comparison")
    print("=" * 60)
    print()

    # Collect results for each method
    all_results = {}
    for method in args.methods:
        output_dir = args.output_dir_pattern.format(method)
        print(f"[+] Collecting results for {method} from {output_dir}...")
        results = collect_results(method, output_dir)
        if results:
            all_results[method] = results
            print(f"    ✓ Results collected")
        else:
            print(f"    ✗ No results found")
        print()

    if not all_results:
        print("[ERROR] No results found for any method!")
        return

    # Print comparison tables
    print("=" * 60)
    print(" PSNR (Peak Signal-to-Noise Ratio) ↑")
    print("=" * 60)
    print()

    psnr_data = {}
    for method, results in all_results.items():
        psnr_data[method] = results['psnr']

    psnr_df = pd.DataFrame(psnr_data).T
    psnr_df = psnr_df[bsdf_names]  # Reorder columns
    print(psnr_df.to_string())
    print()
    print(f"{'Mean':20s}", end='')
    for method in psnr_df.index:
        mean_val = psnr_df.loc[method].replace(0, float('nan')).mean()
        print(f"{mean_val:>15.2f}", end='')
    print()
    print()

    print("=" * 60)
    print(" SSIM (Structural Similarity) ↑")
    print("=" * 60)
    print()

    ssim_data = {}
    for method, results in all_results.items():
        ssim_data[method] = results['ssim']

    ssim_df = pd.DataFrame(ssim_data).T
    ssim_df = ssim_df[bsdf_names]
    print(ssim_df.to_string())
    print()
    print(f"{'Mean':20s}", end='')
    for method in ssim_df.index:
        mean_val = ssim_df.loc[method].replace(0, float('nan')).mean()
        print(f"{mean_val:>15.4f}", end='')
    print()
    print()

    print("=" * 60)
    print(" Chamfer Distance (×100) ↓")
    print("=" * 60)
    print()

    cd_data = {}
    for method, results in all_results.items():
        cd_data[method] = results['chamfer']

    cd_df = pd.DataFrame(cd_data).T
    cd_df = cd_df[bsdf_names]
    print(cd_df.to_string())
    print()
    print(f"{'Mean':20s}", end='')
    for method in cd_df.index:
        mean_val = cd_df.loc[method].replace(0, float('nan')).mean()
        print(f"{mean_val:>15.4f}", end='')
    print()
    print()

    # Save comparison to CSV
    output_csv = 'method_comparison.csv'
    print(f"[+] Saving comparison to {output_csv}...")

    with open(output_csv, 'w') as f:
        f.write("# PSNR\n")
        psnr_df.to_csv(f)
        f.write("\n# SSIM\n")
        ssim_df.to_csv(f)
        f.write("\n# Chamfer Distance\n")
        cd_df.to_csv(f)

    print(f"    ✓ Saved to {output_csv}")
    print()

    # Print best methods for each metric
    print("=" * 60)
    print(" Best Methods per Metric")
    print("=" * 60)
    print()

    for material in bsdf_names:
        print(f"{material:20s}:")

        # Best PSNR
        psnr_col = psnr_df[material]
        psnr_col = psnr_col[psnr_col > 0]  # Filter out zeros
        if len(psnr_col) > 0:
            best_psnr = psnr_col.idxmax()
            print(f"  {'PSNR:':15s} {best_psnr:15s} ({psnr_col[best_psnr]:.2f})")

        # Best SSIM
        ssim_col = ssim_df[material]
        ssim_col = ssim_col[ssim_col > 0]
        if len(ssim_col) > 0:
            best_ssim = ssim_col.idxmax()
            print(f"  {'SSIM:':15s} {best_ssim:15s} ({ssim_col[best_ssim]:.4f})")

        # Best Chamfer
        cd_col = cd_df[material]
        cd_col = cd_col[cd_col > 0]
        if len(cd_col) > 0:
            best_cd = cd_col.idxmin()
            print(f"  {'Chamfer:':15s} {best_cd:15s} ({cd_col[best_cd]:.4f})")

        print()


if __name__ == '__main__':
    main()
