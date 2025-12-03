"""
Benchmark Script for Manga Inpainting Methods

Compares three methods:
1. Telea (OpenCV baseline)
2. Criminisi Standard (full resolution, no pyramid)
3. Adaptive Hybrid Pyramid (proposed method)

Metrics: PSNR, SSIM, Processing Time

Usage:
    python benchmark.py --clean <clean_image> --mask <mask_image> [--output <output_dir>]
    
Example:
    python benchmark.py --clean data/clean.png --mask data/mask.png --output results/
"""

import cv2
import numpy as np
import time
import argparse
import os
from pathlib import Path
from typing import Tuple, Dict

# Metrics
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Import inpainter
from inpainter import HybridInpainter


def load_images(clean_path: str, mask_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load clean image, mask, and create masked input.
    
    Returns:
        clean_image: Ground truth (BGR)
        mask: Binary mask (255=inpaint region)
        masked_image: Image with mask applied (for visualization)
    """
    print(f"[BENCHMARK] Loading clean image: {clean_path}")
    clean_image = cv2.imread(clean_path)
    if clean_image is None:
        raise FileNotFoundError(f"Could not load clean image: {clean_path}")
    
    print(f"[BENCHMARK] Loading mask: {mask_path}")
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not load mask: {mask_path}")
    
    # Ensure mask is binary
    mask = (mask > 127).astype(np.uint8) * 255
    
    # Resize mask if needed
    if mask.shape[:2] != clean_image.shape[:2]:
        print(f"[BENCHMARK] Resizing mask from {mask.shape} to {clean_image.shape[:2]}")
        mask = cv2.resize(mask, (clean_image.shape[1], clean_image.shape[0]), 
                         interpolation=cv2.INTER_NEAREST)
    
    # Create masked visualization (white fill for masked regions)
    masked_image = clean_image.copy()
    masked_image[mask > 0] = [255, 255, 255]  # White for masked regions
    
    print(f"[BENCHMARK] Image size: {clean_image.shape[1]}×{clean_image.shape[0]}")
    print(f"[BENCHMARK] Mask pixels: {np.count_nonzero(mask)}")
    
    return clean_image, mask, masked_image


def calculate_metrics(ground_truth: np.ndarray, result: np.ndarray, 
                     mask: np.ndarray) -> Dict[str, float]:
    """
    Calculate PSNR and SSIM between ground truth and result.
    
    Metrics are calculated on the MASKED REGION ONLY for fair comparison.
    """
    # Convert to grayscale for SSIM calculation
    gt_gray = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2GRAY)
    result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    
    # Calculate PSNR (full image)
    psnr_value = psnr(ground_truth, result)
    
    # Calculate SSIM (full image, with multichannel)
    ssim_value = ssim(ground_truth, result, channel_axis=2, data_range=255)
    
    # Also calculate metrics on masked region only
    mask_bool = mask > 0
    
    # Extract masked regions
    gt_masked = ground_truth[mask_bool]
    result_masked = result[mask_bool]
    
    # PSNR on masked region
    mse_masked = np.mean((gt_masked.astype(float) - result_masked.astype(float)) ** 2)
    if mse_masked > 0:
        psnr_masked = 10 * np.log10(255**2 / mse_masked)
    else:
        psnr_masked = float('inf')
    
    return {
        'psnr_full': psnr_value,
        'ssim_full': ssim_value,
        'psnr_masked': psnr_masked,
    }


def run_benchmark(clean_image: np.ndarray, mask: np.ndarray, 
                  masked_image: np.ndarray) -> Dict[str, Dict]:
    """
    Run all three methods and collect results.
    """
    inpainter = HybridInpainter(target_proc_size=450, patch_size=9)
    
    results = {}
    
    methods = [
        ("Telea", "telea"),
        ("Criminisi (Standard)", "criminisi_standard"),
        ("Proposed (Hybrid)", "adaptive"),
    ]
    
    for name, method_key in methods:
        print(f"\n{'='*60}")
        print(f"[BENCHMARK] Running: {name}")
        print(f"{'='*60}")
        
        # Time the inpainting
        start_time = time.time()
        result = inpainter.inpaint(masked_image.copy(), mask.copy(), method=method_key)
        elapsed_time = time.time() - start_time
        
        # Calculate metrics
        metrics = calculate_metrics(clean_image, result, mask)
        
        results[name] = {
            'result': result,
            'time': elapsed_time,
            'psnr': metrics['psnr_full'],
            'ssim': metrics['ssim_full'],
            'psnr_masked': metrics['psnr_masked'],
        }
        
        print(f"[BENCHMARK] {name} completed in {elapsed_time:.2f}s")
        print(f"[BENCHMARK] PSNR: {metrics['psnr_full']:.2f} dB | SSIM: {metrics['ssim_full']:.4f}")
    
    return results


def create_comparison_grid(clean_image: np.ndarray, masked_image: np.ndarray,
                          results: Dict[str, Dict], mask: np.ndarray) -> np.ndarray:
    """
    Create a visual comparison grid.
    
    Layout: [Original] [Masked] [Telea] [Criminisi] [Proposed]
    """
    h, w = clean_image.shape[:2]
    
    # Resize images if too large for display
    max_width = 300
    if w > max_width:
        scale = max_width / w
        new_w = max_width
        new_h = int(h * scale)
    else:
        new_w, new_h = w, h
        scale = 1.0
    
    def resize_img(img):
        if scale < 1.0:
            return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return img
    
    # Prepare images
    images = [
        ("Ground Truth", resize_img(clean_image)),
        ("Masked Input", resize_img(masked_image)),
    ]
    
    for name in ["Telea", "Criminisi (Standard)", "Proposed (Hybrid)"]:
        if name in results:
            images.append((name, resize_img(results[name]['result'])))
    
    # Create grid
    num_images = len(images)
    label_height = 40
    total_height = new_h + label_height
    total_width = new_w * num_images + 10 * (num_images - 1)  # 10px gap
    
    grid = np.ones((total_height, total_width, 3), dtype=np.uint8) * 40  # Dark gray background
    
    x_offset = 0
    for label, img in images:
        # Place image
        grid[0:new_h, x_offset:x_offset+new_w] = img
        
        # Add label background
        cv2.rectangle(grid, (x_offset, new_h), (x_offset+new_w, total_height), (60, 60, 60), -1)
        
        # Add label text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        # Calculate text position (centered)
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        text_x = x_offset + (new_w - text_size[0]) // 2
        text_y = new_h + (label_height + text_size[1]) // 2
        
        cv2.putText(grid, label, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
        
        x_offset += new_w + 10
    
    return grid


def create_detailed_grid(clean_image: np.ndarray, masked_image: np.ndarray,
                        results: Dict[str, Dict], mask: np.ndarray) -> np.ndarray:
    """
    Create a more detailed comparison grid with metrics overlay.
    
    Layout (2 rows):
    Row 1: [Original] [Masked] [Telea Result]
    Row 2: [Criminisi Result] [Proposed Result] [Difference Map]
    """
    h, w = clean_image.shape[:2]
    
    # Target cell size
    cell_w = min(400, w)
    scale = cell_w / w
    cell_h = int(h * scale)
    
    def resize_img(img):
        return cv2.resize(img, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
    
    def add_text_overlay(img, lines):
        """Add text overlay to image"""
        overlay = img.copy()
        y_pos = 25
        for line in lines:
            cv2.putText(overlay, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 0, 0), 3)  # Black outline
            cv2.putText(overlay, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 255, 255), 1)  # White text
            y_pos += 20
        return overlay
    
    # Prepare cells
    cells = []
    
    # Row 1
    cells.append(add_text_overlay(resize_img(clean_image), ["Ground Truth"]))
    cells.append(add_text_overlay(resize_img(masked_image), ["Masked Input"]))
    
    if "Telea" in results:
        r = results["Telea"]
        cells.append(add_text_overlay(resize_img(r['result']), [
            "Telea (OpenCV)",
            f"Time: {r['time']:.2f}s",
            f"PSNR: {r['psnr']:.2f} dB",
            f"SSIM: {r['ssim']:.4f}"
        ]))
    
    # Row 2
    if "Criminisi (Standard)" in results:
        r = results["Criminisi (Standard)"]
        cells.append(add_text_overlay(resize_img(r['result']), [
            "Criminisi (Standard)",
            f"Time: {r['time']:.2f}s",
            f"PSNR: {r['psnr']:.2f} dB",
            f"SSIM: {r['ssim']:.4f}"
        ]))
    
    if "Proposed (Hybrid)" in results:
        r = results["Proposed (Hybrid)"]
        cells.append(add_text_overlay(resize_img(r['result']), [
            "Proposed (Hybrid)",
            f"Time: {r['time']:.2f}s",
            f"PSNR: {r['psnr']:.2f} dB",
            f"SSIM: {r['ssim']:.4f}"
        ]))
    
    # Create difference map (proposed vs ground truth)
    if "Proposed (Hybrid)" in results:
        diff = cv2.absdiff(clean_image, results["Proposed (Hybrid)"]['result'])
        diff_enhanced = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        diff_color = cv2.applyColorMap(cv2.cvtColor(diff_enhanced, cv2.COLOR_BGR2GRAY), 
                                       cv2.COLORMAP_JET)
        cells.append(add_text_overlay(resize_img(diff_color), ["Difference Map", "(Proposed vs GT)"]))
    
    # Build grid (2 rows x 3 cols)
    gap = 5
    rows = 2
    cols = 3
    
    grid_h = rows * cell_h + (rows - 1) * gap
    grid_w = cols * cell_w + (cols - 1) * gap
    
    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 30
    
    for i, cell in enumerate(cells):
        row = i // cols
        col = i % cols
        y = row * (cell_h + gap)
        x = col * (cell_w + gap)
        grid[y:y+cell_h, x:x+cell_w] = cell
    
    return grid


def print_markdown_table(results: Dict[str, Dict]):
    """Print results as a Markdown table for thesis."""
    print("\n")
    print("=" * 70)
    print("BENCHMARK RESULTS (Markdown Table - Copy for Thesis)")
    print("=" * 70)
    print()
    print("| Method | Time (s) | PSNR (dB) | SSIM |")
    print("|--------|----------|-----------|------|")
    
    for name in ["Telea", "Criminisi (Standard)", "Proposed (Hybrid)"]:
        if name in results:
            r = results[name]
            print(f"| {name} | {r['time']:.2f} | {r['psnr']:.2f} | {r['ssim']:.4f} |")
    
    print()
    
    # Also print LaTeX table
    print("LaTeX Table:")
    print("\\begin{tabular}{|l|c|c|c|}")
    print("\\hline")
    print("\\textbf{Method} & \\textbf{Time (s)} & \\textbf{PSNR (dB)} & \\textbf{SSIM} \\\\")
    print("\\hline")
    
    for name in ["Telea", "Criminisi (Standard)", "Proposed (Hybrid)"]:
        if name in results:
            r = results[name]
            latex_name = name.replace("(", "\\textnormal{(").replace(")", ")}")
            print(f"{latex_name} & {r['time']:.2f} & {r['psnr']:.2f} & {r['ssim']:.4f} \\\\")
    
    print("\\hline")
    print("\\end{tabular}")
    print()
    
    # Print analysis
    print("-" * 70)
    print("ANALYSIS:")
    print("-" * 70)
    
    if all(name in results for name in ["Telea", "Criminisi (Standard)", "Proposed (Hybrid)"]):
        telea = results["Telea"]
        criminisi = results["Criminisi (Standard)"]
        proposed = results["Proposed (Hybrid)"]
        
        # Speed comparison
        speedup_vs_criminisi = criminisi['time'] / proposed['time'] if proposed['time'] > 0 else float('inf')
        print(f"• Proposed is {speedup_vs_criminisi:.1f}x faster than Standard Criminisi")
        print(f"• Proposed is {proposed['time']/telea['time']:.1f}x slower than Telea (but much better quality)")
        
        # Quality comparison
        psnr_improvement = proposed['psnr'] - telea['psnr']
        print(f"• PSNR improvement over Telea: {psnr_improvement:+.2f} dB")
        
        ssim_improvement = (proposed['ssim'] - telea['ssim']) * 100
        print(f"• SSIM improvement over Telea: {ssim_improvement:+.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Manga Inpainting Methods")
    parser.add_argument("--clean", "-c", required=True, help="Path to clean (ground truth) image")
    parser.add_argument("--mask", "-m", required=True, help="Path to mask image")
    parser.add_argument("--output", "-o", default="benchmark_results", help="Output directory")
    parser.add_argument("--skip-criminisi", action="store_true", 
                       help="Skip Standard Criminisi (useful for quick tests)")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("MANGA INPAINTING BENCHMARK")
    print("=" * 70)
    
    # Load images
    clean_image, mask, masked_image = load_images(args.clean, args.mask)
    
    # Save masked input for reference
    cv2.imwrite(str(output_dir / "input_masked.png"), masked_image)
    cv2.imwrite(str(output_dir / "input_mask.png"), mask)
    
    # Run benchmark
    inpainter = HybridInpainter(target_proc_size=450, patch_size=9)
    results = {}
    
    methods = [
        ("Telea", "telea"),
        ("Criminisi (Standard)", "criminisi_standard"),
        ("Proposed (Hybrid)", "adaptive"),
    ]
    
    if args.skip_criminisi:
        methods = [m for m in methods if m[1] != "criminisi_standard"]
        print("[BENCHMARK] Skipping Standard Criminisi (--skip-criminisi flag)")
    
    for name, method_key in methods:
        print(f"\n{'='*60}")
        print(f"[BENCHMARK] Running: {name}")
        print(f"{'='*60}")
        
        # Time the inpainting
        start_time = time.time()
        result = inpainter.inpaint(masked_image.copy(), mask.copy(), method=method_key)
        elapsed_time = time.time() - start_time
        
        # Calculate metrics
        metrics = calculate_metrics(clean_image, result, mask)
        
        results[name] = {
            'result': result,
            'time': elapsed_time,
            'psnr': metrics['psnr_full'],
            'ssim': metrics['ssim_full'],
            'psnr_masked': metrics['psnr_masked'],
        }
        
        # Save individual result
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
        cv2.imwrite(str(output_dir / f"result_{safe_name}.png"), result)
        
        print(f"[BENCHMARK] {name} completed in {elapsed_time:.2f}s")
        print(f"[BENCHMARK] PSNR: {metrics['psnr_full']:.2f} dB | SSIM: {metrics['ssim_full']:.4f}")
    
    # Create and save comparison grids
    print("\n[BENCHMARK] Creating comparison grids...")
    
    simple_grid = create_comparison_grid(clean_image, masked_image, results, mask)
    cv2.imwrite(str(output_dir / "comparison_simple.png"), simple_grid)
    
    detailed_grid = create_detailed_grid(clean_image, masked_image, results, mask)
    cv2.imwrite(str(output_dir / "comparison_detailed.png"), detailed_grid)
    
    # Print results table
    print_markdown_table(results)
    
    print(f"\n[BENCHMARK] Results saved to: {output_dir.absolute()}")
    print("[BENCHMARK] Files created:")
    print(f"  • input_masked.png")
    print(f"  • input_mask.png")
    for name in results.keys():
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
        print(f"  • result_{safe_name}.png")
    print(f"  • comparison_simple.png")
    print(f"  • comparison_detailed.png")
    
    return results


if __name__ == "__main__":
    main()
