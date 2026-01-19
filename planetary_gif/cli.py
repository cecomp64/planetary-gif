"""
Command-line interface for planetary-gif.

Provides commands for:
- process-single: Process a single image (for parameter tuning)
- process-ser: Process a single SER file (stack + sharpen)
- batch: Full pipeline from SER files to GIF
- align: Align existing images and create GIF
"""

import argparse
import glob
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from . import __version__
from .config import Config, load_config, find_config_file
from .ser_reader import SERReader, get_ser_info
from .frame_quality import rank_frames_from_ser, QUALITY_METRICS
from .stacker import stack_from_ser
from .sharpening import sharpen_image
from .alignment import align_image_files
from .gif import create_gif


def process_single(args):
    """Process a single image with sharpening (for parameter tuning)."""
    config = load_config(args.config)

    # Apply CLI overrides
    if args.wavelet_coeffs:
        config.wavelet.coefficients = args.wavelet_coeffs
    if args.no_wavelet:
        config.wavelet.enabled = False
    if args.deconv_iterations:
        config.deconvolution.iterations = args.deconv_iterations
    if args.psf_sigma:
        config.deconvolution.psf_sigma = args.psf_sigma
    if args.no_deconvolve:
        config.deconvolution.enabled = False

    # Load image
    print(f"Loading: {args.input}")
    image = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"Error: Could not load image: {args.input}")
        sys.exit(1)

    print(f"  Size: {image.shape[1]}x{image.shape[0]}, Depth: {image.dtype}")

    # Apply sharpening
    print("Processing...")
    if config.wavelet.enabled:
        print(f"  Wavelet: coefficients={config.wavelet.coefficients}")
    if config.deconvolution.enabled:
        print(f"  Deconvolution: {config.deconvolution.method}, "
              f"iterations={config.deconvolution.iterations}, "
              f"psf_sigma={config.deconvolution.psf_sigma}")

    result = sharpen_image(
        image,
        wavelet_enabled=config.wavelet.enabled,
        wavelet_coefficients=config.wavelet.coefficients,
        wavelet_type=config.wavelet.wavelet_type,
        deconv_enabled=config.deconvolution.enabled,
        deconv_method=config.deconvolution.method,
        deconv_iterations=config.deconvolution.iterations,
        psf_sigma=config.deconvolution.psf_sigma,
    )

    # Save output
    cv2.imwrite(args.output, result)
    print(f"Saved: {args.output}")


def process_ser(args):
    """Process a single SER file (stack + sharpen)."""
    config = load_config(args.config)

    # Apply CLI overrides
    if args.top_percent:
        config.frame_selection.top_percent = args.top_percent
    if args.wavelet_coeffs:
        config.wavelet.coefficients = args.wavelet_coeffs
    if args.no_wavelet:
        config.wavelet.enabled = False
    if args.deconv_iterations:
        config.deconvolution.iterations = args.deconv_iterations
    if args.psf_sigma:
        config.deconvolution.psf_sigma = args.psf_sigma
    if args.no_deconvolve:
        config.deconvolution.enabled = False

    # Get SER info
    print(f"Processing: {args.input}")
    info = get_ser_info(args.input)
    print(f"  Frames: {info['frame_count']}, Size: {info['width']}x{info['height']}, "
          f"Depth: {info['bit_depth']}-bit, Color: {info['color_mode']}")

    # Rank frames
    print(f"Ranking frames by {config.frame_selection.quality_metric}...")
    best_indices = rank_frames_from_ser(
        args.input,
        metric=config.frame_selection.quality_metric,
        top_percent=config.frame_selection.top_percent,
        progress_callback=lambda cur, total: print(f"\r  {cur}/{total}", end='', flush=True)
    )
    print(f"\n  Selected {len(best_indices)}/{info['frame_count']} frames "
          f"(top {config.frame_selection.top_percent*100:.0f}%)")

    # Stack frames
    print(f"Stacking with {config.stacking.method}...")
    stacked = stack_from_ser(
        args.input,
        best_indices,
        method=config.stacking.method,
        sigma_low=config.stacking.sigma_low,
        sigma_high=config.stacking.sigma_high,
    )

    # Apply sharpening
    print("Sharpening...")
    if config.wavelet.enabled:
        print(f"  Wavelet: coefficients={config.wavelet.coefficients}")
    if config.deconvolution.enabled:
        print(f"  Deconvolution: {config.deconvolution.method}, "
              f"iterations={config.deconvolution.iterations}")

    result = sharpen_image(
        stacked,
        wavelet_enabled=config.wavelet.enabled,
        wavelet_coefficients=config.wavelet.coefficients,
        wavelet_type=config.wavelet.wavelet_type,
        deconv_enabled=config.deconvolution.enabled,
        deconv_method=config.deconvolution.method,
        deconv_iterations=config.deconvolution.iterations,
        psf_sigma=config.deconvolution.psf_sigma,
    )

    # Save output
    cv2.imwrite(args.output, result)
    print(f"Saved: {args.output}")


def batch(args):
    """Process multiple SER files and create GIF."""
    config = load_config(args.config)

    # Apply CLI overrides
    if args.top_percent:
        config.frame_selection.top_percent = args.top_percent
    if args.wavelet_coeffs:
        config.wavelet.coefficients = args.wavelet_coeffs
    if args.no_wavelet:
        config.wavelet.enabled = False
    if args.deconv_iterations:
        config.deconvolution.iterations = args.deconv_iterations
    if args.psf_sigma:
        config.deconvolution.psf_sigma = args.psf_sigma
    if args.no_deconvolve:
        config.deconvolution.enabled = False

    # Find SER files
    ser_files = sorted(glob.glob(args.ser_pattern))
    if not ser_files:
        print(f"No SER files found matching: {args.ser_pattern}")
        sys.exit(1)

    print(f"Found {len(ser_files)} SER files")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process each SER file
    processed_images = []
    for i, ser_path in enumerate(ser_files, start=1):
        print(f"\n[{i}/{len(ser_files)}] {os.path.basename(ser_path)}")

        info = get_ser_info(ser_path)
        print(f"  Frames: {info['frame_count']}, Size: {info['width']}x{info['height']}")

        # Rank frames
        print(f"  Ranking frames...", end='', flush=True)
        best_indices = rank_frames_from_ser(
            ser_path,
            metric=config.frame_selection.quality_metric,
            top_percent=config.frame_selection.top_percent,
        )
        print(f" selected {len(best_indices)}/{info['frame_count']}")

        # Stack
        print(f"  Stacking...", end='', flush=True)
        stacked = stack_from_ser(
            ser_path,
            best_indices,
            method=config.stacking.method,
            sigma_low=config.stacking.sigma_low,
            sigma_high=config.stacking.sigma_high,
        )
        print(" done")

        # Sharpen
        print(f"  Sharpening...", end='', flush=True)
        result = sharpen_image(
            stacked,
            wavelet_enabled=config.wavelet.enabled,
            wavelet_coefficients=config.wavelet.coefficients,
            wavelet_type=config.wavelet.wavelet_type,
            deconv_enabled=config.deconvolution.enabled,
            deconv_method=config.deconvolution.method,
            deconv_iterations=config.deconvolution.iterations,
            psf_sigma=config.deconvolution.psf_sigma,
        )
        print(" done")

        # Save processed image
        basename = Path(ser_path).stem
        output_path = os.path.join(args.output_dir, f"{basename}_processed.png")
        cv2.imwrite(output_path, result)
        processed_images.append(output_path)
        print(f"  Saved: {output_path}")

    print(f"\nProcessed {len(processed_images)} SER files")

    # Create GIF if requested
    if args.gif:
        print(f"\nCreating aligned GIF: {args.gif}")

        # Create temp directory for aligned images
        aligned_dir = os.path.join(args.output_dir, "aligned_tmp")
        if os.path.exists(aligned_dir):
            shutil.rmtree(aligned_dir)

        # Align images
        w, h, max_x, max_y = align_image_files(processed_images, aligned_dir)

        # Create GIF
        aligned_pattern = os.path.join(aligned_dir, "aligned_*.png")
        create_gif(
            aligned_pattern,
            args.gif,
            w, h,
            max_x, max_y,
            fps=config.gif.fps,
            crop_padding=config.gif.crop_padding,
            ping_pong=config.gif.ping_pong,
        )

        # Cleanup
        shutil.rmtree(aligned_dir)


def align(args):
    """Align existing images and create GIF."""
    config = load_config(args.config)

    # Find images
    image_paths = sorted(glob.glob(args.image_pattern))
    if not image_paths:
        print(f"No images found matching: {args.image_pattern}")
        sys.exit(1)

    print(f"Found {len(image_paths)} images")

    # Create temp directory for aligned images
    output_dir = os.path.dirname(args.output) or "."
    aligned_dir = os.path.join(output_dir, "aligned_tmp")
    if os.path.exists(aligned_dir):
        shutil.rmtree(aligned_dir)

    # Align images
    w, h, max_x, max_y = align_image_files(image_paths, aligned_dir)

    # Create GIF
    aligned_pattern = os.path.join(aligned_dir, "aligned_*.png")
    create_gif(
        aligned_pattern,
        args.output,
        w, h,
        max_x, max_y,
        fps=config.gif.fps,
        crop_padding=config.gif.crop_padding,
        ping_pong=config.gif.ping_pong,
    )

    # Cleanup
    shutil.rmtree(aligned_dir)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Process planetary astrophotography SER files into rotation GIFs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single image (for tuning parameters)
  planetary-gif process-single input.png -o output.png --config config.yaml

  # Process single SER file
  planetary-gif process-ser capture.ser -o stacked.png

  # Full pipeline: SER files to GIF
  planetary-gif batch 'captures/*.ser' -o processed/ --gif jupiter.gif

  # Align existing images and create GIF
  planetary-gif align 'images/*.png' -o output.gif
""")
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')

    subparsers = parser.add_subparsers(dest='command', help='Command to run',
                                        parser_class=argparse.ArgumentParser)

    # process-single command
    p_single = subparsers.add_parser(
        'process-single',
        help='Process a single image (for parameter tuning)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Apply wavelet sharpening and deconvolution to a single image.',
        epilog="""
Examples:
  planetary-gif process-single input.png -o output.png
  planetary-gif process-single input.png -o output.png --config config.yaml
  planetary-gif process-single input.png -o output.png --wavelet-coeffs 2.0 1.5 1.0
  planetary-gif process-single input.png -o output.png --no-deconvolve
""")
    p_single.add_argument('input', help='Input image path (PNG, TIFF, etc.)')
    p_single.add_argument('-o', '--output', required=True,
                          help='Output image path (required)')
    p_single.add_argument('--config', metavar='FILE',
                          help='Config YAML file (default: auto-detect config.yaml)')
    p_single.add_argument('--wavelet-coeffs', type=float, nargs='+', metavar='N',
                          help='Wavelet layer coefficients, fine to coarse (default: 1.5 1.2 1.0 1.0)')
    p_single.add_argument('--no-wavelet', action='store_true',
                          help='Disable wavelet sharpening')
    p_single.add_argument('--deconv-iterations', type=int, metavar='N',
                          help='Richardson-Lucy iterations (default: 10)')
    p_single.add_argument('--psf-sigma', type=float, metavar='N',
                          help='PSF sigma for deconvolution (default: 1.5)')
    p_single.add_argument('--no-deconvolve', action='store_true',
                          help='Disable deconvolution')
    p_single.set_defaults(func=process_single)

    # process-ser command
    p_ser = subparsers.add_parser(
        'process-ser',
        help='Process a single SER file (stack + sharpen)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Read a SER video file, select best frames, stack, and sharpen.',
        epilog="""
Examples:
  planetary-gif process-ser capture.ser -o stacked.png
  planetary-gif process-ser capture.ser -o stacked.png --top-percent 0.15
  planetary-gif process-ser capture.ser -o stacked.png --config config.yaml
""")
    p_ser.add_argument('input', help='Input SER video file path')
    p_ser.add_argument('-o', '--output', required=True,
                       help='Output image path (required)')
    p_ser.add_argument('--config', metavar='FILE',
                       help='Config YAML file (default: auto-detect config.yaml)')
    p_ser.add_argument('--top-percent', type=float, metavar='N',
                       help='Fraction of best frames to use, 0.0-1.0 (default: 0.10)')
    p_ser.add_argument('--wavelet-coeffs', type=float, nargs='+', metavar='N',
                       help='Wavelet layer coefficients, fine to coarse (default: 1.5 1.2 1.0 1.0)')
    p_ser.add_argument('--no-wavelet', action='store_true',
                       help='Disable wavelet sharpening')
    p_ser.add_argument('--deconv-iterations', type=int, metavar='N',
                       help='Richardson-Lucy iterations (default: 10)')
    p_ser.add_argument('--psf-sigma', type=float, metavar='N',
                       help='PSF sigma for deconvolution (default: 1.5)')
    p_ser.add_argument('--no-deconvolve', action='store_true',
                       help='Disable deconvolution')
    p_ser.set_defaults(func=process_ser)

    # batch command
    p_batch = subparsers.add_parser(
        'batch',
        help='Process multiple SER files and create GIF',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Process multiple SER files: stack, sharpen, align, and create GIF.',
        epilog="""
Examples:
  planetary-gif batch 'videos/*.ser' -o processed/
  planetary-gif batch 'videos/*.ser' -o processed/ --gif jupiter.gif
  planetary-gif batch 'videos/*.ser' -o processed/ --gif jupiter.gif --config config.yaml
  planetary-gif batch 'videos/*.ser' -o processed/ --top-percent 0.15 --deconv-iterations 15
""")
    p_batch.add_argument('ser_pattern',
                         help="Glob pattern for SER files (use quotes: 'videos/*.ser')")
    p_batch.add_argument('-o', '--output-dir', default='processed', metavar='DIR',
                         help='Output directory for processed images (default: processed)')
    p_batch.add_argument('--gif', metavar='FILE',
                         help='Create aligned rotation GIF at this path')
    p_batch.add_argument('--config', metavar='FILE',
                         help='Config YAML file (default: auto-detect config.yaml)')
    p_batch.add_argument('--top-percent', type=float, metavar='N',
                         help='Fraction of best frames to use, 0.0-1.0 (default: 0.10)')
    p_batch.add_argument('--wavelet-coeffs', type=float, nargs='+', metavar='N',
                         help='Wavelet layer coefficients, fine to coarse (default: 1.5 1.2 1.0 1.0)')
    p_batch.add_argument('--no-wavelet', action='store_true',
                         help='Disable wavelet sharpening')
    p_batch.add_argument('--deconv-iterations', type=int, metavar='N',
                         help='Richardson-Lucy iterations (default: 10)')
    p_batch.add_argument('--psf-sigma', type=float, metavar='N',
                         help='PSF sigma for deconvolution (default: 1.5)')
    p_batch.add_argument('--no-deconvolve', action='store_true',
                         help='Disable deconvolution')
    p_batch.set_defaults(func=batch)

    # align command
    p_align = subparsers.add_parser(
        'align',
        help='Align existing images and create GIF',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Align pre-processed images by planet centroid and create GIF.',
        epilog="""
Examples:
  planetary-gif align 'images/*.png' -o jupiter.gif
  planetary-gif align 'processed/*.png' -o output.gif --config config.yaml
""")
    p_align.add_argument('image_pattern',
                         help="Glob pattern for images (use quotes: 'images/*.png')")
    p_align.add_argument('-o', '--output', required=True, metavar='FILE',
                         help='Output GIF path (required)')
    p_align.add_argument('--config', metavar='FILE',
                         help='Config YAML file for GIF settings (fps, crop_padding, ping_pong)')
    p_align.set_defaults(func=align)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Try to find config file if not specified
    if hasattr(args, 'config') and args.config is None:
        args.config = find_config_file()

    args.func(args)


if __name__ == '__main__':
    main()
