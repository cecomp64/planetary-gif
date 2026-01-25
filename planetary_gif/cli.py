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
from .config import save_config


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
    if args.contrast:
        config.contrast.method = args.contrast
    if args.contrast_clip:
        config.contrast.clip_limit = args.contrast_clip

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
    if config.contrast.method != 'none':
        print(f"  Contrast: {config.contrast.method}")

    result = sharpen_image(
        image,
        wavelet_enabled=config.wavelet.enabled,
        wavelet_coefficients=config.wavelet.coefficients,
        wavelet_type=config.wavelet.wavelet_type,
        deconv_enabled=config.deconvolution.enabled,
        deconv_method=config.deconvolution.method,
        deconv_iterations=config.deconvolution.iterations,
        psf_sigma=config.deconvolution.psf_sigma,
        contrast_method=config.contrast.method,
        contrast_clip_limit=config.contrast.clip_limit,
        contrast_stretch_low=config.contrast.stretch_low,
        contrast_stretch_high=config.contrast.stretch_high,
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
    if args.contrast:
        config.contrast.method = args.contrast
    if args.contrast_clip:
        config.contrast.clip_limit = args.contrast_clip

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
    if config.contrast.method != 'none':
        print(f"  Contrast: {config.contrast.method}")

    result = sharpen_image(
        stacked,
        wavelet_enabled=config.wavelet.enabled,
        wavelet_coefficients=config.wavelet.coefficients,
        wavelet_type=config.wavelet.wavelet_type,
        deconv_enabled=config.deconvolution.enabled,
        deconv_method=config.deconvolution.method,
        deconv_iterations=config.deconvolution.iterations,
        psf_sigma=config.deconvolution.psf_sigma,
        contrast_method=config.contrast.method,
        contrast_clip_limit=config.contrast.clip_limit,
        contrast_stretch_low=config.contrast.stretch_low,
        contrast_stretch_high=config.contrast.stretch_high,
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
    if args.contrast:
        config.contrast.method = args.contrast
    if args.contrast_clip:
        config.contrast.clip_limit = args.contrast_clip

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
            contrast_method=config.contrast.method,
            contrast_clip_limit=config.contrast.clip_limit,
            contrast_stretch_low=config.contrast.stretch_low,
            contrast_stretch_high=config.contrast.stretch_high,
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


def auto_tune(args):
    """Auto-tune sharpening parameters using quality metrics."""
    from .auto_tune import (
        optimize_parameters,
        compute_composite_score,
        apply_config_to_image,
        create_comparison_image,
        save_config_with_metadata,
        QualityWeights,
        ParameterBounds,
        load_weights,
    )

    config = load_config(args.config)
    input_path = args.input

    # Detect input type (SER vs image)
    is_ser = input_path.lower().endswith('.ser')

    if is_ser:
        # Process SER file: stack without sharpening to create baseline
        print(f"Processing SER file: {input_path}")
        info = get_ser_info(input_path)
        print(f"  Frames: {info['frame_count']}, Size: {info['width']}x{info['height']}, "
              f"Depth: {info['bit_depth']}-bit")

        # Rank frames
        print(f"Ranking frames by {config.frame_selection.quality_metric}...")
        best_indices = rank_frames_from_ser(
            input_path,
            metric=config.frame_selection.quality_metric,
            top_percent=config.frame_selection.top_percent,
            progress_callback=lambda cur, total: print(f"\r  {cur}/{total}", end='', flush=True)
        )
        print(f"\n  Selected {len(best_indices)}/{info['frame_count']} frames "
              f"(top {config.frame_selection.top_percent*100:.0f}%)")

        # Stack frames WITHOUT sharpening - this is our baseline
        print(f"Stacking with {config.stacking.method} (no sharpening)...")
        baseline = stack_from_ser(
            input_path,
            best_indices,
            method=config.stacking.method,
            sigma_low=config.stacking.sigma_low,
            sigma_high=config.stacking.sigma_high,
        )

        # Save baseline if verbose
        if args.verbose:
            baseline_path = args.output.replace('.png', '_baseline.png')
            cv2.imwrite(baseline_path, baseline)
            print(f"Saved baseline: {baseline_path}")

    else:
        # Load image directly as baseline
        print(f"Loading image: {input_path}")
        baseline = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if baseline is None:
            print(f"Error: Could not load image: {input_path}")
            sys.exit(1)

    print(f"Baseline image: {baseline.shape[1]}x{baseline.shape[0]}, {baseline.dtype}")

    # Set up parameter bounds
    bounds = ParameterBounds()
    if args.quick:
        # Reduce search space for faster results
        bounds.deconv_iterations_max = 15
        bounds.wavelet_types = ['gaussian', 'bior1.3']
        bounds.deconv_methods = ['richardson_lucy']
        bounds.contrast_methods = ['none', 'clahe']
        max_iterations = 20
    else:
        max_iterations = args.max_iterations

    # Set up quality weights
    if args.load_weights:
        print(f"Loading calibrated weights from: {args.load_weights}")
        weights = load_weights(args.load_weights)
    elif args.weights:
        weights = QualityWeights(
            sharpness=args.weights[0],
            noise=args.weights[1],
            artifacts=args.weights[2],
            contrast=args.weights[3],
        )
    else:
        weights = QualityWeights()

    # Set up intermediates directory
    intermediates_dir = None
    if args.verbose:
        intermediates_dir = args.intermediates_dir or 'auto_tune_intermediates'
        os.makedirs(intermediates_dir, exist_ok=True)
        print(f"Saving intermediates to: {intermediates_dir}")

    # Progress callback
    def progress(iteration, total, best_score):
        pct = iteration * 100 // total
        bar = '=' * (pct // 5) + '>' + ' ' * (20 - pct // 5)
        print(f"\r  [{bar}] {iteration}/{total} (best: {best_score:.4f})", end='', flush=True)

    # Run optimization
    print(f"\nStarting auto-tune optimization (max {max_iterations} iterations)...")
    print(f"  Weights: sharpness={weights.sharpness}, noise={weights.noise}, "
          f"artifacts={weights.artifacts}, contrast={weights.contrast}")

    optimized_config, opt_info = optimize_parameters(
        baseline,
        base_config=config,
        bounds=bounds,
        weights=weights,
        max_iterations=max_iterations,
        use_bayesian=not args.grid_search,
        progress_callback=progress,
        save_intermediates=intermediates_dir,
    )

    print()  # Newline after progress bar

    # Apply optimized config
    print("\nApplying optimized parameters...")
    result = apply_config_to_image(baseline, optimized_config)

    # Show final metrics
    final_score, metrics = compute_composite_score(result, baseline, weights)
    print(f"\nFinal Quality Metrics:")
    print(f"  Sharpness:  {metrics['sharpness']:.1f} (norm: {metrics['norm_sharpness']:.3f})")
    print(f"  Noise:      {metrics['noise']:.2f} (norm: {metrics['norm_noise']:.3f})")
    print(f"  Artifacts:  {metrics['artifacts']:.2f} (norm: {metrics['norm_artifacts']:.3f})")
    print(f"  Contrast:   {metrics['contrast']:.3f} (norm: {metrics['norm_contrast']:.3f})")
    print(f"  Composite:  {metrics['composite']:.4f}")

    print(f"\nOptimized Parameters:")
    print(f"  Wavelet coefficients: {optimized_config.wavelet.coefficients}")
    print(f"  Wavelet type: {optimized_config.wavelet.wavelet_type}")
    print(f"  Deconv method: {optimized_config.deconvolution.method}")
    print(f"  Deconv iterations: {optimized_config.deconvolution.iterations}")
    print(f"  PSF sigma: {optimized_config.deconvolution.psf_sigma:.2f}")
    print(f"  Contrast: {optimized_config.contrast.method}")
    if optimized_config.contrast.method == 'clahe':
        print(f"  Clip limit: {optimized_config.contrast.clip_limit:.2f}")

    # Interactive adjustment loop
    if not args.accept and sys.stdin.isatty():
        result, optimized_config = interactive_adjustment(
            baseline, result, optimized_config, weights, args
        )
        # Recompute metrics after any adjustments
        final_score, metrics = compute_composite_score(result, baseline, weights)
        opt_info['best_score'] = final_score
        opt_info['best_metrics'] = metrics

    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save output image
    cv2.imwrite(args.output, result)
    print(f"\nSaved: {args.output}")

    # Save config (use --save-config path, or auto-generate from output path)
    config_path = args.save_config or args.output.replace('.png', '_config.yaml')
    config_dir = os.path.dirname(config_path)
    if config_dir:
        os.makedirs(config_dir, exist_ok=True)
    save_config_with_metadata(
        optimized_config,
        config_path,
        input_path,
        opt_info,
    )
    print(f"Config saved: {config_path}")


def calibrate_weights_cmd(args):
    """Calibrate scoring weights using a known-good image."""
    from .auto_tune import (
        calibrate_weights,
        save_weights,
        QualityWeights,
    )

    # Load images
    print(f"Loading good image: {args.good_image}")
    good_image = cv2.imread(args.good_image, cv2.IMREAD_UNCHANGED)
    if good_image is None:
        print(f"Error: Could not load image: {args.good_image}")
        sys.exit(1)

    print(f"Loading baseline: {args.baseline}")
    baseline = cv2.imread(args.baseline, cv2.IMREAD_UNCHANGED)
    if baseline is None:
        print(f"Error: Could not load image: {args.baseline}")
        sys.exit(1)

    # Load bad images if provided
    bad_images = None
    if args.bad:
        bad_images = []
        for bad_path in args.bad:
            print(f"Loading bad image: {bad_path}")
            bad_img = cv2.imread(bad_path, cv2.IMREAD_UNCHANGED)
            if bad_img is None:
                print(f"Error: Could not load image: {bad_path}")
                sys.exit(1)
            bad_images.append(bad_img)

    print(f"\nCalibrating weights...")
    print(f"  Good image: {good_image.shape[1]}x{good_image.shape[0]}")
    print(f"  Baseline: {baseline.shape[1]}x{baseline.shape[0]}")
    if bad_images:
        print(f"  Bad images: {len(bad_images)}")

    # Run calibration
    optimized_weights, calibration_info = calibrate_weights(
        good_image,
        baseline,
        bad_images=bad_images,
    )

    # Show results
    print(f"\nGood image metrics:")
    metrics = calibration_info['good_image_metrics']
    print(f"  Sharpness:  {metrics['sharpness']:.1f} (norm: {metrics['norm_sharpness']:.3f})")
    print(f"  Noise:      {metrics['noise']:.2f} (norm: {metrics['norm_noise']:.3f})")
    print(f"  Artifacts:  {metrics['artifacts']:.2f} (norm: {metrics['norm_artifacts']:.3f})")
    print(f"  Contrast:   {metrics['contrast']:.3f} (norm: {metrics['norm_contrast']:.3f})")

    print(f"\nDefault weights:")
    init = calibration_info['initial_weights']
    print(f"  sharpness={init['sharpness']:.2f}, noise={init['noise']:.2f}, "
          f"artifacts={init['artifacts']:.2f}, contrast={init['contrast']:.2f}")

    print(f"\nCalibrated weights:")
    opt = calibration_info['optimized_weights']
    print(f"  sharpness={opt['sharpness']:.4f}, noise={opt['noise']:.4f}, "
          f"artifacts={opt['artifacts']:.4f}, contrast={opt['contrast']:.4f}")

    print(f"\nGood image score: {calibration_info['good_image_score']:.4f}")
    if calibration_info['bad_image_scores']:
        print(f"Bad image scores: {[f'{s:.4f}' for s in calibration_info['bad_image_scores']]}")

    # Save weights
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    save_weights(optimized_weights, args.output, calibration_info)
    print(f"\nWeights saved: {args.output}")

    print(f"\nTo use these weights:")
    print(f"  planetary-gif auto-tune input.ser -o output.png --load-weights {args.output}")


def interactive_adjustment(baseline, result, config, weights, args):
    """
    Interactive adjustment loop for human feedback.

    Returns (final_image, final_config) after user approval.
    """
    from .auto_tune import (
        apply_config_to_image,
        compute_composite_score,
        create_comparison_image,
    )

    print("\n" + "=" * 60)
    print("INTERACTIVE ADJUSTMENT")
    print("=" * 60)

    while True:
        print("\nOptions:")
        print("  [a] Accept and save")
        print("  [v] Save side-by-side comparison image")
        print("  [+] Increase sharpening (+10% wavelet coefficients)")
        print("  [-] Decrease sharpening (-10% wavelet coefficients)")
        print("  [d] Adjust deconvolution iterations")
        print("  [p] Adjust PSF sigma")
        print("  [c] Cycle contrast method")
        print("  [q] Quit without saving")

        try:
            choice = input("\nChoice: ").strip().lower()
        except EOFError:
            print("\nNon-interactive mode, accepting result")
            return result, config

        if choice == 'a':
            return result, config

        elif choice == 'v':
            comparison = create_comparison_image(baseline, result)
            comp_path = args.output.replace('.png', '_comparison.png')
            comp_dir = os.path.dirname(comp_path)
            if comp_dir:
                os.makedirs(comp_dir, exist_ok=True)
            cv2.imwrite(comp_path, comparison)
            print(f"Saved: {comp_path}")

        elif choice == '+':
            config.wavelet.coefficients = [c * 1.1 for c in config.wavelet.coefficients]
            result = apply_config_to_image(baseline, config)
            score, _ = compute_composite_score(result, baseline, weights)
            print(f"New coefficients: {[f'{c:.2f}' for c in config.wavelet.coefficients]}")
            print(f"New composite score: {score:.4f}")

        elif choice == '-':
            config.wavelet.coefficients = [c * 0.9 for c in config.wavelet.coefficients]
            result = apply_config_to_image(baseline, config)
            score, _ = compute_composite_score(result, baseline, weights)
            print(f"New coefficients: {[f'{c:.2f}' for c in config.wavelet.coefficients]}")
            print(f"New composite score: {score:.4f}")

        elif choice == 'd':
            try:
                new_iter = int(input(f"Deconv iterations [{config.deconvolution.iterations}]: ").strip())
                config.deconvolution.iterations = max(1, new_iter)
                result = apply_config_to_image(baseline, config)
                score, _ = compute_composite_score(result, baseline, weights)
                print(f"New iterations: {config.deconvolution.iterations}")
                print(f"New composite score: {score:.4f}")
            except (ValueError, EOFError):
                print("Invalid input, keeping current value")

        elif choice == 'p':
            try:
                new_psf = float(input(f"PSF sigma [{config.deconvolution.psf_sigma:.2f}]: ").strip())
                config.deconvolution.psf_sigma = max(0.1, new_psf)
                result = apply_config_to_image(baseline, config)
                score, _ = compute_composite_score(result, baseline, weights)
                print(f"New PSF sigma: {config.deconvolution.psf_sigma:.2f}")
                print(f"New composite score: {score:.4f}")
            except (ValueError, EOFError):
                print("Invalid input, keeping current value")

        elif choice == 'c':
            methods = ['none', 'stretch', 'clahe']
            current_idx = methods.index(config.contrast.method) if config.contrast.method in methods else 0
            config.contrast.method = methods[(current_idx + 1) % len(methods)]
            result = apply_config_to_image(baseline, config)
            score, _ = compute_composite_score(result, baseline, weights)
            print(f"Contrast method: {config.contrast.method}")
            print(f"New composite score: {score:.4f}")

        elif choice == 'q':
            print("Quitting without saving.")
            sys.exit(0)

        else:
            print("Unknown option, try again.")


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
    p_single.add_argument('--contrast', choices=['none', 'stretch', 'clahe'],
                          help='Contrast enhancement method (default: none)')
    p_single.add_argument('--contrast-clip', type=float, metavar='N',
                          help='CLAHE clip limit (default: 2.0)')
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
    p_ser.add_argument('--contrast', choices=['none', 'stretch', 'clahe'],
                       help='Contrast enhancement method (default: none)')
    p_ser.add_argument('--contrast-clip', type=float, metavar='N',
                       help='CLAHE clip limit (default: 2.0)')
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
    p_batch.add_argument('--contrast', choices=['none', 'stretch', 'clahe'],
                         help='Contrast enhancement method (default: none)')
    p_batch.add_argument('--contrast-clip', type=float, metavar='N',
                         help='CLAHE clip limit (default: 2.0)')
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

    # auto-tune command
    p_autotune = subparsers.add_parser(
        'auto-tune',
        help='Automatically optimize sharpening parameters',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Find optimal sharpening parameters using quality metrics.',
        epilog="""
Examples:
  # Auto-tune from SER file (stacks first, then optimizes)
  planetary-gif auto-tune capture.ser -o output.png --save-config tuned.yaml

  # Quick mode (fewer iterations)
  planetary-gif auto-tune capture.ser -o output.png --quick --save-config tuned.yaml

  # From pre-stacked image
  planetary-gif auto-tune stacked.png -o output.png --save-config tuned.yaml

  # Non-interactive mode (for scripts)
  planetary-gif auto-tune capture.ser -o output.png --accept --save-config tuned.yaml

  # Verbose mode (save intermediate results)
  planetary-gif auto-tune capture.ser -o output.png --verbose
""")
    p_autotune.add_argument('input',
                            help='Input SER file or pre-stacked image (PNG/TIFF)')
    p_autotune.add_argument('-o', '--output', required=True,
                            help='Output image path (required)')
    p_autotune.add_argument('--save-config', metavar='FILE',
                            help='Save optimized config to YAML file')
    p_autotune.add_argument('--config', metavar='FILE',
                            help='Base config file for frame selection/stacking settings')
    p_autotune.add_argument('--max-iterations', type=int, default=50, metavar='N',
                            help='Maximum optimization iterations (default: 50)')
    p_autotune.add_argument('--quick', action='store_true',
                            help='Quick mode: fewer iterations, smaller search space')
    p_autotune.add_argument('--verbose', action='store_true',
                            help='Save intermediate results during optimization')
    p_autotune.add_argument('--intermediates-dir', metavar='DIR',
                            help='Directory for intermediate images (default: auto_tune_intermediates)')
    p_autotune.add_argument('--accept', action='store_true',
                            help='Auto-accept result without interactive adjustment')
    p_autotune.add_argument('--weights', type=float, nargs=4, metavar='W',
                            help='Custom weights: sharpness noise artifacts contrast (default: 0.35 0.15 0.30 0.20)')
    p_autotune.add_argument('--grid-search', action='store_true',
                            help='Use grid search instead of Bayesian optimization')
    p_autotune.add_argument('--load-weights', metavar='FILE',
                            help='Load calibrated weights from YAML file')
    p_autotune.set_defaults(func=auto_tune)

    # calibrate-weights command
    p_calibrate = subparsers.add_parser(
        'calibrate-weights',
        help='Calibrate scoring weights using a known-good image',
        description="""
Calibrate the quality scoring weights to match your preferences.

Provide a processed image you consider high quality along with its
unprocessed baseline. The tool will optimize the scoring weights
to give your good image a high score.

Optionally provide bad images to ensure they score lower.
""",
        epilog="""
Examples:
  # Basic calibration with good image and baseline
  planetary-gif calibrate-weights good.png --baseline stacked.png -o weights.yaml

  # Calibration with bad images for comparison
  planetary-gif calibrate-weights good.png --baseline stacked.png \\
      --bad oversharpened.png --bad pixelated.png -o weights.yaml

  # Use calibrated weights in auto-tune
  planetary-gif auto-tune input.ser -o output.png --load-weights weights.yaml
""")
    p_calibrate.add_argument('good_image',
                             help='Path to a processed image you consider high quality')
    p_calibrate.add_argument('--baseline', required=True,
                             help='Path to the unprocessed baseline image')
    p_calibrate.add_argument('--bad', action='append', metavar='IMAGE',
                             help='Path to a bad/low-quality image (can specify multiple)')
    p_calibrate.add_argument('-o', '--output', required=True,
                             help='Output path for calibrated weights YAML')
    p_calibrate.set_defaults(func=calibrate_weights_cmd)

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
