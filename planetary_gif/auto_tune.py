"""
Auto-tuning for planetary image sharpening parameters.

Automatically optimizes wavelet, deconvolution, and contrast parameters
using quality metrics to find optimal sharpening settings.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Callable
import os
import itertools

import numpy as np
import cv2

from .frame_quality import compute_sharpness_laplacian, ensure_grayscale, ensure_8bit
from .sharpening import sharpen_image
from .config import Config


@dataclass
class QualityWeights:
    """Weights for composite quality metric."""
    sharpness: float = 0.35
    noise: float = 0.15
    artifacts: float = 0.30  # Penalize over-processing/halos heavily
    contrast: float = 0.20   # Reward natural contrast


@dataclass
class ParameterBounds:
    """Bounds for tunable parameters."""
    wavelet_coeff_min: float = 1.0  # Start at 1.0 to always enhance (not reduce) detail
    wavelet_coeff_max: float = 3.0
    wavelet_types: List[str] = field(default_factory=lambda: ['gaussian', 'bior1.3', 'db1'])

    deconv_iterations_min: int = 5
    deconv_iterations_max: int = 30
    psf_sigma_min: float = 2.0
    psf_sigma_max: float = 5.0
    deconv_methods: List[str] = field(default_factory=lambda: ['richardson_lucy', 'wiener'])

    contrast_methods: List[str] = field(default_factory=lambda: ['none', 'stretch', 'clahe'])
    clip_limit_min: float = 1.0
    clip_limit_max: float = 3.0


def estimate_noise(image: np.ndarray) -> float:
    """
    Estimate noise level using Median Absolute Deviation on high-frequency components.

    Uses MAD estimator which is robust to actual image content.

    Args:
        image: Input image

    Returns:
        Noise estimate (lower = less noise)
    """
    gray = ensure_grayscale(image).astype(np.float64)

    # High-pass filter to isolate noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    high_freq = gray - blurred

    # MAD estimator (robust to outliers from real edges)
    mad = np.median(np.abs(high_freq - np.median(high_freq)))

    # Convert to standard deviation estimate
    sigma_estimate = 1.4826 * mad
    return float(sigma_estimate)


def detect_pixelation(image: np.ndarray) -> float:
    """
    Detect grid/pixelation artifacts using frequency analysis.

    Pixelation appears as regular patterns at specific frequencies,
    particularly at the Nyquist frequency (checkerboard patterns) and
    low frequencies corresponding to block artifacts.

    Args:
        image: Input image

    Returns:
        Pixelation score (0-1, higher = more pixelated)
    """
    gray = ensure_grayscale(image).astype(np.float64)

    # Normalize to 0-1
    max_val = 65535.0 if image.dtype == np.uint16 else 255.0
    gray = gray / max_val

    # Method 1: Detect checkerboard/grid patterns using horizontal and vertical gradients
    # Alternating high-frequency patterns show up as strong responses
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Look for sign changes in gradients (indicates alternating pattern)
    # Compute second derivative to find gradient reversals
    grad_x_diff = np.diff(sobel_x, axis=1)
    grad_y_diff = np.diff(sobel_y, axis=0)

    # Count sign changes - high count indicates grid pattern
    sign_changes_x = np.sum(grad_x_diff[:-1, :] * grad_x_diff[1:, :] < 0)
    sign_changes_y = np.sum(grad_y_diff[:, :-1] * grad_y_diff[:, 1:] < 0)

    # Normalize by image size
    total_pixels = gray.size
    sign_change_ratio = (sign_changes_x + sign_changes_y) / (2 * total_pixels)

    # Method 2: FFT analysis for periodic patterns
    # Grid artifacts show peaks at regular intervals in frequency domain
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)

    # Look at high frequencies (outer regions of FFT)
    h, w = magnitude.shape
    center_h, center_w = h // 2, w // 2

    # Create mask for high frequencies (exclude DC and low freq)
    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - center_w)**2 + (y - center_h)**2)
    min_radius = min(h, w) * 0.3  # Skip low frequencies
    max_radius = min(h, w) * 0.5  # Focus on mid-high frequencies

    high_freq_mask = (dist_from_center > min_radius) & (dist_from_center < max_radius)

    # Check for strong peaks in high frequencies (indicates periodic artifacts)
    if np.sum(high_freq_mask) > 0:
        high_freq_mag = magnitude[high_freq_mask]
        # Ratio of max to median indicates presence of periodic patterns
        peak_ratio = np.max(high_freq_mag) / (np.median(high_freq_mag) + 1e-10)
        # Normalize: values > 10 indicate strong periodic content
        fft_score = min(peak_ratio / 50.0, 1.0)
    else:
        fft_score = 0.0

    # Method 3: Local variance uniformity
    # Pixelated images have unnaturally uniform local variance
    kernel_size = 3
    local_mean = cv2.blur(gray, (kernel_size, kernel_size))
    local_sq_mean = cv2.blur(gray**2, (kernel_size, kernel_size))
    local_var = local_sq_mean - local_mean**2
    local_var = np.maximum(local_var, 0)  # Ensure non-negative

    # Coefficient of variation of local variance
    # Very uniform variance across the image suggests artificial processing
    var_of_var = np.std(local_var) / (np.mean(local_var) + 1e-10)
    # Low var_of_var = suspicious uniformity, but this is inverted
    # Natural images have var_of_var > 1, pixelated might be lower
    uniformity_score = max(0, 1.0 - var_of_var) if var_of_var < 1.0 else 0.0

    # Combine scores with weights
    pixelation_score = (
        sign_change_ratio * 2.0 +  # Grid pattern detection
        fft_score * 0.5 +          # Periodic artifact detection
        uniformity_score * 0.3     # Unnatural uniformity
    )

    return float(min(pixelation_score, 1.0))


def detect_artifacts(processed: np.ndarray, baseline: np.ndarray) -> float:
    """
    Detect over-sharpening artifacts by measuring how much the image changed.

    Uses multiple indicators:
    1. Overall intensity change from baseline (over-processing)
    2. Local contrast amplification (halos/ringing)
    3. Grid/pixelation artifacts

    Args:
        processed: Sharpened image
        baseline: Original unsharpened image

    Returns:
        Artifact severity score (0-1 normalized, lower = fewer artifacts)
    """
    gray_proc = ensure_grayscale(processed).astype(np.float64)
    gray_base = ensure_grayscale(baseline).astype(np.float64)

    # Normalize to 0-1 range for consistent scoring
    max_val = 65535.0 if processed.dtype == np.uint16 else 255.0
    gray_proc = gray_proc / max_val
    gray_base = gray_base / max_val

    # 1. Measure overall change from baseline (over-processing indicator)
    diff = np.abs(gray_proc - gray_base)
    mean_change = np.mean(diff)

    # 2. Measure local contrast amplification using Laplacian
    # High Laplacian in processed vs baseline = potential halos/ringing
    lap_proc = cv2.Laplacian(gray_proc, cv2.CV_64F)
    lap_base = cv2.Laplacian(gray_base, cv2.CV_64F)

    # How much did local contrast increase?
    lap_diff = np.abs(lap_proc) - np.abs(lap_base)
    contrast_amplification = np.mean(np.maximum(lap_diff, 0))  # Only count increases

    # 3. Check for extreme values (clipping/saturation from over-processing)
    near_black = np.sum(gray_proc < 0.02) / gray_proc.size
    near_white = np.sum(gray_proc > 0.98) / gray_proc.size
    clipping_penalty = (near_black + near_white) * 5  # Weight clipping heavily

    # 4. Detect grid/pixelation artifacts
    pixelation = detect_pixelation(processed)
    # Also check if pixelation increased compared to baseline
    baseline_pixelation = detect_pixelation(baseline)
    pixelation_increase = max(0, pixelation - baseline_pixelation)

    # Combine into artifact score (0-1 scale)
    # Higher values = more artifacts
    artifact_score = (
        mean_change * 3.0 +             # Overall change
        contrast_amplification * 10.0 + # Local contrast boost
        clipping_penalty +              # Clipping
        pixelation_increase * 5.0       # Pixelation introduced by processing
    )

    # Cap at 1.0
    return float(min(artifact_score, 1.0))


def compute_dynamic_range(image: np.ndarray) -> float:
    """
    Compute effective dynamic range / histogram spread.

    Higher values indicate better tonal distribution.

    Args:
        image: Input image

    Returns:
        Dynamic range score (0-1, higher = better spread)
    """
    gray = ensure_grayscale(image).astype(np.float64)

    # Interquartile range
    p25 = np.percentile(gray, 25)
    p75 = np.percentile(gray, 75)
    iqr = p75 - p25

    # Full range usage
    p1 = np.percentile(gray, 1)
    p99 = np.percentile(gray, 99)
    range_usage = p99 - p1

    # Normalize by bit depth
    max_val = 65535.0 if image.dtype == np.uint16 else 255.0
    normalized_iqr = iqr / max_val
    normalized_range = range_usage / max_val

    return float(0.5 * normalized_iqr + 0.5 * normalized_range)


def compute_composite_score(
    processed: np.ndarray,
    baseline: np.ndarray,
    weights: Optional[QualityWeights] = None
) -> Tuple[float, Dict[str, float]]:
    """
    Compute weighted composite quality score.

    Args:
        processed: Sharpened image
        baseline: Original unsharpened image (for artifact detection)
        weights: Quality weights (uses defaults if None)

    Returns:
        (composite_score, {metric_name: individual_score})
    """
    if weights is None:
        weights = QualityWeights()

    # Compute individual metrics
    sharpness = compute_sharpness_laplacian(processed)
    noise = estimate_noise(processed)
    artifacts = detect_artifacts(processed, baseline)
    contrast = compute_dynamic_range(processed)

    # Normalize metrics to [0, 1] range
    # These normalization values are empirically tuned
    norm_sharpness = min(sharpness / 5000.0, 1.0)
    norm_noise = 1.0 - min(noise / 20.0, 1.0)       # Invert: less noise = higher score
    norm_artifacts = 1.0 - artifacts  # Already 0-1, invert: fewer artifacts = higher score
    norm_contrast = contrast  # Already normalized

    # Weighted combination
    composite = (
        weights.sharpness * norm_sharpness +
        weights.noise * norm_noise +
        weights.artifacts * norm_artifacts +
        weights.contrast * norm_contrast
    )

    metrics = {
        'sharpness': sharpness,
        'noise': noise,
        'artifacts': artifacts,
        'contrast': contrast,
        'norm_sharpness': norm_sharpness,
        'norm_noise': norm_noise,
        'norm_artifacts': norm_artifacts,
        'norm_contrast': norm_contrast,
        'composite': composite,
    }

    return composite, metrics


def params_to_config(params: Dict[str, Any]) -> Config:
    """Convert parameter dictionary to Config object."""
    config = Config()

    config.wavelet.enabled = True
    config.wavelet.coefficients = [
        float(params.get('w_coef_0', 1.5)),
        float(params.get('w_coef_1', 1.2)),
        float(params.get('w_coef_2', 1.0)),
        float(params.get('w_coef_3', 1.0)),
    ]
    # Convert numpy string to native Python string
    config.wavelet.wavelet_type = str(params.get('wavelet_type', 'gaussian'))

    config.deconvolution.enabled = True
    config.deconvolution.method = str(params.get('deconv_method', 'richardson_lucy'))
    config.deconvolution.iterations = int(params.get('deconv_iter', 10))
    config.deconvolution.psf_sigma = float(params.get('psf_sigma', 1.5))

    config.contrast.method = str(params.get('contrast_method', 'none'))
    config.contrast.clip_limit = float(params.get('clip_limit', 2.0))

    return config


def save_intermediate_config(
    config: Config,
    output_dir: str,
    iteration: int,
    score: float,
    metrics: Dict[str, float],
) -> None:
    """
    Save config file for an intermediate result.

    Args:
        config: Configuration to save
        output_dir: Directory to save to
        iteration: Iteration number
        score: Composite score
        metrics: Quality metrics dict
    """
    import yaml

    config_path = os.path.join(output_dir, f"iter_{iteration:04d}_score_{score:.4f}.yaml")

    header_lines = [
        "# Auto-tune intermediate result",
        f"# Iteration: {iteration}",
        f"# Composite score: {score:.4f}",
        "#",
        "# Metrics:",
        f"#   Sharpness: {metrics.get('sharpness', 0):.1f}",
        f"#   Noise: {metrics.get('noise', 0):.2f}",
        f"#   Artifacts: {metrics.get('artifacts', 0):.2f}",
        f"#   Contrast: {metrics.get('contrast', 0):.3f}",
        "#",
    ]
    header = "\n".join(header_lines) + "\n"

    config_dict = config.to_dict()
    with open(config_path, 'w') as f:
        f.write(header)
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def apply_config_to_image(image: np.ndarray, config: Config) -> np.ndarray:
    """Apply sharpening config to image."""
    return sharpen_image(
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


def grid_search_optimize(
    baseline: np.ndarray,
    bounds: ParameterBounds,
    weights: QualityWeights,
    max_iterations: int = 50,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
    save_intermediates: Optional[str] = None,
) -> Tuple[Config, Dict[str, Any]]:
    """
    Optimize parameters using grid search with random sampling.

    Args:
        baseline: Unsharpened input image
        bounds: Parameter bounds
        weights: Quality metric weights
        max_iterations: Maximum number of iterations to run
        progress_callback: Optional callback(iteration, total, best_score)
        save_intermediates: Directory to save intermediate images (or None)

    Returns:
        (best_config, optimization_history)
    """
    import random
    random.seed(42)

    # Generate random samples within bounds
    combinations = []
    for _ in range(max_iterations):
        params = (
            random.uniform(bounds.wavelet_coeff_min, bounds.wavelet_coeff_max),  # w_coef_0
            random.uniform(bounds.wavelet_coeff_min, bounds.wavelet_coeff_max),  # w_coef_1
            random.choice(bounds.wavelet_types),
            random.randint(bounds.deconv_iterations_min, bounds.deconv_iterations_max),
            random.uniform(bounds.psf_sigma_min, bounds.psf_sigma_max),
            random.choice(bounds.deconv_methods),
            random.choice(bounds.contrast_methods),
            random.uniform(bounds.clip_limit_min, bounds.clip_limit_max),
        )
        combinations.append(params)

    total = len(combinations)
    best_score = -float('inf')
    best_config = None
    best_metrics = None
    history = []

    if save_intermediates:
        os.makedirs(save_intermediates, exist_ok=True)

    for i, (w0, w1, wtype, d_iter, psf, d_method, c_method, c_clip) in enumerate(combinations):
        params = {
            'w_coef_0': w0,
            'w_coef_1': w1,
            'w_coef_2': 1.0,
            'w_coef_3': 1.0,
            'wavelet_type': wtype,
            'deconv_iter': d_iter,
            'psf_sigma': psf,
            'deconv_method': d_method,
            'contrast_method': c_method,
            'clip_limit': c_clip,
        }

        config = params_to_config(params)
        processed = apply_config_to_image(baseline, config)
        score, metrics = compute_composite_score(processed, baseline, weights)

        history.append({
            'iteration': i,
            'params': params,
            'score': score,
            'metrics': metrics,
        })

        if score > best_score:
            best_score = score
            best_config = config
            best_metrics = metrics

            if save_intermediates:
                path = os.path.join(save_intermediates, f"iter_{i:04d}_score_{score:.4f}.png")
                cv2.imwrite(path, processed)
                save_intermediate_config(config, save_intermediates, i, score, metrics)

        if progress_callback:
            progress_callback(i + 1, total, best_score)

    # Save final best config
    if save_intermediates and best_config is not None:
        final_config_path = os.path.join(save_intermediates, "best_config.yaml")
        save_config_with_metadata(
            best_config,
            final_config_path,
            "auto-tune",
            {
                'method': 'grid_search',
                'iterations': total,
                'best_score': best_score,
                'best_metrics': best_metrics,
            }
        )

    return best_config, {
        'method': 'grid_search',
        'iterations': total,
        'best_score': best_score,
        'best_metrics': best_metrics,
        'history': history,
    }


def bayesian_optimize(
    baseline: np.ndarray,
    bounds: ParameterBounds,
    weights: QualityWeights,
    max_iterations: int = 50,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
    save_intermediates: Optional[str] = None,
) -> Tuple[Config, Dict[str, Any]]:
    """
    Optimize parameters using Bayesian optimization with scikit-optimize.

    Args:
        baseline: Unsharpened input image
        bounds: Parameter bounds
        weights: Quality metric weights
        max_iterations: Maximum optimization iterations
        progress_callback: Optional callback(iteration, total, best_score)
        save_intermediates: Directory to save intermediate images (or None)

    Returns:
        (best_config, optimization_history)
    """
    try:
        from skopt import gp_minimize
        from skopt.space import Real, Integer, Categorical
    except ImportError:
        # Fall back to random search
        print("  scikit-optimize not installed, using random search")
        return grid_search_optimize(
            baseline, bounds, weights,
            max_iterations=max_iterations,
            progress_callback=progress_callback,
            save_intermediates=save_intermediates,
        )

    if save_intermediates:
        os.makedirs(save_intermediates, exist_ok=True)

    # Define search space
    space = [
        Real(bounds.wavelet_coeff_min, bounds.wavelet_coeff_max, name='w_coef_0'),
        Real(bounds.wavelet_coeff_min, bounds.wavelet_coeff_max, name='w_coef_1'),
        Categorical(bounds.wavelet_types, name='wavelet_type'),
        Integer(bounds.deconv_iterations_min, bounds.deconv_iterations_max, name='deconv_iter'),
        Real(bounds.psf_sigma_min, bounds.psf_sigma_max, name='psf_sigma'),
        Categorical(bounds.deconv_methods, name='deconv_method'),
        Categorical(bounds.contrast_methods, name='contrast_method'),
        Real(bounds.clip_limit_min, bounds.clip_limit_max, name='clip_limit'),
    ]

    history = []
    best_score = -float('inf')
    iteration_count = [0]  # Use list to allow modification in nested function

    def objective(x):
        w0, w1, wtype, d_iter, psf, d_method, c_method, c_clip = x

        params = {
            'w_coef_0': w0,
            'w_coef_1': w1,
            'w_coef_2': 1.0,
            'w_coef_3': 1.0,
            'wavelet_type': wtype,
            'deconv_iter': d_iter,
            'psf_sigma': psf,
            'deconv_method': d_method,
            'contrast_method': c_method,
            'clip_limit': c_clip,
        }

        config = params_to_config(params)
        processed = apply_config_to_image(baseline, config)
        score, metrics = compute_composite_score(processed, baseline, weights)

        iteration_count[0] += 1
        history.append({
            'iteration': iteration_count[0],
            'params': params,
            'score': score,
            'metrics': metrics,
        })

        nonlocal best_score
        if score > best_score:
            best_score = score
            if save_intermediates:
                path = os.path.join(save_intermediates, f"iter_{iteration_count[0]:04d}_score_{score:.4f}.png")
                cv2.imwrite(path, processed)
                save_intermediate_config(config, save_intermediates, iteration_count[0], score, metrics)

        if progress_callback:
            progress_callback(iteration_count[0], max_iterations, best_score)

        return -score  # Minimize negative score

    # Run Bayesian optimization
    result = gp_minimize(
        objective,
        space,
        n_calls=max_iterations,
        n_initial_points=min(10, max_iterations // 3),
        random_state=42,
        verbose=False,
    )

    # Get best config
    best_x = result.x
    best_params = {
        'w_coef_0': best_x[0],
        'w_coef_1': best_x[1],
        'w_coef_2': 1.0,
        'w_coef_3': 1.0,
        'wavelet_type': best_x[2],
        'deconv_iter': best_x[3],
        'psf_sigma': best_x[4],
        'deconv_method': best_x[5],
        'contrast_method': best_x[6],
        'clip_limit': best_x[7],
    }
    best_config = params_to_config(best_params)

    # Compute final metrics
    processed = apply_config_to_image(baseline, best_config)
    final_score, final_metrics = compute_composite_score(processed, baseline, weights)

    # Save final best config
    if save_intermediates:
        final_config_path = os.path.join(save_intermediates, "best_config.yaml")
        save_config_with_metadata(
            best_config,
            final_config_path,
            "auto-tune",
            {
                'method': 'bayesian',
                'iterations': max_iterations,
                'best_score': -result.fun,
                'best_metrics': final_metrics,
            }
        )

    return best_config, {
        'method': 'bayesian',
        'iterations': max_iterations,
        'best_score': -result.fun,
        'best_metrics': final_metrics,
        'history': history,
    }


def optimize_parameters(
    baseline: np.ndarray,
    base_config: Optional[Config] = None,
    bounds: Optional[ParameterBounds] = None,
    weights: Optional[QualityWeights] = None,
    max_iterations: int = 50,
    use_bayesian: bool = True,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
    save_intermediates: Optional[str] = None,
) -> Tuple[Config, Dict[str, Any]]:
    """
    Optimize sharpening parameters for the given baseline image.

    Args:
        baseline: Unsharpened stacked image
        base_config: Starting configuration (uses defaults if None)
        bounds: Parameter bounds (uses defaults if None)
        weights: Quality metric weights (uses defaults if None)
        max_iterations: Maximum optimization iterations
        use_bayesian: Use Bayesian optimization (falls back to grid if unavailable)
        progress_callback: Optional callback(iteration, total, best_score)
        save_intermediates: Directory to save intermediate images

    Returns:
        (optimized_config, optimization_history)
    """
    if bounds is None:
        bounds = ParameterBounds()
    if weights is None:
        weights = QualityWeights()

    if use_bayesian:
        return bayesian_optimize(
            baseline, bounds, weights,
            max_iterations=max_iterations,
            progress_callback=progress_callback,
            save_intermediates=save_intermediates,
        )
    else:
        return grid_search_optimize(
            baseline, bounds, weights,
            grid_points=4,
            progress_callback=progress_callback,
            save_intermediates=save_intermediates,
        )


def create_comparison_image(baseline: np.ndarray, processed: np.ndarray) -> np.ndarray:
    """
    Create a side-by-side comparison image.

    Args:
        baseline: Original image
        processed: Processed image

    Returns:
        Comparison image with baseline on left, processed on right
    """
    # Ensure same dtype
    if baseline.dtype != processed.dtype:
        if baseline.dtype == np.uint16:
            processed = processed.astype(np.uint16)
        else:
            baseline = baseline.astype(processed.dtype)

    # Add labels
    h, w = baseline.shape[:2]
    label_height = 30

    # Create canvas
    if len(baseline.shape) == 3:
        canvas = np.zeros((h + label_height, w * 2 + 10, baseline.shape[2]), dtype=baseline.dtype)
    else:
        canvas = np.zeros((h + label_height, w * 2 + 10), dtype=baseline.dtype)

    # Place images
    canvas[label_height:label_height + h, :w] = baseline
    canvas[label_height:label_height + h, w + 10:] = processed

    return canvas


def save_config_with_metadata(
    config: Config,
    output_path: str,
    input_file: str,
    optimization_info: Dict[str, Any],
) -> None:
    """
    Save configuration with auto-tune metadata as comments.

    Args:
        config: Configuration to save
        output_path: Output YAML file path
        input_file: Original input file name
        optimization_info: Optimization results
    """
    import yaml

    # Build metadata header
    header_lines = [
        "# Auto-tuned configuration",
        f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"# Input: {os.path.basename(input_file)}",
        f"# Optimization method: {optimization_info.get('method', 'unknown')}",
        f"# Iterations: {optimization_info.get('iterations', 'unknown')}",
        f"# Composite score: {optimization_info.get('best_score', 0):.4f}",
        "#",
    ]

    # Add metrics if available
    metrics = optimization_info.get('best_metrics', {})
    if metrics:
        header_lines.extend([
            "# Quality metrics:",
            f"#   Sharpness: {metrics.get('sharpness', 0):.1f}",
            f"#   Noise: {metrics.get('noise', 0):.2f}",
            f"#   Artifacts: {metrics.get('artifacts', 0):.2f}",
            f"#   Contrast: {metrics.get('contrast', 0):.3f}",
            "#",
        ])

    header = "\n".join(header_lines) + "\n"

    # Convert config to dict and dump
    config_dict = config.to_dict()

    with open(output_path, 'w') as f:
        f.write(header)
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def calibrate_weights(
    good_image: np.ndarray,
    baseline: np.ndarray,
    bad_images: Optional[List[np.ndarray]] = None,
    initial_weights: Optional[QualityWeights] = None,
    max_iterations: int = 100,
) -> Tuple[QualityWeights, Dict[str, Any]]:
    """
    Calibrate scoring weights to give a known-good image a high score.

    Uses optimization to find weights that maximize the score of the good image
    while optionally minimizing scores of known-bad images.

    Args:
        good_image: A processed image you consider high quality
        baseline: The unprocessed baseline image
        bad_images: Optional list of images you consider low quality
        initial_weights: Starting weights (uses defaults if None)
        max_iterations: Maximum optimization iterations

    Returns:
        (optimized_weights, calibration_info)
    """
    from scipy.optimize import minimize

    if initial_weights is None:
        initial_weights = QualityWeights()

    # Compute raw metrics for the good image
    good_sharpness = compute_sharpness_laplacian(good_image)
    good_noise = estimate_noise(good_image)
    good_artifacts = detect_artifacts(good_image, baseline)
    good_contrast = compute_dynamic_range(good_image)

    # Normalize metrics
    norm_good_sharpness = min(good_sharpness / 5000.0, 1.0)
    norm_good_noise = 1.0 - min(good_noise / 20.0, 1.0)
    norm_good_artifacts = 1.0 - good_artifacts
    norm_good_contrast = good_contrast

    good_metrics = {
        'sharpness': good_sharpness,
        'noise': good_noise,
        'artifacts': good_artifacts,
        'contrast': good_contrast,
        'norm_sharpness': norm_good_sharpness,
        'norm_noise': norm_good_noise,
        'norm_artifacts': norm_good_artifacts,
        'norm_contrast': norm_good_contrast,
    }

    # Compute metrics for bad images if provided
    bad_metrics_list = []
    if bad_images:
        for bad_img in bad_images:
            bad_sharpness = compute_sharpness_laplacian(bad_img)
            bad_noise = estimate_noise(bad_img)
            bad_artifacts = detect_artifacts(bad_img, baseline)
            bad_contrast = compute_dynamic_range(bad_img)

            bad_metrics_list.append({
                'norm_sharpness': min(bad_sharpness / 5000.0, 1.0),
                'norm_noise': 1.0 - min(bad_noise / 20.0, 1.0),
                'norm_artifacts': 1.0 - bad_artifacts,
                'norm_contrast': bad_contrast,
            })

    def compute_score(weights_array):
        """Compute weighted score from weights array [sharpness, noise, artifacts, contrast]."""
        return (
            weights_array[0] * norm_good_sharpness +
            weights_array[1] * norm_good_noise +
            weights_array[2] * norm_good_artifacts +
            weights_array[3] * norm_good_contrast
        )

    def compute_bad_score(weights_array, bad_m):
        """Compute score for a bad image."""
        return (
            weights_array[0] * bad_m['norm_sharpness'] +
            weights_array[1] * bad_m['norm_noise'] +
            weights_array[2] * bad_m['norm_artifacts'] +
            weights_array[3] * bad_m['norm_contrast']
        )

    def objective(weights_array):
        """
        Objective: maximize good image score, minimize bad image scores.
        Returns negative because scipy minimizes.
        """
        good_score = compute_score(weights_array)

        # Penalty for bad images scoring high
        bad_penalty = 0.0
        if bad_metrics_list:
            for bad_m in bad_metrics_list:
                bad_score = compute_bad_score(weights_array, bad_m)
                # Penalize if bad score is close to or higher than good score
                margin = good_score - bad_score
                if margin < 0.1:  # Want at least 0.1 margin
                    bad_penalty += (0.1 - margin) ** 2

        # We want to maximize good_score and minimize bad_penalty
        # Also add small regularization to keep weights balanced
        regularization = 0.01 * np.var(weights_array)

        return -good_score + bad_penalty + regularization

    # Initial weights as array
    x0 = np.array([
        initial_weights.sharpness,
        initial_weights.noise,
        initial_weights.artifacts,
        initial_weights.contrast,
    ])

    # Constraints: weights must be positive and sum to 1
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},  # Sum to 1
    ]
    bounds = [(0.01, 0.8)] * 4  # Each weight between 0.01 and 0.8

    # Optimize
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': max_iterations},
    )

    # Extract optimized weights
    optimized = QualityWeights(
        sharpness=float(result.x[0]),
        noise=float(result.x[1]),
        artifacts=float(result.x[2]),
        contrast=float(result.x[3]),
    )

    # Compute final scores
    final_good_score = compute_score(result.x)
    final_bad_scores = []
    if bad_metrics_list:
        for bad_m in bad_metrics_list:
            final_bad_scores.append(compute_bad_score(result.x, bad_m))

    calibration_info = {
        'good_image_metrics': good_metrics,
        'good_image_score': final_good_score,
        'bad_image_scores': final_bad_scores,
        'optimization_success': result.success,
        'optimization_message': result.message,
        'initial_weights': {
            'sharpness': initial_weights.sharpness,
            'noise': initial_weights.noise,
            'artifacts': initial_weights.artifacts,
            'contrast': initial_weights.contrast,
        },
        'optimized_weights': {
            'sharpness': optimized.sharpness,
            'noise': optimized.noise,
            'artifacts': optimized.artifacts,
            'contrast': optimized.contrast,
        },
    }

    return optimized, calibration_info


def save_weights(weights: QualityWeights, output_path: str, calibration_info: Optional[Dict] = None) -> None:
    """
    Save calibrated weights to a YAML file.

    Args:
        weights: The weights to save
        output_path: Output file path
        calibration_info: Optional calibration info to include as comments
    """
    import yaml

    header_lines = [
        "# Calibrated quality weights for auto-tune",
        f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ]

    if calibration_info:
        header_lines.extend([
            "#",
            f"# Good image score: {calibration_info.get('good_image_score', 0):.4f}",
        ])
        if calibration_info.get('bad_image_scores'):
            header_lines.append(f"# Bad image scores: {[f'{s:.4f}' for s in calibration_info['bad_image_scores']]}")
        header_lines.append("#")

    header = "\n".join(header_lines) + "\n"

    weights_dict = {
        'quality_weights': {
            'sharpness': round(weights.sharpness, 4),
            'noise': round(weights.noise, 4),
            'artifacts': round(weights.artifacts, 4),
            'contrast': round(weights.contrast, 4),
        }
    }

    with open(output_path, 'w') as f:
        f.write(header)
        yaml.dump(weights_dict, f, default_flow_style=False, sort_keys=False)


def load_weights(weights_path: str) -> QualityWeights:
    """
    Load calibrated weights from a YAML file.

    Args:
        weights_path: Path to weights YAML file

    Returns:
        QualityWeights object
    """
    import yaml

    with open(weights_path, 'r') as f:
        data = yaml.safe_load(f)

    w = data.get('quality_weights', {})
    return QualityWeights(
        sharpness=w.get('sharpness', 0.35),
        noise=w.get('noise', 0.15),
        artifacts=w.get('artifacts', 0.30),
        contrast=w.get('contrast', 0.20),
    )
