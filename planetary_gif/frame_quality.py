"""
Frame quality assessment for planetary imaging.

Provides sharpness metrics to rank frames for stacking.
"""

from typing import List, Tuple, Callable
import numpy as np
import cv2


def ensure_grayscale(frame: np.ndarray) -> np.ndarray:
    """Convert frame to grayscale if needed."""
    if len(frame.shape) == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


def ensure_8bit(frame: np.ndarray) -> np.ndarray:
    """Convert to 8-bit for OpenCV operations that require it."""
    if frame.dtype == np.uint16:
        return (frame / 256).astype(np.uint8)
    return frame.astype(np.uint8)


def compute_sharpness_laplacian(frame: np.ndarray) -> float:
    """
    Compute sharpness using variance of Laplacian.

    Fast and reliable general-purpose sharpness metric.
    Higher values indicate sharper images.

    Args:
        frame: Input image (grayscale or color)

    Returns:
        Sharpness score (higher = sharper)
    """
    gray = ensure_grayscale(frame)
    # Use float64 for Laplacian to avoid overflow
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def compute_sharpness_tenengrad(frame: np.ndarray) -> float:
    """
    Compute sharpness using Tenengrad (gradient magnitude).

    Better for planetary edges, slightly slower than Laplacian.
    Higher values indicate sharper images.

    Args:
        frame: Input image (grayscale or color)

    Returns:
        Sharpness score (higher = sharper)
    """
    gray = ensure_grayscale(frame)
    # Sobel gradients
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    # Gradient magnitude squared, averaged
    return float(np.mean(gx**2 + gy**2))


def compute_sharpness_brenner(frame: np.ndarray) -> float:
    """
    Compute sharpness using Brenner gradient.

    Simple and fast focus measure.

    Args:
        frame: Input image (grayscale or color)

    Returns:
        Sharpness score (higher = sharper)
    """
    gray = ensure_grayscale(frame).astype(np.float64)
    # Horizontal difference with step of 2
    diff = gray[:, 2:] - gray[:, :-2]
    return float(np.mean(diff**2))


def compute_contrast(frame: np.ndarray) -> float:
    """
    Compute RMS contrast.

    Useful for filtering hazy or low-contrast frames.

    Args:
        frame: Input image

    Returns:
        Contrast score (higher = more contrast)
    """
    gray = ensure_grayscale(frame).astype(np.float64)
    return float(np.std(gray))


# Available quality metrics
QUALITY_METRICS = {
    'laplacian': compute_sharpness_laplacian,
    'tenengrad': compute_sharpness_tenengrad,
    'brenner': compute_sharpness_brenner,
    'contrast': compute_contrast,
}


def score_frames(frames: List[np.ndarray],
                 metric: str = 'laplacian') -> List[Tuple[int, float]]:
    """
    Score all frames using the specified quality metric.

    Args:
        frames: List of frame arrays
        metric: Quality metric name ('laplacian', 'tenengrad', 'brenner', 'contrast')

    Returns:
        List of (index, score) tuples, sorted by score descending
    """
    if metric not in QUALITY_METRICS:
        raise ValueError(f"Unknown metric '{metric}'. Available: {list(QUALITY_METRICS.keys())}")

    metric_func = QUALITY_METRICS[metric]
    scores = [(i, metric_func(frame)) for i, frame in enumerate(frames)]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def rank_frames(frames: List[np.ndarray],
                metric: str = 'laplacian',
                top_percent: float = 0.1) -> List[int]:
    """
    Rank frames and return indices of the best ones.

    Args:
        frames: List of frame arrays
        metric: Quality metric name
        top_percent: Fraction of best frames to return (0.0-1.0)

    Returns:
        List of frame indices (best frames)
    """
    scores = score_frames(frames, metric)
    n_select = max(1, int(len(frames) * top_percent))
    return [idx for idx, _ in scores[:n_select]]


def rank_frames_from_ser(ser_path: str,
                         metric: str = 'laplacian',
                         top_percent: float = 0.1,
                         progress_callback: Callable[[int, int], None] = None) -> List[int]:
    """
    Rank frames directly from a SER file without loading all into memory.

    Args:
        ser_path: Path to SER file
        metric: Quality metric name
        top_percent: Fraction of best frames to return
        progress_callback: Optional callback(current, total) for progress reporting

    Returns:
        List of frame indices (best frames)
    """
    from .ser_reader import SERReader

    metric_func = QUALITY_METRICS.get(metric)
    if metric_func is None:
        raise ValueError(f"Unknown metric '{metric}'. Available: {list(QUALITY_METRICS.keys())}")

    scores = []
    with SERReader(ser_path) as ser:
        total = ser.frame_count
        for i, frame in enumerate(ser.iter_frames()):
            score = metric_func(frame)
            scores.append((i, score))
            if progress_callback:
                progress_callback(i + 1, total)

    scores.sort(key=lambda x: x[1], reverse=True)
    n_select = max(1, int(len(scores) * top_percent))
    return [idx for idx, _ in scores[:n_select]]
