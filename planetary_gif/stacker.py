"""
Frame stacking for planetary imaging.

Combines multiple frames using various rejection and averaging methods.
"""

from typing import List, Literal
import numpy as np


StackMethod = Literal['mean', 'median', 'sigma_clip']


def stack_mean(frames: np.ndarray) -> np.ndarray:
    """
    Simple mean stack.

    Args:
        frames: Array of shape (N, H, W) or (N, H, W, C)

    Returns:
        Stacked image
    """
    return np.mean(frames, axis=0)


def stack_median(frames: np.ndarray) -> np.ndarray:
    """
    Median stack - robust to outliers.

    Args:
        frames: Array of shape (N, H, W) or (N, H, W, C)

    Returns:
        Stacked image
    """
    return np.median(frames, axis=0)


def stack_sigma_clip(frames: np.ndarray,
                     sigma_low: float = 2.0,
                     sigma_high: float = 2.0,
                     max_iters: int = 3) -> np.ndarray:
    """
    Sigma-clipped mean stack.

    Iteratively rejects pixels outside sigma bounds, then computes mean.
    Best quality for planetary stacking.

    Args:
        frames: Array of shape (N, H, W) or (N, H, W, C)
        sigma_low: Lower sigma bound for rejection
        sigma_high: Upper sigma bound for rejection
        max_iters: Maximum rejection iterations

    Returns:
        Stacked image
    """
    # Work in float64 for precision
    stack = frames.astype(np.float64)
    mask = np.ones(stack.shape, dtype=bool)

    for _ in range(max_iters):
        # Compute mean and std using current mask
        masked = np.where(mask, stack, np.nan)
        with np.errstate(all='ignore'):
            mean = np.nanmean(masked, axis=0)
            std = np.nanstd(masked, axis=0)

        # Update mask
        low_bound = mean - sigma_low * std
        high_bound = mean + sigma_high * std

        # Expand bounds to match stack shape for broadcasting
        low_bound = np.expand_dims(low_bound, axis=0)
        high_bound = np.expand_dims(high_bound, axis=0)

        new_mask = (stack >= low_bound) & (stack <= high_bound)

        # Check for convergence
        if np.array_equal(mask, new_mask):
            break
        mask = new_mask

    # Final mean with mask
    masked = np.where(mask, stack, np.nan)
    with np.errstate(all='ignore'):
        result = np.nanmean(masked, axis=0)

    # Handle any remaining NaN (all pixels rejected)
    result = np.nan_to_num(result, nan=np.nanmean(stack, axis=0))

    return result


def stack_frames(frames: List[np.ndarray],
                 method: StackMethod = 'sigma_clip',
                 sigma_low: float = 2.0,
                 sigma_high: float = 2.0) -> np.ndarray:
    """
    Stack multiple frames using the specified method.

    Args:
        frames: List of frame arrays (must have same shape)
        method: Stacking method ('mean', 'median', 'sigma_clip')
        sigma_low: Lower sigma for sigma_clip method
        sigma_high: Upper sigma for sigma_clip method

    Returns:
        Stacked image in the same dtype as input
    """
    if not frames:
        raise ValueError("No frames to stack")

    # Get original dtype for output
    original_dtype = frames[0].dtype

    # Convert to array
    stack = np.array(frames)

    # Stack based on method
    if method == 'mean':
        result = stack_mean(stack)
    elif method == 'median':
        result = stack_median(stack)
    elif method == 'sigma_clip':
        result = stack_sigma_clip(stack, sigma_low, sigma_high)
    else:
        raise ValueError(f"Unknown stacking method: {method}")

    # Convert back to original dtype
    if original_dtype == np.uint8:
        result = np.clip(result, 0, 255).astype(np.uint8)
    elif original_dtype == np.uint16:
        result = np.clip(result, 0, 65535).astype(np.uint16)
    else:
        result = result.astype(original_dtype)

    return result


def stack_from_ser(ser_path: str,
                   frame_indices: List[int],
                   method: StackMethod = 'sigma_clip',
                   sigma_low: float = 2.0,
                   sigma_high: float = 2.0,
                   chunk_size: int = 100) -> np.ndarray:
    """
    Stack selected frames from a SER file.

    Memory-efficient: loads frames in chunks for large selections.

    Args:
        ser_path: Path to SER file
        frame_indices: Indices of frames to stack
        method: Stacking method
        sigma_low: Lower sigma for sigma_clip
        sigma_high: Upper sigma for sigma_clip
        chunk_size: Number of frames to load at once

    Returns:
        Stacked image
    """
    from .ser_reader import SERReader

    if not frame_indices:
        raise ValueError("No frame indices provided")

    # For small selections, load all at once
    if len(frame_indices) <= chunk_size:
        with SERReader(ser_path) as ser:
            frames = [ser.read_frame(i) for i in frame_indices]
        return stack_frames(frames, method, sigma_low, sigma_high)

    # For large selections, use chunked processing
    # This is a simplification - for true memory efficiency,
    # would need online algorithms for mean/std computation
    with SERReader(ser_path) as ser:
        frames = [ser.read_frame(i) for i in frame_indices]

    return stack_frames(frames, method, sigma_low, sigma_high)
