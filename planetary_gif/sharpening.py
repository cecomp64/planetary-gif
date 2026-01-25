"""
Image sharpening for planetary astrophotography.

Provides wavelet-based sharpening, deconvolution, and contrast enhancement.
"""

from typing import List, Optional, Literal
import numpy as np
import pywt
import cv2
from skimage.restoration import richardson_lucy, wiener


DeconvMethod = Literal['richardson_lucy', 'wiener']


def create_gaussian_psf(sigma: float = 1.5, size: Optional[int] = None) -> np.ndarray:
    """
    Create a 2D Gaussian point spread function.

    Args:
        sigma: Standard deviation of Gaussian
        size: Kernel size (odd number). If None, auto-calculated as 6*sigma.

    Returns:
        Normalized PSF array
    """
    if size is None:
        size = int(6 * sigma) | 1  # Ensure odd

    x = np.arange(size) - size // 2
    xx, yy = np.meshgrid(x, x)
    psf = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return psf / psf.sum()


def normalize_image(image: np.ndarray) -> tuple:
    """
    Normalize image to [0, 1] range and return original range.

    Returns:
        (normalized_image, min_val, max_val)
    """
    min_val = image.min()
    max_val = image.max()
    if max_val - min_val < 1e-10:
        return image.astype(np.float64), min_val, max_val
    normalized = (image.astype(np.float64) - min_val) / (max_val - min_val)
    return normalized, min_val, max_val


def denormalize_image(image: np.ndarray, min_val: float, max_val: float,
                      dtype: np.dtype) -> np.ndarray:
    """
    Restore image from [0, 1] to original range and dtype.
    """
    restored = image * (max_val - min_val) + min_val
    if dtype == np.uint8:
        return np.clip(restored, 0, 255).astype(np.uint8)
    elif dtype == np.uint16:
        return np.clip(restored, 0, 65535).astype(np.uint16)
    return restored.astype(dtype)


def wavelet_sharpen(image: np.ndarray,
                    coefficients: List[float] = [1.5, 1.2, 1.0, 1.0],
                    wavelet_type: str = 'bior1.3') -> np.ndarray:
    """
    Apply à-trous wavelet sharpening similar to Siril/RegiStax.

    Decomposes image into wavelet layers and applies per-layer
    enhancement coefficients.

    Args:
        image: Input image (grayscale or color, any bit depth)
        coefficients: Multipliers for each wavelet layer (finest to coarsest)
                     Values > 1.0 enhance detail, < 1.0 suppress, 1.0 preserve
        wavelet_type: Wavelet basis (e.g., 'bior1.3', 'db1', 'haar', 'gaussian')
                     'gaussian' uses Gaussian-Laplacian pyramid decomposition

    Returns:
        Sharpened image in same dtype as input
    """
    original_dtype = image.dtype
    is_color = len(image.shape) == 3

    # Use Gaussian pyramid for 'gaussian' wavelet type
    if wavelet_type == 'gaussian':
        if is_color:
            channels = [image[:, :, i] for i in range(image.shape[2])]
            sharpened_channels = [
                _gaussian_wavelet_sharpen_channel(ch, coefficients)
                for ch in channels
            ]
            result = np.stack(sharpened_channels, axis=2)
        else:
            result = _gaussian_wavelet_sharpen_channel(image, coefficients)
    else:
        if is_color:
            # Process each channel separately
            channels = [image[:, :, i] for i in range(image.shape[2])]
            sharpened_channels = [
                _wavelet_sharpen_channel(ch, coefficients, wavelet_type)
                for ch in channels
            ]
            result = np.stack(sharpened_channels, axis=2)
        else:
            result = _wavelet_sharpen_channel(image, coefficients, wavelet_type)

    # Result is in original value range, just need to clip and cast
    if original_dtype == np.uint8:
        return np.clip(result, 0, 255).astype(np.uint8)
    elif original_dtype == np.uint16:
        return np.clip(result, 0, 65535).astype(np.uint16)
    return result


def _gaussian_wavelet_sharpen_channel(channel: np.ndarray,
                                       coefficients: List[float]) -> np.ndarray:
    """
    Apply Gaussian-Laplacian pyramid sharpening to a single channel.

    Uses successive Gaussian blurs to decompose the image into detail
    layers at different scales, similar to the à-trous algorithm but
    using Gaussian kernels instead of discrete wavelets.

    This approach is used in tools like GIMP's wavelet decompose,
    RegiStax, and AstroSurface.
    """
    # Normalize to float
    normalized, min_val, max_val = normalize_image(channel)

    levels = len(coefficients)

    # Build Gaussian pyramid - each level is progressively blurred
    # Sigma doubles at each level (à-trous style)
    pyramid = [normalized]
    for i in range(levels):
        # Kernel size should be ~6*sigma and odd
        sigma = 2 ** i  # 1, 2, 4, 8, ...
        ksize = int(6 * sigma) | 1  # Ensure odd
        ksize = max(3, ksize)  # Minimum kernel size of 3
        blurred = cv2.GaussianBlur(pyramid[-1], (ksize, ksize), sigma)
        pyramid.append(blurred)

    # Extract detail layers (Laplacian pyramid)
    # Each detail layer = current level - next blurred level
    detail_layers = []
    for i in range(levels):
        detail = pyramid[i] - pyramid[i + 1]
        detail_layers.append(detail)

    # Residual (lowest frequency component)
    residual = pyramid[-1]

    # Apply coefficients to detail layers and reconstruct
    result = residual.copy()
    for i, (detail, coef) in enumerate(zip(detail_layers, coefficients)):
        result += detail * coef

    # Clip and restore range
    result = np.clip(result, 0, 1)
    return result * (max_val - min_val) + min_val


def _wavelet_sharpen_channel(channel: np.ndarray,
                             coefficients: List[float],
                             wavelet_type: str) -> np.ndarray:
    """Process a single channel with wavelet sharpening."""
    # Normalize to [0, 1]
    normalized, min_val, max_val = normalize_image(channel)

    levels = len(coefficients)

    try:
        # Use stationary wavelet transform (à-trous)
        coeffs = pywt.swt2(normalized, wavelet_type, level=levels)

        # Modify detail coefficients
        modified_coeffs = []
        for i, (cA, (cH, cV, cD)) in enumerate(coeffs):
            # Map level index to coefficient
            # swt2 returns coarsest first, so reverse the coefficient index
            level_idx = levels - 1 - i
            coef = coefficients[level_idx] if level_idx < len(coefficients) else 1.0

            # Apply coefficient to detail bands
            modified_details = (cH * coef, cV * coef, cD * coef)
            modified_coeffs.append((cA, modified_details))

        # Reconstruct
        result = pywt.iswt2(modified_coeffs, wavelet_type)

    except ValueError:
        # Fallback for images that don't fit SWT requirements
        # Use standard DWT instead
        result = _wavelet_sharpen_dwt(normalized, coefficients, wavelet_type)

    # Clip and restore range
    result = np.clip(result, 0, 1)
    return result * (max_val - min_val) + min_val


def _wavelet_sharpen_dwt(channel: np.ndarray,
                         coefficients: List[float],
                         wavelet_type: str) -> np.ndarray:
    """Fallback using standard DWT for non-dyadic images."""
    levels = len(coefficients)

    # Pad image to power of 2 if needed
    h, w = channel.shape
    new_h = int(2 ** np.ceil(np.log2(h)))
    new_w = int(2 ** np.ceil(np.log2(w)))

    if new_h != h or new_w != w:
        padded = np.zeros((new_h, new_w), dtype=channel.dtype)
        padded[:h, :w] = channel
    else:
        padded = channel

    # Decompose
    coeffs = pywt.wavedec2(padded, wavelet_type, level=levels)

    # Modify coefficients
    # coeffs = [cAn, (cHn, cVn, cDn), ..., (cH1, cV1, cD1)]
    modified = [coeffs[0]]  # Keep approximation
    for i, (cH, cV, cD) in enumerate(coeffs[1:]):
        level_idx = levels - 1 - i
        coef = coefficients[level_idx] if level_idx < len(coefficients) else 1.0
        modified.append((cH * coef, cV * coef, cD * coef))

    # Reconstruct
    result = pywt.waverec2(modified, wavelet_type)

    # Remove padding
    return result[:h, :w]


def deconvolve_rl(image: np.ndarray,
                  iterations: int = 10,
                  psf_sigma: float = 1.5,
                  psf: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Apply Richardson-Lucy deconvolution.

    Args:
        image: Input image (grayscale or color)
        iterations: Number of RL iterations (5-15 typical for planetary)
        psf_sigma: Sigma for auto-generated Gaussian PSF
        psf: Optional custom PSF array

    Returns:
        Deconvolved image in same dtype as input
    """
    original_dtype = image.dtype
    is_color = len(image.shape) == 3

    if psf is None:
        psf = create_gaussian_psf(psf_sigma)

    if is_color:
        # Process each channel
        channels = [image[:, :, i] for i in range(image.shape[2])]
        deconv_channels = [
            _deconvolve_channel_rl(ch, psf, iterations)
            for ch in channels
        ]
        result = np.stack(deconv_channels, axis=2)
    else:
        result = _deconvolve_channel_rl(image, psf, iterations)

    # Result is in [0, 1] range, convert to original dtype
    if original_dtype == np.uint8:
        return np.clip(result * 255, 0, 255).astype(np.uint8)
    elif original_dtype == np.uint16:
        return np.clip(result * 65535, 0, 65535).astype(np.uint16)
    return result


def _deconvolve_channel_rl(channel: np.ndarray,
                           psf: np.ndarray,
                           iterations: int) -> np.ndarray:
    """Apply RL deconvolution to a single channel."""
    normalized, min_val, max_val = normalize_image(channel)

    # Add small epsilon to avoid division by zero
    normalized = np.clip(normalized, 1e-10, 1.0)

    # Apply Richardson-Lucy
    result = richardson_lucy(normalized, psf, num_iter=iterations, clip=True)

    return np.clip(result, 0, 1)


def deconvolve_wiener(image: np.ndarray,
                      psf_sigma: float = 1.5,
                      psf: Optional[np.ndarray] = None,
                      balance: float = 0.1) -> np.ndarray:
    """
    Apply Wiener filter deconvolution.

    More stable than RL but may produce less sharpening.

    Args:
        image: Input image (grayscale or color)
        psf_sigma: Sigma for auto-generated Gaussian PSF
        psf: Optional custom PSF array
        balance: Regularization parameter (higher = smoother)

    Returns:
        Deconvolved image in same dtype as input
    """
    original_dtype = image.dtype
    is_color = len(image.shape) == 3

    if psf is None:
        psf = create_gaussian_psf(psf_sigma)

    if is_color:
        channels = [image[:, :, i] for i in range(image.shape[2])]
        deconv_channels = [
            _deconvolve_channel_wiener(ch, psf, balance)
            for ch in channels
        ]
        result = np.stack(deconv_channels, axis=2)
    else:
        result = _deconvolve_channel_wiener(image, psf, balance)

    # Result is in [0, 1] range, convert to original dtype
    if original_dtype == np.uint8:
        return np.clip(result * 255, 0, 255).astype(np.uint8)
    elif original_dtype == np.uint16:
        return np.clip(result * 65535, 0, 65535).astype(np.uint16)
    return result


def _deconvolve_channel_wiener(channel: np.ndarray,
                               psf: np.ndarray,
                               balance: float) -> np.ndarray:
    """Apply Wiener deconvolution to a single channel."""
    normalized, min_val, max_val = normalize_image(channel)

    # Apply Wiener filter
    result = wiener(normalized, psf, balance=balance)

    return np.clip(result, 0, 1)


ContrastMethod = Literal['none', 'stretch', 'clahe']


def enhance_contrast(image: np.ndarray,
                     method: ContrastMethod = 'clahe',
                     clip_limit: float = 2.0,
                     stretch_low: float = 0.5,
                     stretch_high: float = 99.5) -> np.ndarray:
    """
    Enhance image contrast.

    Args:
        image: Input image (grayscale or color, any bit depth)
        method: Contrast method - 'none', 'stretch', or 'clahe'
        clip_limit: CLAHE clip limit (higher = more contrast, default 2.0)
        stretch_low: Percentile for black point in stretch (default 0.5)
        stretch_high: Percentile for white point in stretch (default 99.5)

    Returns:
        Contrast-enhanced image in same dtype as input
    """
    if method == 'none':
        return image

    original_dtype = image.dtype
    is_color = len(image.shape) == 3

    if method == 'stretch':
        return _stretch_contrast(image, stretch_low, stretch_high)
    elif method == 'clahe':
        return _apply_clahe(image, clip_limit)

    return image


def _stretch_contrast(image: np.ndarray,
                      low_percentile: float = 0.5,
                      high_percentile: float = 99.5) -> np.ndarray:
    """Apply histogram stretch based on percentiles."""
    original_dtype = image.dtype
    is_color = len(image.shape) == 3

    if is_color:
        channels = [image[:, :, i] for i in range(image.shape[2])]
        stretched = [
            _stretch_channel(ch, low_percentile, high_percentile)
            for ch in channels
        ]
        result = np.stack(stretched, axis=2)
    else:
        result = _stretch_channel(image, low_percentile, high_percentile)

    if original_dtype == np.uint8:
        return np.clip(result, 0, 255).astype(np.uint8)
    elif original_dtype == np.uint16:
        return np.clip(result, 0, 65535).astype(np.uint16)
    return result


def _stretch_channel(channel: np.ndarray,
                     low_percentile: float,
                     high_percentile: float) -> np.ndarray:
    """Stretch a single channel based on percentiles."""
    low_val = np.percentile(channel, low_percentile)
    high_val = np.percentile(channel, high_percentile)

    if high_val - low_val < 1e-10:
        return channel.astype(np.float64)

    # Stretch to full range
    stretched = (channel.astype(np.float64) - low_val) / (high_val - low_val)

    if channel.dtype == np.uint8:
        return stretched * 255
    elif channel.dtype == np.uint16:
        return stretched * 65535
    return stretched


def _apply_clahe(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

    For color images, converts to LAB color space and applies CLAHE
    only to the luminance channel to preserve color balance.
    """
    original_dtype = image.dtype
    is_color = len(image.shape) == 3

    # Convert to 8-bit for CLAHE (OpenCV requirement)
    if original_dtype == np.uint16:
        work_image = (image / 256).astype(np.uint8)
    else:
        work_image = image.copy()

    # Create CLAHE object
    # Tile grid size of 8x8 works well for planetary images
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))

    if is_color:
        # Convert BGR to LAB
        lab = cv2.cvtColor(work_image, cv2.COLOR_BGR2LAB)
        # Apply CLAHE to L channel only
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        # Convert back to BGR
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        result = clahe.apply(work_image)

    # Convert back to original bit depth
    if original_dtype == np.uint16:
        return (result.astype(np.uint16) * 256)
    return result


def sharpen_image(image: np.ndarray,
                  wavelet_enabled: bool = True,
                  wavelet_coefficients: List[float] = [1.5, 1.2, 1.0, 1.0],
                  wavelet_type: str = 'bior1.3',
                  deconv_enabled: bool = True,
                  deconv_method: DeconvMethod = 'richardson_lucy',
                  deconv_iterations: int = 10,
                  psf_sigma: float = 1.5,
                  contrast_method: ContrastMethod = 'none',
                  contrast_clip_limit: float = 2.0,
                  contrast_stretch_low: float = 0.5,
                  contrast_stretch_high: float = 99.5) -> np.ndarray:
    """
    Apply full sharpening pipeline (wavelet + deconvolution + contrast).

    Args:
        image: Input image
        wavelet_enabled: Whether to apply wavelet sharpening
        wavelet_coefficients: Wavelet layer multipliers
        wavelet_type: Wavelet basis function
        deconv_enabled: Whether to apply deconvolution
        deconv_method: Deconvolution method
        deconv_iterations: RL iterations (if using richardson_lucy)
        psf_sigma: PSF sigma for deconvolution
        contrast_method: Contrast enhancement - 'none', 'stretch', or 'clahe'
        contrast_clip_limit: CLAHE clip limit (default 2.0)
        contrast_stretch_low: Percentile for black point (default 0.5)
        contrast_stretch_high: Percentile for white point (default 99.5)

    Returns:
        Sharpened image
    """
    result = image

    if wavelet_enabled:
        result = wavelet_sharpen(result, wavelet_coefficients, wavelet_type)

    if deconv_enabled:
        if deconv_method == 'richardson_lucy':
            result = deconvolve_rl(result, deconv_iterations, psf_sigma)
        elif deconv_method == 'wiener':
            result = deconvolve_wiener(result, psf_sigma)

    if contrast_method != 'none':
        result = enhance_contrast(
            result,
            method=contrast_method,
            clip_limit=contrast_clip_limit,
            stretch_low=contrast_stretch_low,
            stretch_high=contrast_stretch_high,
        )

    return result
