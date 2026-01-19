"""
Planet alignment via centroid detection.

Detects the planet in each image and shifts to align centroids.
"""

from typing import List, Tuple, Optional
import os

import numpy as np
import cv2


def find_planet_center(image: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    Find the center of a planet using thresholding and moments.

    Uses Otsu's method to threshold the image and finds the centroid
    of the largest bright region.

    Args:
        image: Input image (grayscale or color, any bit depth)

    Returns:
        (cx, cy) tuple of center coordinates, or None if not found
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Normalize to 8-bit for thresholding
    if gray.dtype == np.uint16:
        gray_8bit = (gray / 256).astype(np.uint8)
    else:
        gray_8bit = gray.astype(np.uint8)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray_8bit, (15, 15), 0)

    # Threshold using Otsu's method
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Find the largest contour (should be the planet)
    largest = max(contours, key=cv2.contourArea)

    # Calculate centroid using moments
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None

    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    return (cx, cy)


def align_image(image: np.ndarray,
                reference_center: Tuple[float, float],
                current_center: Optional[Tuple[float, float]] = None) -> Tuple[np.ndarray, float, float]:
    """
    Align a single image to a reference center position.

    Args:
        image: Image to align
        reference_center: Target (cx, cy) position
        current_center: Current planet center, or None to detect

    Returns:
        (aligned_image, dx, dy) - the aligned image and shift applied
    """
    if current_center is None:
        current_center = find_planet_center(image)
        if current_center is None:
            # Can't find planet, return original
            return image, 0.0, 0.0

    # Calculate shift
    dx = reference_center[0] - current_center[0]
    dy = reference_center[1] - current_center[1]

    # Create translation matrix
    h, w = image.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])

    # Apply translation
    aligned = cv2.warpAffine(image, M, (w, h),
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=0)

    return aligned, dx, dy


def align_images(images: List[np.ndarray],
                 reference_index: int = 0) -> Tuple[List[np.ndarray], float, float]:
    """
    Align a list of images to a reference image.

    Args:
        images: List of images to align
        reference_index: Index of reference image (default: first)

    Returns:
        (aligned_images, max_shift_x, max_shift_y)
    """
    if not images:
        raise ValueError("No images to align")

    # Find reference center
    ref_center = find_planet_center(images[reference_index])
    if ref_center is None:
        raise ValueError("Could not find planet in reference image")

    aligned = []
    max_shift_x = 0.0
    max_shift_y = 0.0

    for i, img in enumerate(images):
        if i == reference_index:
            aligned.append(img)
            continue

        aligned_img, dx, dy = align_image(img, ref_center)
        aligned.append(aligned_img)
        max_shift_x = max(max_shift_x, abs(dx))
        max_shift_y = max(max_shift_y, abs(dy))

    return aligned, max_shift_x, max_shift_y


def align_image_files(image_paths: List[str],
                      output_dir: str,
                      reference_index: int = 0,
                      verbose: bool = True) -> Tuple[int, int, int, int]:
    """
    Align image files and save to output directory.

    Args:
        image_paths: List of input image paths
        output_dir: Directory for aligned images
        reference_index: Index of reference image
        verbose: Print progress messages

    Returns:
        (width, height, max_shift_x, max_shift_y)
    """
    if not image_paths:
        raise ValueError("No images to align")

    os.makedirs(output_dir, exist_ok=True)

    # Load reference image
    ref_path = image_paths[reference_index]
    ref_img = cv2.imread(ref_path, cv2.IMREAD_UNCHANGED)
    if ref_img is None:
        raise ValueError(f"Could not load reference image: {ref_path}")

    h, w = ref_img.shape[:2]

    # Find reference center
    ref_center = find_planet_center(ref_img)
    if ref_center is None:
        raise ValueError("Could not find planet in reference image")

    if verbose:
        print(f"Reference image: {os.path.basename(ref_path)}")
        print(f"Image size: {w}x{h}")
        print(f"Reference planet center: ({ref_center[0]:.1f}, {ref_center[1]:.1f})")

    max_shift_x = 0.0
    max_shift_y = 0.0

    for i, img_path in enumerate(image_paths):
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            if verbose:
                print(f"  Warning: Could not load {img_path}, skipping")
            continue

        if i == reference_index:
            # Save reference as-is
            out_path = os.path.join(output_dir, f"aligned_{os.path.basename(img_path)}")
            cv2.imwrite(out_path, img)
            if verbose:
                print(f"  [{i+1}] {os.path.basename(img_path)} -> no shift (reference)")
            continue

        # Align image
        aligned_img, dx, dy = align_image(img, ref_center)
        max_shift_x = max(max_shift_x, abs(dx))
        max_shift_y = max(max_shift_y, abs(dy))

        # Save aligned image
        out_path = os.path.join(output_dir, f"aligned_{os.path.basename(img_path)}")
        cv2.imwrite(out_path, aligned_img)

        if verbose:
            if dx == 0 and dy == 0:
                print(f"  [{i+1}] {os.path.basename(img_path)} -> WARNING: planet not found, copied as-is")
            else:
                print(f"  [{i+1}] {os.path.basename(img_path)} -> shift: dx={dx:+.1f}, dy={dy:+.1f}")

    if verbose:
        print(f"\nAlignment complete! {len(image_paths)} images aligned")
        print(f"Maximum shifts: x={max_shift_x:.1f}, y={max_shift_y:.1f}")

    return w, h, int(np.ceil(max_shift_x)), int(np.ceil(max_shift_y))
