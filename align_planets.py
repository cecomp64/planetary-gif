#!/usr/bin/env python3
"""
Align planetary images and create a ping-pong GIF.
Detects planet centroid, aligns frames, auto-crops based on shifts,
and generates an optimized GIF.
"""

import cv2
import numpy as np
import glob
import os
import sys
import subprocess
import shutil


def find_planet_center(img_gray):
    """Find the center of the planet using thresholding and moments."""
    # Normalize to 8-bit for thresholding
    if img_gray.dtype == np.uint16:
        img_8bit = (img_gray / 256).astype(np.uint8)
    else:
        img_8bit = img_gray.astype(np.uint8)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(img_8bit, (15, 15), 0)

    # Threshold to find bright regions (the planet)
    # Use Otsu's method to automatically find threshold
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


def align_images(image_paths, output_dir):
    """Align images by centering on the planet's position.

    Returns: (max_shift_x, max_shift_y) - maximum absolute shifts applied
    """
    if not image_paths:
        print("No images found!")
        return None

    os.makedirs(output_dir, exist_ok=True)

    # Load reference image (first one)
    ref_path = image_paths[0]
    ref_img = cv2.imread(ref_path, cv2.IMREAD_UNCHANGED)
    if ref_img is None:
        print(f"Error: Could not load reference image {ref_path}")
        return None

    h, w = ref_img.shape[:2]

    # Convert to grayscale for detection
    if len(ref_img.shape) == 3:
        ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    else:
        ref_gray = ref_img

    # Find planet center in reference
    ref_center = find_planet_center(ref_gray)
    if ref_center is None:
        print("Error: Could not find planet in reference image")
        return None

    print(f"Reference image: {os.path.basename(ref_path)}")
    print(f"Image size: {w}x{h}")
    print(f"Reference planet center: ({ref_center[0]:.1f}, {ref_center[1]:.1f})")

    # Save reference image as-is
    out_path = os.path.join(output_dir, f"aligned_{os.path.basename(ref_path)}")
    cv2.imwrite(out_path, ref_img)
    print(f"  [1] {os.path.basename(ref_path)} -> no shift (reference)")

    # Track maximum shifts for cropping
    max_shift_x = 0
    max_shift_y = 0

    # Align each subsequent image
    for i, img_path in enumerate(image_paths[1:], start=2):
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"  Warning: Could not load {img_path}, skipping")
            continue

        # Convert to grayscale
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img

        # Find planet center
        center = find_planet_center(img_gray)
        if center is None:
            print(f"  [{i}] {os.path.basename(img_path)} -> WARNING: planet not found, copying as-is")
            out_path = os.path.join(output_dir, f"aligned_{os.path.basename(img_path)}")
            cv2.imwrite(out_path, img)
            continue

        # Calculate shift needed to align with reference
        dx = ref_center[0] - center[0]
        dy = ref_center[1] - center[1]

        # Track max shifts
        max_shift_x = max(max_shift_x, abs(dx))
        max_shift_y = max(max_shift_y, abs(dy))

        # Create translation matrix
        M = np.float32([[1, 0, dx], [0, 1, dy]])

        # Apply translation
        aligned = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # Save aligned image
        out_path = os.path.join(output_dir, f"aligned_{os.path.basename(img_path)}")
        cv2.imwrite(out_path, aligned)
        print(f"  [{i}] {os.path.basename(img_path)} -> shift: dx={dx:+.1f}, dy={dy:+.1f}")

    print(f"\nAlignment complete! {len(image_paths)} images aligned")
    print(f"Maximum shifts: x={max_shift_x:.1f}, y={max_shift_y:.1f}")

    return w, h, int(np.ceil(max_shift_x)), int(np.ceil(max_shift_y))


def create_gif(aligned_dir, output_path, img_width, img_height, max_shift_x, max_shift_y, fps=10, padding=10):
    """Create a ping-pong GIF from aligned images with auto-crop."""

    # Calculate crop dimensions to remove borders from shifts
    # Add padding to be safe, then ensure we don't exceed image bounds
    crop_margin_x = max_shift_x + padding
    crop_margin_y = max_shift_y + padding

    crop_w = img_width - 2 * crop_margin_x
    crop_h = img_height - 2 * crop_margin_y

    # Ensure even dimensions for video encoding
    crop_w = crop_w - (crop_w % 2)
    crop_h = crop_h - (crop_h % 2)

    print(f"\nCreating GIF:")
    print(f"  Crop margins: {crop_margin_x}px x {crop_margin_y}px")
    print(f"  Output size: {crop_w}x{crop_h}")

    # Temporary palette file
    palette_path = os.path.join(os.path.dirname(output_path), "palette_tmp.png")

    input_pattern = os.path.join(aligned_dir, "aligned_*.png")

    # Build ffmpeg filter for cropping and ping-pong
    crop_filter = f"crop={crop_w}:{crop_h}:(in_w-{crop_w})/2:(in_h-{crop_h})/2"

    # Pass 1: Generate palette
    cmd1 = [
        "ffmpeg", "-y",
        "-pattern_type", "glob", "-i", input_pattern,
        "-filter_complex",
        f"[0:v]{crop_filter},split[fwd][rev];[rev]reverse[rev2];[fwd][rev2]concat=n=2:v=1:a=0,fps={fps},palettegen=stats_mode=full",
        palette_path
    ]

    print("  Generating palette...")
    subprocess.run(cmd1, check=True, capture_output=True)

    # Pass 2: Create GIF with palette
    cmd2 = [
        "ffmpeg", "-y",
        "-pattern_type", "glob", "-i", input_pattern,
        "-i", palette_path,
        "-filter_complex",
        f"[0:v]{crop_filter},split[fwd][rev];[rev]reverse[rev2];[fwd][rev2]concat=n=2:v=1:a=0,fps={fps}[v];[v][1:v]paletteuse=dither=floyd_steinberg",
        "-loop", "0",
        output_path
    ]

    print("  Encoding GIF...")
    subprocess.run(cmd2, check=True, capture_output=True)

    # Cleanup
    os.remove(palette_path)

    # Get file size
    size_kb = os.path.getsize(output_path) / 1024
    print(f"  Done! {output_path} ({size_kb:.1f} KB)")


def main():
    if len(sys.argv) < 3:
        print("Usage: align_planets.py <input_glob_pattern> <output.gif>")
        print("Example: align_planets.py 'images/21_*.png' jupiter.gif")
        sys.exit(1)

    pattern = sys.argv[1]
    output_gif = sys.argv[2]

    # Find and sort images
    image_paths = sorted(glob.glob(pattern))
    print(f"Found {len(image_paths)} images matching '{pattern}'")

    if not image_paths:
        print("No images found!")
        sys.exit(1)

    # Create temp directory for aligned images
    aligned_dir = os.path.join(os.path.dirname(output_gif) or ".", "aligned_tmp")

    # Clean up any previous run
    if os.path.exists(aligned_dir):
        shutil.rmtree(aligned_dir)

    # Align images
    result = align_images(image_paths, aligned_dir)
    if result is None:
        sys.exit(1)

    img_width, img_height, max_shift_x, max_shift_y = result

    # Create GIF
    create_gif(aligned_dir, output_gif, img_width, img_height, max_shift_x, max_shift_y)

    # Cleanup aligned images
    shutil.rmtree(aligned_dir)


if __name__ == "__main__":
    main()
