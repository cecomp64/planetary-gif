"""
GIF creation for planetary rotation animations.

Uses ffmpeg for high-quality palette-based GIF encoding.
"""

import os
import subprocess
from typing import Optional


def create_gif(input_pattern: str,
               output_path: str,
               img_width: int,
               img_height: int,
               max_shift_x: int = 0,
               max_shift_y: int = 0,
               fps: int = 10,
               crop_padding: int = 10,
               ping_pong: bool = True,
               verbose: bool = True) -> str:
    """
    Create a GIF from aligned images with auto-crop.

    Uses two-pass ffmpeg encoding with palette generation for
    high-quality output.

    Args:
        input_pattern: Glob pattern for input images (e.g., "aligned/*.png")
        output_path: Output GIF path
        img_width: Original image width
        img_height: Original image height
        max_shift_x: Maximum X shift from alignment (for crop calculation)
        max_shift_y: Maximum Y shift from alignment (for crop calculation)
        fps: Frames per second
        crop_padding: Extra pixels to crop beyond alignment shifts
        ping_pong: If True, play forward then reverse
        verbose: Print progress messages

    Returns:
        Path to created GIF
    """
    # Calculate crop dimensions to remove borders from shifts
    crop_margin_x = max_shift_x + crop_padding
    crop_margin_y = max_shift_y + crop_padding

    crop_w = img_width - 2 * crop_margin_x
    crop_h = img_height - 2 * crop_margin_y

    # Ensure positive dimensions
    crop_w = max(crop_w, 100)
    crop_h = max(crop_h, 100)

    # Ensure even dimensions for video encoding
    crop_w = crop_w - (crop_w % 2)
    crop_h = crop_h - (crop_h % 2)

    if verbose:
        print(f"\nCreating GIF:")
        print(f"  Crop margins: {crop_margin_x}px x {crop_margin_y}px")
        print(f"  Output size: {crop_w}x{crop_h}")

    # Temporary palette file
    output_dir = os.path.dirname(output_path) or "."
    palette_path = os.path.join(output_dir, "palette_tmp.png")

    # Build ffmpeg filter for cropping
    crop_filter = f"crop={crop_w}:{crop_h}:(in_w-{crop_w})/2:(in_h-{crop_h})/2"

    if ping_pong:
        # Ping-pong: play forward, then reverse
        filter_base = f"[0:v]{crop_filter},split[fwd][rev];[rev]reverse[rev2];[fwd][rev2]concat=n=2:v=1:a=0"
    else:
        # Simple forward playback
        filter_base = f"[0:v]{crop_filter}"

    # Pass 1: Generate palette
    cmd1 = [
        "ffmpeg", "-y",
        "-pattern_type", "glob", "-i", input_pattern,
        "-filter_complex",
        f"{filter_base},fps={fps},palettegen=stats_mode=full",
        palette_path
    ]

    if verbose:
        print("  Generating palette...")

    try:
        subprocess.run(cmd1, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg palette generation failed: {e.stderr.decode()}")

    # Pass 2: Create GIF with palette
    if ping_pong:
        filter_with_palette = f"{filter_base},fps={fps}[v];[v][1:v]paletteuse=dither=floyd_steinberg"
    else:
        filter_with_palette = f"{filter_base},fps={fps}[v];[v][1:v]paletteuse=dither=floyd_steinberg"

    cmd2 = [
        "ffmpeg", "-y",
        "-pattern_type", "glob", "-i", input_pattern,
        "-i", palette_path,
        "-filter_complex", filter_with_palette,
        "-loop", "0",
        output_path
    ]

    if verbose:
        print("  Encoding GIF...")

    try:
        subprocess.run(cmd2, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg GIF encoding failed: {e.stderr.decode()}")

    # Cleanup palette
    if os.path.exists(palette_path):
        os.remove(palette_path)

    # Report result
    if verbose:
        size_kb = os.path.getsize(output_path) / 1024
        print(f"  Done! {output_path} ({size_kb:.1f} KB)")

    return output_path


def create_gif_from_images(images_dir: str,
                           output_path: str,
                           pattern: str = "aligned_*.png",
                           **kwargs) -> str:
    """
    Create a GIF from images in a directory.

    Convenience wrapper that detects image dimensions automatically.

    Args:
        images_dir: Directory containing aligned images
        output_path: Output GIF path
        pattern: Glob pattern for images within the directory
        **kwargs: Additional arguments passed to create_gif

    Returns:
        Path to created GIF
    """
    import glob
    import cv2

    # Build full pattern
    full_pattern = os.path.join(images_dir, pattern)

    # Get image dimensions from first image
    image_files = sorted(glob.glob(full_pattern))
    if not image_files:
        raise ValueError(f"No images found matching: {full_pattern}")

    first_img = cv2.imread(image_files[0], cv2.IMREAD_UNCHANGED)
    if first_img is None:
        raise ValueError(f"Could not read image: {image_files[0]}")

    h, w = first_img.shape[:2]

    return create_gif(
        input_pattern=full_pattern,
        output_path=output_path,
        img_width=w,
        img_height=h,
        **kwargs
    )
