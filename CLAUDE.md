# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project creates ping-pong GIF animations from planetary astrophotography images. It aligns frames by detecting the planet's centroid and auto-crops to remove borders caused by alignment shifts.

## Commands

**Run the full pipeline (align + GIF):**
```bash
./make_jupiter.sh
```

**Run directly with Python:**
```bash
python3 align_planets.py 'images/21_*.png' images/jupiter_pingpong.gif
```

## Dependencies

- Python 3 with OpenCV (`cv2`) and NumPy
- ffmpeg (for GIF creation with palette optimization)

## Architecture

### align_planets.py

The main script that handles the complete pipeline:

1. **Planet detection** (`find_planet_center`): Uses Otsu thresholding to find bright regions, finds the largest contour (the planet), and calculates centroid via image moments.

2. **Alignment** (`align_images`): Shifts each frame so the planet centroid matches the reference (first) image. Tracks maximum shifts for later cropping.

3. **GIF creation** (`create_gif`): Uses ffmpeg two-pass encoding with palette generation for high-quality output. Auto-crops based on maximum alignment shifts plus padding. Creates ping-pong (forward + reverse) animation.

### align_and_export.ssf

Siril script file (not currently used in the main pipeline). Note: Siril CLI's `register` command only supports star-based alignment, not planetary alignment via cross-correlation—that feature is GUI-only.

## Image Naming Convention

Source images should match a glob pattern like `21_*.png` (timestamp-based naming from astrophotography capture software).
