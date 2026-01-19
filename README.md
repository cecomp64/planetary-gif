# planetary-gif

## Project Overview

A Python package for processing planetary astrophotography. Takes SER video files, extracts and stacks the best frames, applies wavelet sharpening and deconvolution, then creates aligned rotation GIF animations.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Commands

```bash
# Process single image (for parameter tuning)
planetary-gif process-single input.png -o output.png --config config.yaml

# Process single SER file (stack + sharpen)
planetary-gif process-ser capture.ser -o stacked.png --config config.yaml

# Full pipeline: SER files → stacked/sharpened images → aligned GIF
planetary-gif batch 'videos/*.ser' -o processed/ --gif jupiter.gif --config config.yaml

# Align existing images and create GIF (legacy workflow)
planetary-gif align 'images/*.png' -o output.gif
```

## Configuration

Edit `config.yaml` to tune processing parameters:

- **frame_selection**: `top_percent`, `quality_metric` (laplacian/tenengrad)
- **stacking**: `method` (mean/median/sigma_clip), `sigma_low`, `sigma_high`
- **wavelet**: `enabled`, `coefficients` (per-layer multipliers), `wavelet_type`
- **deconvolution**: `enabled`, `method` (richardson_lucy/wiener), `iterations`, `psf_sigma`
- **gif**: `fps`, `crop_padding`, `ping_pong`

CLI flags override config file settings.

## Package Structure

```
planetary_gif/
├── ser_reader.py      # SER file parser (178-byte header + raw frames)
├── frame_quality.py   # Sharpness metrics (Laplacian variance, Tenengrad)
├── stacker.py         # Frame stacking (mean, median, sigma-clipped)
├── sharpening.py      # Wavelet (à-trous) + Richardson-Lucy deconvolution
├── alignment.py       # Planet centroid detection and image alignment
├── gif.py             # ffmpeg two-pass palette-optimized GIF creation
├── config.py          # YAML config loading
└── cli.py             # Command-line interface
```

## Processing Pipeline

1. **SER Reading**: Parse header, extract frames, debayer if needed
2. **Quality Ranking**: Score frames by Laplacian variance, select top N%
3. **Stacking**: Sigma-clipped mean to reject outliers
4. **Wavelet Sharpening**: Multi-scale detail enhancement
5. **Deconvolution**: Richardson-Lucy with Gaussian PSF
6. **Alignment**: Detect planet centroid, shift to align
7. **GIF Creation**: ffmpeg with palette optimization, auto-crop

## Dependencies

- numpy, opencv-python, PyWavelets, scikit-image, PyYAML
- ffmpeg (external, for GIF creation)
