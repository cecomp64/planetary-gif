# planetary-gif

## Project Overview

A Python package for processing planetary astrophotography. Takes SER video files, extracts and stacks the best frames, applies wavelet sharpening and deconvolution, then creates aligned rotation GIF animations.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quick Start

```bash
# Process single SER file with auto-tuned sharpening
planetary-gif auto-tune capture.ser -o output.png

# Full pipeline: SER files → stacked/sharpened images → aligned GIF
planetary-gif batch 'videos/*.ser' -o processed/ --gif jupiter.gif --config config.yaml
```

## Recommended Workflow: Auto-Tune → Calibrate → Batch Process

For best results with a batch of SER files, use this iterative workflow:

### Step 1: Run Auto-Tune on a Representative SER File

Pick one of your SER files and run auto-tune with intermediate file saving:

```bash
planetary-gif auto-tune captures/jupiter_001.ser \
    -o tuning/jupiter_tuned.png \
    --verbose \
    --intermediates-dir tuning/intermediates
```

This will:
- Stack frames from the SER file (without sharpening) as a baseline
- Search for optimal sharpening parameters using Bayesian optimization
- Save intermediate results as it finds better configurations
- Enter interactive mode where you can adjust and preview

**Interactive controls:**
- `v` - Save side-by-side comparison image
- `+`/`-` - Increase/decrease sharpening
- `d` - Adjust deconvolution iterations
- `p` - Adjust PSF sigma (higher = gentler, less pixelation)
- `a` - Accept and save
- `q` - Quit without saving

### Step 2: Review Intermediate Results

Browse `tuning/intermediates/` to find results you like:

```
tuning/intermediates/
├── iter_0005_score_0.4123.png
├── iter_0005_score_0.4123.yaml
├── iter_0031_score_0.4786.png
├── iter_0031_score_0.4786.yaml
├── ...
└── best_config.yaml
```

Each intermediate has a matching `.yaml` config file you can use directly.

### Step 3: Calibrate Weights (Optional but Recommended)

If you found an intermediate result that looks better than what auto-tune picked, calibrate the scoring weights to match your preferences:

```bash
# Calibrate using your preferred result as the "good" image
planetary-gif calibrate-weights tuning/intermediates/iter_0031_score_0.4786.png \
    --baseline tuning/jupiter_baseline.png \
    -o tuning/my_weights.yaml

# Optionally specify "bad" images to ensure they score lower
planetary-gif calibrate-weights tuning/good_result.png \
    --baseline tuning/baseline.png \
    --bad tuning/too_pixelated.png \
    --bad tuning/too_soft.png \
    -o tuning/my_weights.yaml
```

### Step 4: Re-run Auto-Tune with Calibrated Weights

Run auto-tune again using your calibrated weights:

```bash
planetary-gif auto-tune captures/jupiter_001.ser \
    -o tuning/jupiter_final.png \
    --load-weights tuning/my_weights.yaml \
    --verbose
```

The tuned config is automatically saved alongside the output (`jupiter_final_config.yaml`).

### Step 5: Batch Process All SER Files

Apply your tuned configuration to all SER files:

```bash
planetary-gif batch 'captures/*.ser' \
    -o processed/ \
    --gif jupiter_rotation.gif \
    --config tuning/jupiter_final_config.yaml
```

This processes each SER file with identical settings and creates an aligned rotation GIF.

## Commands Reference

```bash
# Auto-tune sharpening parameters
planetary-gif auto-tune input.ser -o output.png [options]
  --verbose                  Save intermediate results
  --intermediates-dir DIR    Directory for intermediates (default: auto_tune_intermediates)
  --load-weights FILE        Use calibrated scoring weights
  --max-iterations N         Optimization iterations (default: 50)
  --quick                    Faster search with smaller parameter space
  --accept                   Skip interactive mode, auto-accept result
  --grid-search              Use grid search instead of Bayesian optimization

# Calibrate scoring weights
planetary-gif calibrate-weights good.png --baseline baseline.png -o weights.yaml
  --bad IMAGE                Add known-bad images (can specify multiple)

# Process single image (apply config to existing image)
planetary-gif process-single input.png -o output.png --config config.yaml

# Process single SER file (stack + sharpen)
planetary-gif process-ser capture.ser -o stacked.png --config config.yaml

# Full pipeline: SER files → stacked/sharpened images → aligned GIF
planetary-gif batch 'videos/*.ser' -o processed/ --gif output.gif --config config.yaml

# Align existing images and create GIF
planetary-gif align 'images/*.png' -o output.gif
```

## Configuration

Edit `config.yaml` to tune processing parameters:

- **frame_selection**: `top_percent`, `quality_metric` (laplacian/tenengrad)
- **stacking**: `method` (mean/median/sigma_clip), `sigma_low`, `sigma_high`
- **wavelet**: `enabled`, `coefficients` (per-layer multipliers), `wavelet_type`
- **deconvolution**: `enabled`, `method` (richardson_lucy/wiener), `iterations`, `psf_sigma`
- **contrast**: `method` (none/stretch/clahe), `clip_limit`
- **gif**: `fps`, `crop_padding`, `ping_pong`

CLI flags override config file settings.

## Quality Scoring

Auto-tune optimizes a weighted composite score:

| Metric | Default Weight | Description |
|--------|---------------|-------------|
| Sharpness | 0.35 | Laplacian variance (higher = sharper) |
| Noise | 0.15 | MAD-based noise estimate (lower = cleaner) |
| Artifacts | 0.30 | Over-sharpening, halos, pixelation (lower = better) |
| Contrast | 0.20 | Dynamic range utilization |

Use `calibrate-weights` to adjust these based on your visual preferences.

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
