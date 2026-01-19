"""
Configuration management for planetary-gif.

Loads settings from YAML config files and merges with CLI overrides.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any
import os

import yaml


@dataclass
class FrameSelectionConfig:
    """Frame selection settings."""
    top_percent: float = 0.10
    quality_metric: str = 'laplacian'


@dataclass
class StackingConfig:
    """Stacking settings."""
    method: str = 'sigma_clip'
    sigma_low: float = 2.0
    sigma_high: float = 2.0


@dataclass
class WaveletConfig:
    """Wavelet sharpening settings."""
    enabled: bool = True
    coefficients: List[float] = field(default_factory=lambda: [1.5, 1.2, 1.0, 1.0])
    wavelet_type: str = 'bior1.3'


@dataclass
class DeconvolutionConfig:
    """Deconvolution settings."""
    enabled: bool = True
    method: str = 'richardson_lucy'
    iterations: int = 10
    psf_sigma: float = 1.5


@dataclass
class ContrastConfig:
    """Contrast enhancement settings."""
    method: str = 'none'  # 'none', 'stretch', or 'clahe'
    clip_limit: float = 2.0  # CLAHE clip limit
    stretch_low: float = 0.5  # Percentile for black point
    stretch_high: float = 99.5  # Percentile for white point


@dataclass
class OutputConfig:
    """Output settings."""
    bit_depth: int = 16
    format: str = 'png'


@dataclass
class GifConfig:
    """GIF animation settings."""
    fps: int = 10
    crop_padding: int = 10
    ping_pong: bool = True


@dataclass
class Config:
    """Complete configuration for planetary-gif processing."""
    frame_selection: FrameSelectionConfig = field(default_factory=FrameSelectionConfig)
    stacking: StackingConfig = field(default_factory=StackingConfig)
    wavelet: WaveletConfig = field(default_factory=WaveletConfig)
    deconvolution: DeconvolutionConfig = field(default_factory=DeconvolutionConfig)
    contrast: ContrastConfig = field(default_factory=ContrastConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    gif: GifConfig = field(default_factory=GifConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'frame_selection': {
                'top_percent': self.frame_selection.top_percent,
                'quality_metric': self.frame_selection.quality_metric,
            },
            'stacking': {
                'method': self.stacking.method,
                'sigma_low': self.stacking.sigma_low,
                'sigma_high': self.stacking.sigma_high,
            },
            'wavelet': {
                'enabled': self.wavelet.enabled,
                'coefficients': self.wavelet.coefficients,
                'wavelet_type': self.wavelet.wavelet_type,
            },
            'deconvolution': {
                'enabled': self.deconvolution.enabled,
                'method': self.deconvolution.method,
                'iterations': self.deconvolution.iterations,
                'psf_sigma': self.deconvolution.psf_sigma,
            },
            'contrast': {
                'method': self.contrast.method,
                'clip_limit': self.contrast.clip_limit,
                'stretch_low': self.contrast.stretch_low,
                'stretch_high': self.contrast.stretch_high,
            },
            'output': {
                'bit_depth': self.output.bit_depth,
                'format': self.output.format,
            },
            'gif': {
                'fps': self.gif.fps,
                'crop_padding': self.gif.crop_padding,
                'ping_pong': self.gif.ping_pong,
            },
        }


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to config file, or None to use defaults

    Returns:
        Config object with loaded settings
    """
    config = Config()

    if config_path is None:
        return config

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, 'r') as f:
        data = yaml.safe_load(f)

    if data is None:
        return config

    # Update frame_selection
    if 'frame_selection' in data:
        fs = data['frame_selection']
        if 'top_percent' in fs:
            config.frame_selection.top_percent = float(fs['top_percent'])
        if 'quality_metric' in fs:
            config.frame_selection.quality_metric = fs['quality_metric']

    # Update stacking
    if 'stacking' in data:
        st = data['stacking']
        if 'method' in st:
            config.stacking.method = st['method']
        if 'sigma_low' in st:
            config.stacking.sigma_low = float(st['sigma_low'])
        if 'sigma_high' in st:
            config.stacking.sigma_high = float(st['sigma_high'])

    # Update wavelet
    if 'wavelet' in data:
        wv = data['wavelet']
        if 'enabled' in wv:
            config.wavelet.enabled = bool(wv['enabled'])
        if 'coefficients' in wv:
            config.wavelet.coefficients = [float(c) for c in wv['coefficients']]
        if 'wavelet_type' in wv:
            config.wavelet.wavelet_type = wv['wavelet_type']

    # Update deconvolution
    if 'deconvolution' in data:
        dc = data['deconvolution']
        if 'enabled' in dc:
            config.deconvolution.enabled = bool(dc['enabled'])
        if 'method' in dc:
            config.deconvolution.method = dc['method']
        if 'iterations' in dc:
            config.deconvolution.iterations = int(dc['iterations'])
        if 'psf_sigma' in dc:
            config.deconvolution.psf_sigma = float(dc['psf_sigma'])

    # Update contrast
    if 'contrast' in data:
        ct = data['contrast']
        if 'method' in ct:
            config.contrast.method = ct['method']
        if 'clip_limit' in ct:
            config.contrast.clip_limit = float(ct['clip_limit'])
        if 'stretch_low' in ct:
            config.contrast.stretch_low = float(ct['stretch_low'])
        if 'stretch_high' in ct:
            config.contrast.stretch_high = float(ct['stretch_high'])

    # Update output
    if 'output' in data:
        out = data['output']
        if 'bit_depth' in out:
            config.output.bit_depth = int(out['bit_depth'])
        if 'format' in out:
            config.output.format = out['format']

    # Update gif
    if 'gif' in data:
        gif = data['gif']
        if 'fps' in gif:
            config.gif.fps = int(gif['fps'])
        if 'crop_padding' in gif:
            config.gif.crop_padding = int(gif['crop_padding'])
        if 'ping_pong' in gif:
            config.gif.ping_pong = bool(gif['ping_pong'])

    return config


def save_config(config: Config, config_path: str) -> None:
    """
    Save configuration to a YAML file.

    Args:
        config: Config object to save
        config_path: Output file path
    """
    data = config.to_dict()

    with open(config_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def find_config_file() -> Optional[str]:
    """
    Search for a config file in standard locations.

    Searches:
    1. ./config.yaml
    2. ./planetary-gif.yaml
    3. ~/.config/planetary-gif/config.yaml

    Returns:
        Path to config file, or None if not found
    """
    candidates = [
        'config.yaml',
        'planetary-gif.yaml',
        os.path.expanduser('~/.config/planetary-gif/config.yaml'),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    return None
