"""
SER file reader for planetary astrophotography videos.

SER is a simple uncompressed video format used by planetary imaging software.
Format specification: https://free-astro.org/index.php/SER
"""

import struct
from pathlib import Path
from typing import Iterator, Optional, List
from datetime import datetime

import numpy as np
import cv2


# SER color modes
COLOR_MONO = 0
COLOR_BAYER_RGGB = 8
COLOR_BAYER_GRBG = 9
COLOR_BAYER_GBRG = 10
COLOR_BAYER_BGGR = 11
COLOR_RGB = 100
COLOR_BGR = 101

COLOR_NAMES = {
    COLOR_MONO: "mono",
    COLOR_BAYER_RGGB: "bayer_rggb",
    COLOR_BAYER_GRBG: "bayer_grbg",
    COLOR_BAYER_GBRG: "bayer_gbrg",
    COLOR_BAYER_BGGR: "bayer_bggr",
    COLOR_RGB: "rgb",
    COLOR_BGR: "bgr",
}

# OpenCV debayer codes
DEBAYER_CODES = {
    COLOR_BAYER_RGGB: cv2.COLOR_BAYER_RG2RGB,
    COLOR_BAYER_GRBG: cv2.COLOR_BAYER_GR2RGB,
    COLOR_BAYER_GBRG: cv2.COLOR_BAYER_GB2RGB,
    COLOR_BAYER_BGGR: cv2.COLOR_BAYER_BG2RGB,
}


class SERReader:
    """
    Reader for SER (Simple uncompressed video) files.

    Usage:
        with SERReader('capture.ser') as ser:
            print(f"Frames: {ser.frame_count}")
            for frame in ser.iter_frames():
                process(frame)
    """

    # Header format: 178 bytes total
    # Note: LittleEndian field is historically inverted in many implementations
    HEADER_SIZE = 178
    HEADER_FORMAT = '<14sIIIIIII40s40s40sQQ'

    def __init__(self, filepath: str, debayer: bool = True):
        """
        Open a SER file for reading.

        Args:
            filepath: Path to SER file
            debayer: If True, convert Bayer pattern to RGB
        """
        self.filepath = Path(filepath)
        self.debayer = debayer
        self._file = None
        self._header = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def open(self):
        """Open the SER file and read header."""
        self._file = open(self.filepath, 'rb')
        self._read_header()

    def close(self):
        """Close the SER file."""
        if self._file:
            self._file.close()
            self._file = None

    def _read_header(self):
        """Parse the 178-byte SER header."""
        header_data = self._file.read(self.HEADER_SIZE)
        if len(header_data) < self.HEADER_SIZE:
            raise ValueError(f"Invalid SER file: header too short ({len(header_data)} bytes)")

        unpacked = struct.unpack(self.HEADER_FORMAT, header_data)

        file_id = unpacked[0].rstrip(b'\x00').decode('ascii', errors='ignore')
        if file_id != "LUCAM-RECORDER":
            raise ValueError(f"Invalid SER file: wrong file ID '{file_id}'")

        self._header = {
            'file_id': file_id,
            'lu_id': unpacked[1],
            'color_id': unpacked[2],
            'little_endian': unpacked[3],  # Note: historically inverted
            'width': unpacked[4],
            'height': unpacked[5],
            'bit_depth': unpacked[6],
            'frame_count': unpacked[7],
            'observer': unpacked[8].rstrip(b'\x00').decode('utf-8', errors='ignore'),
            'instrument': unpacked[9].rstrip(b'\x00').decode('utf-8', errors='ignore'),
            'telescope': unpacked[10].rstrip(b'\x00').decode('utf-8', errors='ignore'),
            'datetime': unpacked[11],
            'datetime_utc': unpacked[12],
        }

        # Calculate frame size
        bytes_per_pixel = 2 if self._header['bit_depth'] > 8 else 1
        planes = 3 if self._header['color_id'] in (COLOR_RGB, COLOR_BGR) else 1
        self._frame_size = self._header['width'] * self._header['height'] * bytes_per_pixel * planes
        self._bytes_per_pixel = bytes_per_pixel
        self._planes = planes

    @property
    def frame_count(self) -> int:
        return self._header['frame_count']

    @property
    def width(self) -> int:
        return self._header['width']

    @property
    def height(self) -> int:
        return self._header['height']

    @property
    def bit_depth(self) -> int:
        return self._header['bit_depth']

    @property
    def color_mode(self) -> str:
        return COLOR_NAMES.get(self._header['color_id'], 'unknown')

    @property
    def is_color(self) -> bool:
        return self._header['color_id'] != COLOR_MONO

    @property
    def metadata(self) -> dict:
        """Return full header metadata."""
        return self._header.copy()

    def read_frame(self, index: int) -> np.ndarray:
        """
        Read a single frame by index.

        Args:
            index: Frame index (0-based)

        Returns:
            Frame as numpy array (H, W) for mono or (H, W, 3) for color
        """
        if index < 0 or index >= self.frame_count:
            raise IndexError(f"Frame index {index} out of range (0-{self.frame_count-1})")

        # Seek to frame position
        offset = self.HEADER_SIZE + index * self._frame_size
        self._file.seek(offset)

        # Read raw data
        raw = self._file.read(self._frame_size)
        if len(raw) < self._frame_size:
            raise IOError(f"Incomplete frame {index}: expected {self._frame_size} bytes, got {len(raw)}")

        # Convert to numpy array
        dtype = np.uint16 if self._bytes_per_pixel == 2 else np.uint8
        frame = np.frombuffer(raw, dtype=dtype)

        # Reshape based on color mode
        color_id = self._header['color_id']

        if color_id in (COLOR_RGB, COLOR_BGR):
            # Interleaved RGB/BGR
            frame = frame.reshape((self._header['height'], self._header['width'], 3))
            if color_id == COLOR_RGB:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # OpenCV uses BGR
        elif color_id in DEBAYER_CODES and self.debayer:
            # Bayer pattern - debayer to color
            frame = frame.reshape((self._header['height'], self._header['width']))
            frame = cv2.cvtColor(frame, DEBAYER_CODES[color_id])
        else:
            # Mono or raw Bayer (no debayer)
            frame = frame.reshape((self._header['height'], self._header['width']))

        return frame

    def iter_frames(self, start: int = 0, end: Optional[int] = None) -> Iterator[np.ndarray]:
        """
        Iterate over frames.

        Args:
            start: Starting frame index
            end: Ending frame index (exclusive), None for all frames

        Yields:
            Frame arrays
        """
        if end is None:
            end = self.frame_count

        for i in range(start, min(end, self.frame_count)):
            yield self.read_frame(i)

    def read_timestamps(self) -> Optional[List[datetime]]:
        """
        Read frame timestamps from trailer (if present).

        Returns:
            List of datetime objects, or None if no timestamps
        """
        # Timestamps are stored after all frames, 8 bytes per frame
        trailer_offset = self.HEADER_SIZE + self.frame_count * self._frame_size
        self._file.seek(0, 2)  # Seek to end
        file_size = self._file.tell()

        expected_trailer_size = self.frame_count * 8
        if file_size < trailer_offset + expected_trailer_size:
            return None  # No trailer

        self._file.seek(trailer_offset)
        timestamps = []

        for _ in range(self.frame_count):
            ts_data = self._file.read(8)
            if len(ts_data) < 8:
                break
            # Timestamp is 100-nanosecond intervals since Jan 1, year 1
            ticks = struct.unpack('<Q', ts_data)[0]
            if ticks > 0:
                # Convert to Python datetime (approximate)
                # Windows FILETIME epoch is Jan 1, 1601
                # SER uses Jan 1, year 1, so add offset
                try:
                    seconds = ticks / 10_000_000
                    dt = datetime(1, 1, 1) + timedelta(seconds=seconds)
                    timestamps.append(dt)
                except (ValueError, OverflowError):
                    timestamps.append(None)
            else:
                timestamps.append(None)

        return timestamps if any(t is not None for t in timestamps) else None


def get_ser_info(filepath: str) -> dict:
    """
    Get metadata from a SER file without reading frames.

    Args:
        filepath: Path to SER file

    Returns:
        Dictionary with file metadata
    """
    with SERReader(filepath) as ser:
        info = ser.metadata
        info['color_mode'] = ser.color_mode
        info['is_color'] = ser.is_color
        return info
