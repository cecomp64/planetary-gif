#!/bin/bash
set -e

SCRIPTDIR="$(cd "$(dirname "$0")" && pwd)"

python3 "$SCRIPTDIR/align_planets.py" "$SCRIPTDIR/images/21_*.png" "$SCRIPTDIR/images/jupiter_pingpong.gif"

