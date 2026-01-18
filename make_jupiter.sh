#!/bin/bash
set -e

# --- CONFIG ---
IMGDIR="/Users/csvensson/Git/planetary-gif/images"
SIRIL_ALIGN_SCRIPT="/Users/csvensson/Git/planetary-gif/align_and_export.ssf"

echo ">>> Step 1: Creating Siril-compatible sequence (symlinks)"
cd "$IMGDIR"

# Remove old symlinks and FITS
rm -f jup_*.png jup_*.fit

n=1
for f in 21_*.png; do
    printf -v new "jup_%05d.png" "$n"
    ln -s "$f" "$new"
    n=$((n+1))
done

echo ">>> Step 2: Converting PNG → FITS for Siril CLI"
for f in jup_*.png; do
    base="${f%.png}"
    TMP=$(mktemp /tmp/siril_convert_XXXXXX.ssf)
    cat > "$TMP" <<EOF
requires 1.4.0
cd $IMGDIR
load $f
save $base fit
EOF
    siril-cli -s "$TMP"
    rm "$TMP"
done

echo ">>> Step 3: Running Siril alignment + auto-crop"
siril-cli -s "$SIRIL_ALIGN_SCRIPT"

echo ">>> Step 4: Building ping-pong GIF with ffmpeg"
ffmpeg -pattern_type glob -i "$IMGDIR/aligned_pngs_*.png" \
-filter_complex "
  [0:v]split[fwd][rev];
  [rev]reverse[rev2];
  [fwd][rev2]concat=n=2:v=1:a=0, fps=15
" \
-loop 0 "$IMGDIR/jupiter_pingpong.gif"

echo ">>> DONE. GIF created:"
echo "$IMGDIR/jupiter_pingpong.gif"

