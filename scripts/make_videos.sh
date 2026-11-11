#!/usr/bin/env bash
# Turn listen/*.wav clips into small .mp4 files (audio + moving waveform) that GitHub plays inline.
# Needs ffmpeg:  sudo apt install -y ffmpeg
#
# Usage: ./make_videos.sh <listen_dir> <out_dir> <clip> [<clip> ...]
#   ./make_videos.sh outputs/baseline_full/listen issue_videos baseline_white/W14_Aria baseline_saudi/S05_Aria
set -euo pipefail

LISTEN_DIR="${1:?listen dir}"; OUT_DIR="${2:?output dir}"; shift 2
mkdir -p "$OUT_DIR"

for clip in "$@"; do
    in="$LISTEN_DIR/$clip.wav"
    out="$OUT_DIR/$(basename "$clip").mp4"
    ffmpeg -y -loglevel error -i "$in" \
        -filter_complex "[0:a]showwaves=s=640x200:mode=cline:colors=0x4f9dde,format=yuv420p[v]" \
        -map "[v]" -map 0:a -c:v libx264 -preset veryfast -crf 28 -c:a aac -b:a 128k -ar 44100 -ac 2 -movflags +faststart \
        "$out"
    echo "$out ($(du -h "$out" | cut -f1))"
done
