#!/usr/bin/env bash
# Build one square before/after video (for social posts): the same sentence said twice, each half labeled.
# Usage: ./make_compare_video.sh <before.wav> <after.wav> <out.mp4>
set -euo pipefail
BEFORE="${1:?before wav}"; AFTER="${2:?after wav}"; OUT="${3:?out mp4}"
FONT=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf
FONT_BOLD=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf
[[ -f "$FONT_BOLD" ]] || FONT_BOLD="$FONT"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

segment() {  # <wav> <title> <subtitle> <wave color> <out>
    ffmpeg -y -loglevel error -i "$1" -f lavfi -t 1 -i anullsrc=r=22050:cl=mono \
        -filter_complex "[0:a][1:a]concat=n=2:v=0:a=1,aresample=44100,asplit[a][w];
            [w]showwaves=s=1080x360:mode=cline:draw=full:colors=$4,format=rgba[wv];
            color=c=black:s=1080x1080:r=25[bg];
            [bg][wv]overlay=0:400:shortest=1:format=auto,
            drawtext=fontfile=$FONT_BOLD:text='$2':fontcolor=white:fontsize=72:x=(w-tw)/2:y=150,
            drawtext=fontfile=$FONT:text='$3':fontcolor=0xb8c0cc:fontsize=34:x=(w-tw)/2:y=250,
            drawtext=fontfile=$FONT:text='Magpie TTS multilingual 357M  |  speaker\\: Aria':fontcolor=0x6b7480:fontsize=28:x=(w-tw)/2:y=900,format=yuv420p[v]" \
        -map "[v]" -map "[a]" -c:v libx264 -preset veryfast -crf 23 -c:a aac -b:a 128k -ar 44100 -ac 2 "$5"
}

segment "$BEFORE" "BEFORE" "one Arabic tokenizer for the whole sentence" "tomato" "$TMP/before.mp4"
segment "$AFTER" "AFTER" "Arabic words to Arabic, English words to English" "dodgerblue" "$TMP/after.mp4"
printf "file '%s'\nfile '%s'\n" "$TMP/before.mp4" "$TMP/after.mp4" > "$TMP/list.txt"
ffmpeg -y -loglevel error -f concat -safe 0 -i "$TMP/list.txt" -c:v libx264 -preset veryfast -crf 23 -pix_fmt yuv420p -c:a aac -b:a 160k -ar 44100 -ac 2 -movflags +faststart "$OUT"
echo "$OUT ($(du -h "$OUT" | cut -f1), $(ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUT" | cut -c1-4)s)"
