#!/usr/bin/env bash
# Run NVIDIA's official Magpie inference script on our evalsets, with the flags from
# tests/functional_tests/L2_TTS_InferEvaluate_Magpietts_SeenSpeakers.sh
# (batch size lowered for a 4 GB GPU; evaluation disabled for now).
#
# Usage: scripts/run_inference.sh <datasets> <out_name> [batch_size] [extra magpietts_inference.py flags...]
#   scripts/run_inference.sh baseline_white,baseline_saudi baseline
set -euo pipefail

DATASETS="${1:?datasets, e.g. baseline_white,baseline_saudi}"
OUT_NAME="${2:?output folder name, e.g. baseline_ar}"
BATCH_SIZE="${3:-2}"
EXTRA_ARGS=("${@:4}")  # e.g. --temperature 0.4 --num_repeats 3 (override the defaults below)

EVAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"  # repo root
SPEECH_DIR="${SPEECH_DIR:-$HOME/projects/Speech}"  # NeMo Speech checkout with a built .venv
OUT_DIR="$EVAL_DIR/outputs/$OUT_NAME"
mkdir -p "$OUT_DIR"

cd "$SPEECH_DIR"

# Call .venv/bin/python directly, never `uv run`: a default sync follows .python-version (3.12)
# with no extras and wipes the env, and `uv run --no-sync` chokes on nemo_automodel egg-info metadata.
PYVER="$(.venv/bin/python -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
if [[ "$PYVER" != "3.13" ]] || ! .venv/bin/python -c "import hydra" 2>/dev/null; then
    echo "ERROR: .venv is Python $PYVER or missing extras. Rebuild with:"
    echo "  cd $SPEECH_DIR && uv sync --locked --python 3.13 --managed-python --extra all --extra cu13 --group test"
    exit 1
fi

echo "Speech repo: $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)" | tee "$OUT_DIR/run.log"

CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python examples/tts/magpietts_inference.py \
    --model_type magpie \
    --nemo_files "$EVAL_DIR/models/magpie_v2607/magpie_tts_multilingual_357m.nemo" \
    --codecmodel_path "$EVAL_DIR/models/codec/nemo-nano-codec-22khz-1.89kbps-21.5fps.nemo" \
    --datasets_json_path "$EVAL_DIR/evalset_config.json" \
    --datasets "$DATASETS" \
    --out_dir "$OUT_DIR" \
    --deterministic \
    --batch_size "$BATCH_SIZE" \
    --use_cfg --cfg_scale 2.5 \
    --temperature 0.6 \
    --apply_attention_prior \
    --apply_prior_to_layers "3,4,5,6,7,8,9" \
    --estimate_alignment_from_layers "3,4,5,6" \
    --use_local_transformer \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee -a "$OUT_DIR/run.log"

echo
echo "Tokenizer lines from the log (should be arabic_*_chartokenizer, not english_phoneme):"
grep -i "tokenizer" "$OUT_DIR/run.log" | grep -iv "warn" | tail -5 || true
echo
echo "Generated files:"
find "$OUT_DIR" -name "predicted_audio_*.wav" | wc -l
