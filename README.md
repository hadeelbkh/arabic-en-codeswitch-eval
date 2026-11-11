# Arabic-English code-switching eval for Magpie TTS

A small test set and scripts to measure how well [NVIDIA Magpie TTS](https://huggingface.co/nvidia/magpie_tts_multilingual_357m) says English words inside Arabic sentences, which is how most Arabic speakers actually talk.

This is the evaluation behind:
- Issue [NVIDIA-NeMo/Speech#16592](https://github.com/NVIDIA-NeMo/Speech/issues/16592): English words in Arabic sentences are dropped or hallucinated
- PR [NVIDIA-NeMo/Speech#16599](https://github.com/NVIDIA-NeMo/Speech/pull/16599): Arabic-English code-switching tokenization
- PR [NVIDIA-NeMo/Speech#16591](https://github.com/NVIDIA-NeMo/Speech/pull/16591): Arabic and other languages falling back to the English tokenizer

Audio samples (before / after): [hadeelbkh/arabic-en-codeswitch-tts-samples](https://huggingface.co/datasets/hadeelbkh/arabic-en-codeswitch-tts-samples)

## Results

`magpie_tts_multilingual_357m` v2607, all 5 built-in speakers, 30 sentences, 150 clips per run. The only difference between the two runs is the tokenizer.

| | Before (Arabic tokenizer only) | After (#16599) |
|---|---|---|
| English terms found | 0% (0/115) | 91% (105/115) |
| English phrases | 0% | 100% |
| Saudi sentences | 0% | 96% |
| English at the start of the sentence | 0% | 73% |

Per-clip transcripts are in [`results/`](results/).

## The test set

[`sentences.jsonl`](sentences.jsonl): 30 banking / customer-service sentences, 24 in white (simple) Arabic and 6 in Saudi dialect. They cover:

- single English words, phrases, acronyms (PIN, OTP, ATM...), brand names (Apple Pay, Visa...)
- English at the start, middle and end of the sentence
- numbers, written as Arabic words, because the Arabic char tokenizer has no digits

Each line has the sentence, its dialect, category, position of the English, and the English terms to check.

## How it's scored

Every clip is transcribed with `whisper-large-v3-turbo`, then each English term is checked:

- **strict**: the term appears in Latin letters
- **skeleton**: the term's consonants appear in the transcript in Latin *or* Arabic letters ("credit card" → `krdt krd` ← "كريدت كارد"), because Whisper often writes correct English words in Arabic script

Terms with fewer than 4 consonants (OTP, PIN, API, Visa...) are left out of the skeleton score, since they match normal Arabic words by chance. With that rule the baseline scores 0%, so there are no false positives. Use the scores to compare runs, not as absolute quality numbers, and listen to the clips.

## Reproduce

Needs an NVIDIA GPU (all of this ran on a 4 GB laptop GPU) and a checkout of [NeMo Speech](https://github.com/NVIDIA-NeMo/Speech). Use `main` for the baseline, and the branch of #16599 (or `main` once it's merged) for the code-switched run.

```bash
# 1. NeMo Speech environment (in the Speech checkout)
uv sync --locked --python 3.13 --managed-python --extra all --extra cu13 --group test
PY=~/projects/Speech/.venv/bin/python          # call this python directly, not `uv run`

# 2. Models (into this repo's models/)
~/projects/Speech/.venv/bin/hf download nvidia/magpie_tts_multilingual_357m magpie_tts_multilingual_357m.nemo \
    --revision v2607 --local-dir <this repo>/models/magpie_v2607
~/projects/Speech/.venv/bin/hf download nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps nemo-nano-codec-22khz-1.89kbps-21.5fps.nemo \
    --local-dir <this repo>/models/codec

# 3. Manifests + evalset config (baseline: one Arabic tokenizer per dialect)
cd <this repo>
python3 scripts/build_manifest.py --prefix baseline
#    code-switched: tokenizer pairs (needs #16599)
python3 scripts/build_manifest.py --prefix routed \
    --white-tokenizer arabic_MSA_chartokenizer+english_phoneme \
    --saudi-tokenizer arabic_SA_chartokenizer+english_phoneme

# 4. Generate with NVIDIA's examples/tts/magpietts_inference.py (CI flags), then transcribe and score
SPEECH_DIR=~/projects/Speech scripts/run_inference.sh baseline_white,baseline_saudi baseline
$PY scripts/collect_outputs.py outputs/baseline
$PY scripts/transcribe.py outputs/baseline
python3 scripts/score.py outputs/baseline outputs/routed
```

Notes:
- The evalset sets `tokenizer_names` explicitly. Without #16591, `language: "ar"` alone silently uses `english_phoneme`.
- `--speakers 0,4` and `--ids W03,S01,...` in `build_manifest.py` make smaller runs.

## Other scripts

| Script | What it does |
|---|---|
| `print_tokenizers.py` | Lists the tokenizers inside a `.nemo` checkpoint |
| `check_routing.py` | Shows which tokenizer each span goes to with the real checkpoint (no audio) |
| `make_videos.sh` | Turns clips into small `.mp4` files GitHub can play inline |
| `make_compare_video.sh` | One before/after video for the same sentence |

## License

The scripts and test sentences are released under the [Apache License 2.0](LICENSE).
The generated audio samples are in the Hugging Face dataset, under CC-BY-4.0.
