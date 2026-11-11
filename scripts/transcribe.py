"""Transcribe collected clips with Whisper and score whether each English term was actually spoken.

Reads <out_dir>/listen/listen.tsv (from collect_outputs.py) and writes <out_dir>/listen/asr.tsv with:
  - the Whisper transcript
  - english_hits: how many of the sentence's English terms (sentences.jsonl "english") appear in the
    transcript, matched case-insensitively on letters only (so "two-factor" == "two factor")

Whisper is a proxy, not ground truth: it can "correct" a mispronounced word from context, and with
language="ar" it sometimes writes an English word in Arabic script. Use english_hits to find clips to
listen to, and compare baseline vs routed on the same clips rather than trusting absolute numbers.

Usage:
  ~/projects/Speech/.venv/bin/python transcribe.py outputs/baseline_quick [--model openai/whisper-large-v3-turbo]
"""
import argparse
import csv
import json
import re
from pathlib import Path

import librosa
import soundfile as sf
import torch
from transformers import pipeline

HERE = Path(__file__).resolve().parent.parent  # repo root (sentences.jsonl, manifests/, outputs/, models/)

p = argparse.ArgumentParser()
p.add_argument("out_dir")
p.add_argument("--model", default="openai/whisper-large-v3-turbo")
p.add_argument("--language", default="ar", help='Whisper language hint ("ar"), or "auto" to let it detect')
args = p.parse_args()

listen_dir = Path(args.out_dir).resolve() / "listen"
rows = list(csv.DictReader(open(listen_dir / "listen.tsv", encoding="utf-8"), delimiter="\t"))
english = {r["id"]: r["english"] for r in map(json.loads, open(HERE / "sentences.jsonl", encoding="utf-8"))}


def letters(s: str) -> str:
    return re.sub(r"[^a-z]+", " ", s.lower()).strip()


asr = pipeline(
    "automatic-speech-recognition",
    model=args.model,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device=0 if torch.cuda.is_available() else -1,
)
generate_kwargs = {"task": "transcribe"}
if args.language != "auto":
    generate_kwargs["language"] = args.language

out_rows, total_hits, total_terms = [], 0, 0
for r in rows:
    wav = listen_dir / r["dataset"] / r["file"]
    audio, sr = sf.read(wav, dtype="float32")  # load ourselves: the pipeline's file loader needs ffmpeg
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    text = asr({"raw": audio, "sampling_rate": 16000}, generate_kwargs=generate_kwargs)["text"].strip()
    sid = Path(r["file"]).stem.rsplit("_", 1)[0]
    terms = english[sid]
    hyp = f" {letters(text)} "
    hit = [t for t in terms if f" {letters(t)} " in hyp]
    missed = [t for t in terms if t not in hit]
    total_hits += len(hit)
    total_terms += len(terms)
    out_rows.append([r["dataset"], r["file"], f"{len(hit)}/{len(terms)}", ", ".join(missed), r["text"], text])
    print(f"{r['file']:14s} {len(hit)}/{len(terms)}  missed={missed}\n    ref: {r['text']}\n    asr: {text}")

with open(listen_dir / "asr.tsv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["dataset", "file", "english_hits", "missed_terms", "reference", "whisper_transcript"])
    w.writerows(out_rows)

print(f"\nEnglish terms recovered by Whisper: {total_hits}/{total_terms} "
      f"({100 * total_hits / max(total_terms, 1):.0f}%)  [model={args.model}, language={args.language}]")
print(f"-> {listen_dir / 'asr.tsv'}")
