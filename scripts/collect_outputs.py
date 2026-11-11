"""Copy magpietts_inference.py outputs into readable names and flag suspicious clips.

The official script writes <out_dir>/<long_experiment_name>_<dataset>/audio/repeat_0/predicted_audio_<row>.wav.
This maps each row back to its sentence id + speaker via manifests/<dataset>_index.tsv and copies it to
<out_dir>/listen/<dataset_suffix>/<id>_<speaker>.wav, plus a listen.tsv sheet with duration/level stats.

Usage: ~/projects/Speech/.venv/bin/python collect_outputs.py outputs/baseline_quick
"""
import csv
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent.parent  # repo root (sentences.jsonl, manifests/, outputs/, models/)
out_dir = Path(sys.argv[1]).resolve()
listen_dir = out_dir / "listen"
listen_dir.mkdir(exist_ok=True)

sheet = []
for run_dir in sorted(p for p in out_dir.iterdir() if p.is_dir() and p.name != "listen"):
    # The run folder name ends with "_<dataset>"; match it against the known index files.
    dataset = next(
        (f.name[: -len("_index.tsv")] for f in (HERE / "manifests").glob("*_index.tsv")
         if run_dir.name.endswith("_" + f.name[: -len("_index.tsv")])),
        None,
    )
    if dataset is None:
        print(f"skip {run_dir.name}: no matching manifests/*_index.tsv")
        continue
    with open(HERE / "manifests" / f"{dataset}_index.tsv", encoding="utf-8") as f:
        index = {int(r["row"]): r for r in csv.DictReader(f, delimiter="\t")}

    dest = listen_dir / dataset
    dest.mkdir(exist_ok=True)
    for wav in sorted(run_dir.glob("audio/repeat_*/predicted_audio_*.wav"),
                      key=lambda p: (p.parent.name, int(p.stem.rsplit("_", 1)[1]))):
        row = int(wav.stem.rsplit("_", 1)[1])
        meta = index[row]
        audio, sr = sf.read(wav)
        dur = len(audio) / sr
        rms_db = 20 * np.log10(np.sqrt(np.mean(audio**2)) + 1e-9)
        words = len(meta["text"].split())
        sec_per_word = dur / max(words, 1)
        flag = ""
        if rms_db < -45:
            flag = "near-silent"
        elif sec_per_word < 0.2:
            flag = "too short (words skipped?)"
        elif sec_per_word > 1.0:
            flag = "too long (babbling/repeats?)"
        name = f"{meta['id']}_{meta['speaker']}.wav"
        repeat = wav.parent.name  # repeat_0, repeat_1, ...
        if repeat != "repeat_0":
            name = f"{repeat}/{name}"
            (dest / repeat).mkdir(exist_ok=True)
        shutil.copy2(wav, dest / name)
        sheet.append([dataset, name, f"{dur:.2f}", f"{sec_per_word:.2f}", f"{rms_db:.1f}", flag,
                      meta["category"], meta["position"], meta["text"]])

with open(listen_dir / "listen.tsv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["dataset", "file", "sec", "sec_per_word", "rms_db", "flag", "category", "position", "text",
                "english_ok (fill in)", "arabic_ok (fill in)", "notes"])
    w.writerows(sheet)

for r in sheet:
    print(f"{r[0]:12s} {r[1]:14s} {r[2]:>6s}s {r[3]:>5s}s/word {r[4]:>6s}dB  {r[5]}")
print(f"\n{len(sheet)} clips -> {listen_dir}  (sheet: {listen_dir / 'listen.tsv'})")
