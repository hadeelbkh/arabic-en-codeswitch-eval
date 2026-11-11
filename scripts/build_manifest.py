"""Build manifests + evalset config consumed by NeMo's examples/tts/magpietts_inference.py.

One dataset per dialect, each forced to its own Arabic tokenizer via `tokenizer_names`
(the CLI inference dataset ignores the checkpoint's language_to_tokenizer_mapping and would
otherwise silently fall back to english_phoneme for "ar"):
    <prefix>_white -> arabic_MSA_chartokenizer
    <prefix>_saudi -> arabic_SA_chartokenizer

Each sentence is rendered once per built-in speaker (Aria 0, Jason 1, John 2, Leo 3, Sofia 4).
The official script names outputs predicted_audio_<row>.wav, so index TSVs map rows back to ids.

Text cleanup applied (Arabic char tokenizers silently drop these chars):
    tatweel "ـ" is removed. Numbers were already spelled out in sentences.jsonl (no digits in charset).

Usage:
  python3 build_manifest.py [--prefix baseline] [--speakers 0,1,2,3,4]
      [--white-tokenizer arabic_MSA_chartokenizer] [--saudi-tokenizer arabic_SA_chartokenizer]
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent  # repo root (sentences.jsonl, manifests/, outputs/, models/)
SPEAKERS = {0: "Aria", 1: "Jason", 2: "John", 3: "Leo", 4: "Sofia"}

p = argparse.ArgumentParser()
p.add_argument("--prefix", default="baseline", help="Dataset name prefix (datasets: <prefix>_white, <prefix>_saudi)")
p.add_argument("--speakers", default="0,1,2,3,4")
p.add_argument("--white-tokenizer", default="arabic_MSA_chartokenizer")
p.add_argument("--saudi-tokenizer", default="arabic_SA_chartokenizer")
p.add_argument("--ids", default=None, help="Comma-separated sentence ids to keep (default: all), e.g. W03,W06,S01")
args = p.parse_args()

speakers = [int(s) for s in args.speakers.split(",")]
rows = [json.loads(line) for line in open(HERE / "sentences.jsonl", encoding="utf-8")]
if args.ids:
    keep = set(args.ids.split(","))
    unknown = keep - {r["id"] for r in rows}
    if unknown:
        raise SystemExit(f"Unknown sentence ids: {sorted(unknown)}")
    rows = [r for r in rows if r["id"] in keep]
tokenizer_for = {"white": args.white_tokenizer, "saudi": args.saudi_tokenizer}

out = HERE / "manifests"
out.mkdir(exist_ok=True)
evalset_path = HERE / "evalset_config.json"
evalset = json.loads(evalset_path.read_text()) if evalset_path.exists() else {}

for dialect, tokenizer in tokenizer_for.items():
    name = f"{args.prefix}_{dialect}"
    subset = [r for r in rows if r["dialect"] == dialect]
    manifest_path = out / f"{name}.json"
    index_path = out / f"{name}_index.tsv"
    with open(manifest_path, "w", encoding="utf-8") as m, open(index_path, "w", encoding="utf-8") as ix:
        ix.write("row\tid\tspeaker_index\tspeaker\tcategory\tposition\ttext\n")
        row = 0
        for spk in speakers:
            for r in subset:
                text = r["text"].replace("ـ", "")
                entry = {
                    "text": text,
                    "duration": 5.0,  # required by the dataset's duration filter; no target audio exists
                    "speaker_index": spk,
                    "language": "ar",
                    "id": r["id"],
                }
                m.write(json.dumps(entry, ensure_ascii=False) + "\n")
                ix.write(f"{row}\t{r['id']}\t{spk}\t{SPEAKERS[spk]}\t{r['category']}\t{r['position']}\t{text}\n")
                row += 1
    evalset[name] = {
        "manifest_path": str(manifest_path),
        "audio_dir": "/",
        "language": "ar",
        "tokenizer_names": [tokenizer],
    }
    print(f"{name}: {row} rows, tokenizer={tokenizer} -> {manifest_path}")

evalset_path.write_text(json.dumps(evalset, indent=2, ensure_ascii=False) + "\n")
print(f"Evalset config -> {evalset_path}")
