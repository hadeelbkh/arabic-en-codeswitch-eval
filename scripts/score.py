"""Score English-term recovery in Whisper transcripts, strict and script-agnostic, for one or more runs.

strict:    the English term appears in Latin letters (what transcribe.py reports).
skeleton:  the term's consonant skeleton appears in the transcript, whether Whisper wrote it in Latin or
           Arabic letters ("credit card" -> krdt krd <- "كريدت كارد"). Vowels are dropped and sounds Arabic
           lacks are merged (p->b, v->f, c/q->k, g->j, x->ks). Terms with fewer than 4 skeleton consonants (mostly
           acronyms) are excluded, because they match ordinary Arabic words by chance.
           This is a heuristic proxy: apply it to every run and compare runs, don't read it as an absolute score.

Usage: python3 score.py outputs/baseline_full outputs/routed_full
"""
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent  # repo root (sentences.jsonl, manifests/, outputs/, models/)
SENTENCES = {r["id"]: r for r in map(json.loads, open(HERE / "sentences.jsonl", encoding="utf-8"))}

LATIN = {"p": "b", "v": "f", "c": "k", "q": "k", "g": "j", "x": "ks", "z": "z"}
ARABIC = {
    "ب": "b", "ت": "t", "ث": "t", "ج": "j", "ح": "", "خ": "k", "د": "d", "ذ": "d", "ر": "r", "ز": "z",
    "س": "s", "ش": "s", "ص": "s", "ض": "d", "ط": "t", "ظ": "z", "ع": "", "غ": "j", "ف": "f", "ق": "k",
    "ك": "k", "ل": "l", "م": "m", "ن": "n", "ه": "", "ة": "", "پ": "b", "ڤ": "f", "گ": "j",
}  # alef/waw/yaa/hamza are vowels here and drop out


def latin_skeleton(text: str) -> str:
    out = []
    for ch in text.lower():
        if ch in "aeiouyhw":
            continue
        if ch.isalpha():
            out.append(LATIN.get(ch, ch))
        elif ch.isspace() and out and out[-1] != " ":
            out.append(" ")
    s = "".join(out).replace("ck", "k").replace("sh", "sh")
    return re.sub(r"(.)\1+", r"\1", s).strip()


def arabic_or_latin_skeleton(text: str) -> str:
    out = []
    for ch in text:
        if "a" <= ch.lower() <= "z":
            out.append(latin_skeleton(ch))
        elif ch in ARABIC:
            out.append(ARABIC[ch])
        elif ch.isspace() and out and out[-1] != " ":
            out.append(" ")
    return re.sub(r"(.)\1+", r"\1", "".join(out)).strip()


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


MIN_SKELETON = 4


def skeleton_hit(term: str, transcript: str):
    term_sk = latin_skeleton(term).replace(" ", "")
    hyp_sk = arabic_or_latin_skeleton(transcript).replace(" ", "")
    if len(term_sk) < MIN_SKELETON:
        return None  # too short to tell apart from ordinary Arabic words; check these by ear
    budget = 0 if len(term_sk) <= 5 else len(term_sk) // 5
    for length in range(max(1, len(term_sk) - budget), len(term_sk) + budget + 1):
        for start in range(0, max(1, len(hyp_sk) - length + 1)):
            if edit_distance(term_sk, hyp_sk[start : start + length]) <= budget:
                return True
    return False


def strict_hit(term: str, transcript: str) -> bool:
    letters = lambda s: f" {re.sub(r'[^a-z]+', ' ', s.lower()).strip()} "
    return letters(term) in letters(transcript)


def term_position(text: str, term: str) -> str:
    """Where the term sits in the sentence: start, middle or end (ignoring punctuation)."""
    words = re.sub(r"[^\w\s'-]", " ", text).split()
    term_words = term.split()
    for i in range(len(words) - len(term_words) + 1):
        if [w.lower() for w in words[i : i + len(term_words)]] == [w.lower() for w in term_words]:
            if i == 0:
                return "start"
            return "end" if i + len(term_words) == len(words) else "middle"
    return "?"


def score(run_dir: Path):
    rows = list(csv.DictReader(open(run_dir / "listen" / "asr.tsv", encoding="utf-8"), delimiter="\t"))
    groups = defaultdict(lambda: [0, 0, 0])  # strict, skeleton, total
    for r in rows:
        sid, speaker = Path(r["file"]).stem.rsplit("_", 1)
        meta = SENTENCES[sid]
        for term in meta["english"]:
            s, k = strict_hit(term, r["whisper_transcript"]), skeleton_hit(term, r["whisper_transcript"])
            if k is None:
                short_terms.add(term)
                continue
            position = term_position(meta["text"], term)
            for key in ("ALL", f"dialect:{meta['dialect']}", f"category:{meta['category']}", f"speaker:{speaker}",
                        f"position:{position}"):
                groups[key][0] += s
                groups[key][1] += k
                groups[key][2] += 1
    return groups


short_terms = set()
runs = [Path(p).resolve() for p in sys.argv[1:]]
results = {run.name: score(run) for run in runs}
keys = sorted({k for g in results.values() for k in g}, key=lambda k: (k != "ALL", k))
header = f"{'':22s}" + "".join(f"{name:>28s}" for name in results)
print(header + "\n" + f"{'':22s}" + "".join(f"{'strict    skeleton':>28s}" for _ in results))
for key in keys:
    line = f"{key:22s}"
    for g in results.values():
        s, k, n = g.get(key, [0, 0, 0])
        line += f"{f'{s}/{n} {100*s/max(n,1):3.0f}%   {k}/{n} {100*k/max(n,1):3.0f}%':>28s}"
    print(line)
print(f"\nExcluded (skeleton < {MIN_SKELETON} consonants, check by ear): {', '.join(sorted(short_terms))}")
