"""Sanity-check code-switched tokenization on the real Magpie checkpoint tokenizers (no audio generated).

Usage: ~/projects/Speech/.venv/bin/python check_routing.py
"""
import itertools
from pathlib import Path

import torch

from nemo.collections.tts.modules.magpietts_inference.utils import ModelLoadConfig, load_magpie_model

HERE = Path(__file__).resolve().parent.parent  # repo root (sentences.jsonl, manifests/, outputs/, models/)
SENTENCES = [
    "أبغى أسوي transfer لحساب أخوي، كيف أسويه؟",
    "Premium account يمنحك cashback أعلى ودخولًا مجانيًا إلى airport lounges.",
    "وش الفرق بين ال credit card وال debit card؟",
    "ننصحك بتفعيل two-factor authentication لحماية حسابك من الاختراق.",
    "تم تحويل 2500 ريال عبر Instant Transfer.",
]

model, _ = load_magpie_model(
    ModelLoadConfig(
        nemo_file=str(HERE / "models/magpie_v2607/magpie_tts_multilingual_357m.nemo"),
        codecmodel_path=str(HERE / "models/codec/nemo-nano-codec-22khz-1.89kbps-21.5fps.nemo"),
    ),
    device="cpu",
)
tok = model.tokenizer
pair = ["arabic_MSA_chartokenizer", "english_phoneme"]
name = "+".join(pair)

for text in SENTENCES:
    ids = tok.encode(text, name)
    plain = tok.encode(text, "arabic_MSA_chartokenizer")
    runs = [
        (owner, tok.decode(list(group), owner))
        for owner, group in itertools.groupby(ids, key=lambda t: tok._owning_tokenizer(t, pair))
    ]
    print(f"\n{text}\n  plain Arabic: {len(plain)} tokens | code-switched: {len(ids)} tokens")
    for owner, piece in runs:
        print(f"    {'EN' if owner == 'english_phoneme' else 'AR'} [{piece}]")
    assert ids == tok.encode(text, name), "encoding is not deterministic"

print("\nG2P phoneme_probability after encoding:", tok.tokenizers["english_phoneme"].g2p.phoneme_probability)
