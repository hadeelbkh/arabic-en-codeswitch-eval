"""Print the text tokenizers and language->tokenizer mapping stored inside a Magpie .nemo checkpoint.

Usage: ~/projects/Speech/.venv/bin/python print_tokenizers.py models/magpie_v2607/magpie_tts_multilingual_357m.nemo
"""
import sys
import tarfile

import yaml

nemo_path = sys.argv[1]
with tarfile.open(nemo_path, "r:*") as tar:
    member = next(m for m in tar if m.name.endswith("model_config.yaml"))
    cfg = yaml.safe_load(tar.extractfile(member))

tokenizers = cfg.get("text_tokenizers", {}) or {}
print("Text tokenizers in checkpoint:")
for name, tcfg in tokenizers.items():
    target = (tcfg or {}).get("_target_", "?")
    print(f"  {name:30s} {target}")

print("\nlanguage_to_tokenizer_mapping:")
print(f"  {cfg.get('language_to_tokenizer_mapping')}")
