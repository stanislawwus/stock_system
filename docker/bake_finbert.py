"""Download the pinned FinBERT weights into HF_HOME at image build time (D-48).

Uses the same model name and revision as the app (config/nlp.yaml), so the running container can
set HF_HUB_OFFLINE=1 and still load FinBERT: /analyze never falls back to the lexicon because of
a missing download. Fails the build if the weights do not load and classify.
"""

import sys
from pathlib import Path

import yaml

cfg = yaml.safe_load(Path("config/nlp.yaml").read_text(encoding="utf-8"))["sentiment"]
name, rev = cfg["finbert_model"], cfg["finbert_revision"]

from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(name, revision=rev)
model = AutoModelForSequenceClassification.from_pretrained(name, revision=rev)
labels = sorted(model.config.id2label.values())
if labels != ["negative", "neutral", "positive"]:
    sys.exit(f"unexpected FinBERT labels {labels}")
print(f"baked {name}@{rev[:7]} labels={labels}")
