"""Repo-level rules from AGENTS.md that can be enforced mechanically.

R3  config/splits.yaml is frozen (sha256 pinned here; change it only in a deliberate commit that
    also explains why in docs/decisions.md).
R5  only src/llm.py may talk to the model provider.
R1  the commit-msg hook strips AI attribution lines.
R9  .env is never tracked by git.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from datetime import date
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "config" / "splits.yaml"
SPLITS_SHA256 = "2113e2dda4b450e7d42738ba73edce169996f123ea989962e2983853fbe52e02"


def _normalized_sha256(path: Path) -> str:
    text = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(text).hexdigest()


def test_splits_file_is_frozen() -> None:
    assert _normalized_sha256(SPLITS) == SPLITS_SHA256, (
        "config/splits.yaml changed. AGENTS.md R3: splits are frozen. If this is deliberate, "
        "record a D-nn in docs/decisions.md and update SPLITS_SHA256 in the same commit."
    )


def test_splits_are_ordered_and_disjoint() -> None:
    cfg = yaml.safe_load(SPLITS.read_text(encoding="utf-8"))
    periods = [cfg[k] for k in ("warmup", "train", "validation", "test")]
    for p in periods:
        assert p["start"] <= p["end"]
    for earlier, later in zip(periods, periods[1:], strict=False):
        assert earlier["end"] < later["start"], "periods must not overlap"
    assert cfg["test"]["end"] < date(2026, 11, 13), "test period must be fully in the past"


def test_only_llm_module_talks_to_the_provider() -> None:
    pattern = re.compile(r"openrouter|OPENROUTER_API_KEY|api\.openai|anthropic", re.IGNORECASE)
    offenders = []
    for py in (ROOT / "src").rglob("*.py"):
        if py.name == "llm.py" and py.parent == ROOT / "src":
            continue
        if pattern.search(py.read_text(encoding="utf-8")):
            offenders.append(str(py.relative_to(ROOT)))
    assert not offenders, f"AGENTS.md R5: provider access outside src/llm.py in {offenders}"


def test_env_is_not_tracked() -> None:
    if shutil.which("git") is None:
        pytest.skip("git not available")
    out = subprocess.run(
        ["git", "ls-files", ".env"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    assert out.stdout.strip() == "", "AGENTS.md R9: .env must never be committed"


def test_commit_msg_hook_strips_attribution(tmp_path: Path) -> None:
    sh = shutil.which("sh")
    if sh is None:
        pytest.skip("POSIX sh not available")
    msg = tmp_path / "COMMIT_EDITMSG"
    msg.write_text(
        "feat(nlp): add sentiment service\n"
        "\n"
        "Body line stays.\n"
        "\n"
        "🤖 Generated with [Claude Code](https://claude.ai/code)\n"
        "Co-Authored-By: Claude <noreply@anthropic.com>\n"
        "Claude-Session: abc123\n",
        encoding="utf-8",
    )
    hook = ROOT / ".githooks" / "commit-msg"
    subprocess.run([sh, str(hook), str(msg)], cwd=ROOT, check=True)
    cleaned = msg.read_text(encoding="utf-8")
    assert "feat(nlp): add sentiment service" in cleaned
    assert "Body line stays." in cleaned
    assert "Co-Authored-By" not in cleaned
    assert "Generated with" not in cleaned
    assert "Claude-Session" not in cleaned
