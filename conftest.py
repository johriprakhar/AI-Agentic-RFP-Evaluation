"""
Shared test configuration.

Two jobs:

* put the project root on ``sys.path`` and make it the working directory, so the
  suites behave identically whether they are launched from the repo root, from
  an IDE, or by pytest's rootdir discovery;
* guarantee the synthetic sample PDFs exist before any test needs them, by
  generating them on demand rather than depending on committed binaries.

Both exist so that a fresh clone passes on the first attempt with no manual
setup step.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = PROJECT_ROOT / "sample_pdfs"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def ensure_sample_pdfs() -> list:
    """Return the sample proposal paths, generating any that are missing."""
    from generate_sample_pdfs import PROPOSALS, generate

    expected = [SAMPLE_DIR / f"{stem}.pdf" for stem in PROPOSALS]
    if not all(path.exists() for path in expected):
        generate(SAMPLE_DIR, force=False)
    return sorted(SAMPLE_DIR.glob("*.pdf"))


# Tests reference `sample_pdfs/...` relatively in a few places; anchoring the
# working directory keeps those paths valid regardless of the launch directory.
os.chdir(PROJECT_ROOT)
ensure_sample_pdfs()
