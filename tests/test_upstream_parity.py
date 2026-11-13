"""Compare package inference with saved direct author-model output.

Reads: original HF CPU score fixture, the synthetic WAV, and the pinned
checkpoint in the Hugging Face cache. Marked model because it loads weights.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from wstag_infer import WSTAGSession, default_checkpoint

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.model
def test_direct_author_cpu_scores_match_pinned_inference_code():
    baseline = json.loads((FIXTURES / "upstream_cpu.json").read_text())
    checkpoint = default_checkpoint()
    assert baseline["revision"] == checkpoint.revision
    with WSTAGSession(device="cpu", local_files_only=True) as session:
        result = session.infer(FIXTURES / "synthetic_events.wav", tuple(baseline["prompts"]))
        assert session.status == "ready"
    assert session.status == "closed"
    assert result.duration == 10.0
    assert result.checkpoint_revision == checkpoint.revision
    for phrase, expected in baseline["prompts"].items():
        actual = [frame.score for frame in result.frames if frame.phrase == phrase]
        assert len(actual) == len(expected) == 250
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)
