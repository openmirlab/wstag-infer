"""Check lifecycle and source-clock localization without checkpoint downloads.

Reads: the public session API, synthetic temporary WAVs, and a deterministic
model double whose output grid follows the checkpoint's 40 ms shape.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf
import torch

from wstag_infer import WSTAGSession, result_semantics
from wstag_infer.results import FRAME_HOP_SECONDS


class GridModel:
    def __init__(self, *, values: list[float] | None = None):
        self.calls: list[tuple[int, int, str]] = []
        self.values = values

    def __call__(self, *, audio, audio_len, text):
        self.calls.append((audio.shape[-1], int(audio_len[0]), text[0]))
        count = audio.shape[-1] // round(32_000 * FRAME_HOP_SECONDS)
        if self.values is None:
            # A later overlapping window wins; this tests max aggregation.
            score = 0.2 + 0.1 * (len(self.calls) - 1)
            return torch.full((1, count), score)
        return torch.tensor([self.values[:count]])


def _ready(model: GridModel) -> WSTAGSession:
    session = WSTAGSession(device="cpu")
    session._model = model
    session._resolved_device = "cpu"
    session._status = "ready"
    return session


def test_short_input_padding_discards_scores_beyond_real_audio(tmp_path):
    path = tmp_path / "short.wav"
    sf.write(path, np.zeros(3200, dtype="float32"), 32_000)
    model = GridModel(values=[0.2, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
    session = _ready(model)

    result = session.infer(path, "a siren", threshold=0.5)

    assert model.calls == [(10_240, 10_240, "a siren")]
    assert [(frame.start, frame.end) for frame in result.frames] == [
        (0.0, 0.04), (0.04, 0.08), (0.08, 0.1)
    ]
    assert [(event.start, event.end) for event in result.events] == [(0.04, 0.1)]
    assert result.analyzed_until == 0.1
    assert result.to_dict()["semantics"]["version"] == result.semantics_version
    json.dumps(result.to_dict())


def test_long_file_window_map_overlap_and_trailing_partial_bin(tmp_path):
    path = tmp_path / "long_stereo.wav"
    sf.write(path, np.zeros((18 * 32_000 + 3200, 2), dtype="float32"), 32_000)
    model = GridModel()
    session = _ready(model)

    result = session.infer(path, ["glass"])

    assert [window.start for window in result.windows] == [0.0, 8.0, 16.0]
    assert [window.frame_count for window in result.windows] == [250, 250, 52]
    assert len(result.frames) == 452
    assert result.analyzed_until == pytest.approx(18.08)
    assert result.frames[-1].end == pytest.approx(18.08)
    # Second window takes the max on the 8–10 s overlap.
    assert result.frames[200].score == pytest.approx(0.3)
    assert result.frames[399].score == pytest.approx(0.3)
    # Third window takes the max on the 16–18 s overlap.
    assert result.frames[400].score == pytest.approx(0.4)
    assert not result.events


def test_input_validation_and_terminal_close(tmp_path):
    path = tmp_path / "one.wav"
    sf.write(path, np.zeros(3200, dtype="float32"), 32_000)
    model = GridModel()
    session = _ready(model)
    with pytest.raises(ValueError, match="unique"):
        session.infer(path, ["bell", "bell"])
    with pytest.raises(ValueError, match="threshold"):
        session.infer(path, "bell", threshold=1.2)
    assert not model.calls
    session.release()
    assert session.status == "unloaded"
    with pytest.raises(RuntimeError, match="load"):
        session.infer(path, "bell")
    session.close()
    session.close()
    assert session.status == "closed"
    with pytest.raises(RuntimeError, match="closed"):
        session.load()


def test_semantics_make_uncalibrated_scores_and_coverage_explicit():
    semantics = result_semantics()
    assert "not calibrated" in semantics["score"]
    assert semantics["frame_hop_seconds"] == 0.04
    assert "unscored" in semantics["coverage"]
