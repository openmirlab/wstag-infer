"""Declare WSTAG score, time, and interval meaning independently of model load.

Reads: the pinned author frontend's 10 ms mel hop and fourfold temporal pooling,
plus the package session's window/overlap and threshold policies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

SEMANTICS_VERSION = "wstag-localization-v1"
FRAME_HOP_SECONDS = 0.04
WINDOW_SECONDS = 10.0
WINDOW_HOP_SECONDS = 8.0


@dataclass(frozen=True)
class FrameScore:
    """One nominal [start, end) bin on the decoded input-file clock."""

    start: float
    end: float
    score: float
    phrase: str


@dataclass(frozen=True)
class Event:
    """A threshold-derived, approximate [start, end) group of adjacent bins."""

    start: float
    end: float
    peak_score: float
    phrase: str


@dataclass(frozen=True)
class Window:
    """Actual source-time coverage for one model call."""

    start: float
    input_end: float
    scored_until: float
    frame_count: int


@dataclass(frozen=True)
class LocalizationResult:
    """All effective timing and scoring evidence for one input file."""

    duration: float
    phrases: tuple[str, ...]
    frames: tuple[FrameScore, ...]
    events: tuple[Event, ...]
    windows: tuple[Window, ...]
    threshold: float | None
    checkpoint_revision: str
    device: str
    analyzed_until: float
    semantics_version: str = SEMANTICS_VERSION

    def to_dict(self) -> dict:
        """Return a JSON-serializable dictionary, including effective semantics."""
        return {**asdict(self), "semantics": result_semantics()}


def result_semantics() -> dict:
    """Describe the native result without importing or loading model weights."""
    return {
        "version": SEMANTICS_VERSION,
        "scope": "input_file",
        "time_unit": "seconds",
        "time_origin": "start of decoded input file",
        "frame_interval": "[start,end)",
        "frame_anchor": "nominal bin start; first bin starts at 0",
        "frame_hop_seconds": FRAME_HOP_SECONDS,
        "frontend": "32 kHz mono; 32 ms mel window; 10 ms mel hop; center-padded STFT; fourfold temporal pooling",
        "model_context": "CNN and bidirectional GRU; effective acoustic onset precision is not established by the 40 ms output grid",
        "short_input": "zero-pad only below 0.32 s, then discard bins beyond the real input",
        "window_seconds": WINDOW_SECONDS,
        "window_hop_seconds": WINDOW_HOP_SECONDS,
        "window_aggregation": "maximum score for equal source-time bins in overlapping windows",
        "score": "author sigmoid of audio/text dot product; not calibrated event probability",
        "threshold": "caller-supplied inclusive bin cutoff; None returns no events",
        "event_interval": "contiguous passing bins on input-file clock, [start,end); approximate, not exact acoustic onset/offset",
        "coverage": "windows and analyzed_until report actual scored input-time extent; any trailing partial bin may be unscored",
        "text": "English whitespace tokens matched against the pinned 5,221-word vocabulary; unmatched tokens map to <unk>",
    }
