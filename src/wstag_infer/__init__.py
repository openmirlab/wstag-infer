"""Public inference-only API for the small WSTAG checkpoint.

Reads: session, results, checkpoints, and __about__; importing does not load Torch.
"""

from .__about__ import __version__
from .checkpoints import Checkpoint, default_checkpoint
from .results import Event, FrameScore, LocalizationResult, Window, result_semantics
from .session import WSTAGSession, localize

__all__ = [
    "Checkpoint",
    "Event",
    "FrameScore",
    "LocalizationResult",
    "WSTAGSession",
    "Window",
    "__version__",
    "default_checkpoint",
    "localize",
    "result_semantics",
]
