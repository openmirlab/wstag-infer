"""Own WSTAG model residency, audio windows, and input-file time mapping.

Reads: checkpoint resolver, the locally adapted author model, SoundFile,
TorchAudio resampling, and the package's result semantics.
"""

from __future__ import annotations

from pathlib import Path

import soundfile as sf

from .checkpoints import Checkpoint, default_checkpoint, resolve_snapshot
from .results import (
    FRAME_HOP_SECONDS,
    WINDOW_HOP_SECONDS,
    WINDOW_SECONDS,
    Event,
    FrameScore,
    LocalizationResult,
    Window,
)

MODEL_SAMPLE_RATE = 32_000
MIN_INPUT_SECONDS = 0.32


def _device(value: str) -> str:
    import torch

    if value == "mps":
        raise ValueError("WSTAG does not support device='mps'")
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cpu":
        return value
    if value == "cuda" or value.startswith("cuda:"):
        try:
            target = torch.device(value)
        except RuntimeError as exc:
            raise ValueError(f"Unsupported device: {value}") from exc
        if not torch.cuda.is_available() or (
            target.index is not None and target.index >= torch.cuda.device_count()
        ):
            raise ValueError(f"Requested CUDA device is unavailable: {value}")
        return value
    raise ValueError(f"Unsupported device: {value}")


def _starts(duration: float) -> tuple[float, ...]:
    starts = [0.0]
    while starts[-1] + WINDOW_SECONDS < duration - 1e-8:
        starts.append(starts[-1] + WINDOW_HOP_SECONDS)
    return tuple(starts)


def _events(frames: tuple[FrameScore, ...], phrases: tuple[str, ...], threshold: float | None) -> tuple[Event, ...]:
    if threshold is None:
        return ()
    events: list[Event] = []
    for phrase in phrases:
        selected = sorted((f for f in frames if f.phrase == phrase), key=lambda f: f.start)
        start: float | None = None
        end = 0.0
        peak = 0.0
        for frame in selected:
            if frame.score < threshold:
                if start is not None:
                    events.append(Event(start, end, peak, phrase))
                    start = None
                continue
            if start is None:
                start, end, peak = frame.start, frame.end, frame.score
            elif frame.start <= end + 1e-8:
                end = frame.end
                peak = max(peak, frame.score)
            else:
                events.append(Event(start, end, peak, phrase))
                start, end, peak = frame.start, frame.end, frame.score
        if start is not None:
            events.append(Event(start, end, peak, phrase))
    return tuple(events)


class WSTAGSession:
    """Reusable, independently loaded small WSTAG model. Not thread-safe."""

    def __init__(
        self,
        *,
        device: str = "auto",
        cache_dir: str | Path | None = None,
        checkpoint_dir: str | Path | None = None,
        checkpoint: Checkpoint | None = None,
        local_files_only: bool = False,
    ) -> None:
        self.device = device
        self.cache_dir = cache_dir
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint = checkpoint or default_checkpoint()
        self.local_files_only = local_files_only
        self._model = None
        self._resolved_device: str | None = None
        self._status = "unloaded"

    @property
    def status(self) -> str:
        return self._status

    def cache_info(self) -> dict:
        """Inspect the same verified snapshot resolution used by load(), offline."""
        try:
            path = resolve_snapshot(
                self.checkpoint,
                cache_dir=self.cache_dir,
                checkpoint_dir=self.checkpoint_dir,
                local_files_only=True,
            )
        except (OSError, ValueError, FileNotFoundError) as exc:
            return {"cached": False, "revision": self.checkpoint.revision, "path": None, "error": str(exc)}
        return {"cached": True, "revision": self.checkpoint.revision, "path": str(path), "error": None}

    def load(self) -> WSTAGSession:
        """Verify assets and load one model; repeated calls retain the same instance."""
        if self._status == "closed":
            raise RuntimeError("WSTAG session is closed")
        if self._model is not None:
            return self
        self._status = "loading"
        try:
            device = _device(self.device)
            snapshot = resolve_snapshot(
                self.checkpoint,
                cache_dir=self.cache_dir,
                checkpoint_dir=self.checkpoint_dir,
                local_files_only=self.local_files_only,
            )
            try:
                import torchaudio  # noqa: F401 - required by the author frontend and resampler
            except ImportError as exc:
                raise ImportError("WSTAG requires torchaudio; install wstag-infer[model]") from exc
            from ._impl.modeling_wstag import Cnn8RnnW2vMeanGroundingModel

            model = Cnn8RnnW2vMeanGroundingModel.from_pretrained(
                str(snapshot), local_files_only=True, use_safetensors=True
            ).to(device).eval()
            if model.config.sample_rate != MODEL_SAMPLE_RATE:
                raise ValueError(f"Unexpected WSTAG sample rate: {model.config.sample_rate}")
            self._model = model
            self._resolved_device = device
            self._status = "ready"
        except Exception:
            self._model = None
            self._resolved_device = None
            self._status = "failed"
            raise
        return self

    def infer(
        self,
        audio_path: str | Path,
        phrases: str | list[str] | tuple[str, ...],
        *,
        threshold: float | None = None,
    ) -> LocalizationResult:
        """Return native frame scores and optional threshold-derived event spans."""
        if self._status != "ready":
            raise RuntimeError("Call load() before infer()")
        if isinstance(phrases, str):
            phrases = (phrases,)
        else:
            phrases = tuple(phrases)
        if not phrases or any(not isinstance(p, str) or not p.strip() for p in phrases):
            raise ValueError("phrases must contain nonempty strings")
        if len(set(phrases)) != len(phrases):
            raise ValueError("phrases must be unique within one call")
        if threshold is not None and not 0 <= threshold <= 1:
            raise ValueError("threshold must be in [0,1]; scores are not calibrated probabilities")
        path = Path(audio_path).expanduser().resolve()
        info = sf.info(str(path))
        duration = info.frames / info.samplerate
        if duration <= 0:
            raise ValueError("Audio must be nonempty")

        import torch
        import torch.nn.functional as torch_functional
        import torchaudio

        scores: dict[tuple[int, int], float] = {}
        windows: list[Window] = []
        with sf.SoundFile(str(path)) as source:
            for start in _starts(duration):
                source.seek(round(start * info.samplerate))
                samples = source.read(
                    round(WINDOW_SECONDS * info.samplerate), dtype="float32", always_2d=True
                )
                real_seconds = len(samples) / info.samplerate
                waveform = torch.from_numpy(samples.mean(axis=1).copy())
                if info.samplerate != MODEL_SAMPLE_RATE:
                    waveform = torchaudio.functional.resample(
                        waveform, info.samplerate, MODEL_SAMPLE_RATE
                    )
                real_model_samples = waveform.numel()
                minimum = round(MIN_INPUT_SECONDS * MODEL_SAMPLE_RATE)
                if real_model_samples < minimum:
                    waveform = torch_functional.pad(
                        waveform, (0, minimum - real_model_samples)
                    )
                waveform = waveform.to(self._resolved_device)
                used_frames = 0
                scored_until = start
                for phrase_index, phrase in enumerate(phrases):
                    with torch.inference_mode():
                        track = self._model(
                            audio=waveform[None, :],
                            audio_len=torch.tensor([waveform.numel()]),
                            text=[phrase],
                        )[0].detach().cpu().numpy()
                    for local_index, value in enumerate(track):
                        local_start = local_index * FRAME_HOP_SECONDS
                        if local_start >= real_seconds - 1e-9:
                            break
                        index = round((start + local_start) / FRAME_HOP_SECONDS)
                        key = (phrase_index, index)
                        scores[key] = max(scores.get(key, -1.0), float(value))
                        if phrase_index == 0:
                            used_frames += 1
                            scored_until = min(duration, start + min(real_seconds, local_start + FRAME_HOP_SECONDS))
                windows.append(
                    Window(
                        start=round(start, 9),
                        input_end=round(min(duration, start + real_seconds), 9),
                        scored_until=round(scored_until, 9),
                        frame_count=used_frames,
                    )
                )
        frames = tuple(
            FrameScore(
                start=round(index * FRAME_HOP_SECONDS, 9),
                end=round(min(duration, (index + 1) * FRAME_HOP_SECONDS), 9),
                score=value,
                phrase=phrases[phrase_index],
            )
            for (phrase_index, index), value in sorted(scores.items())
        )
        return LocalizationResult(
            duration=duration,
            phrases=phrases,
            frames=frames,
            events=_events(frames, phrases, threshold),
            windows=tuple(windows),
            threshold=threshold,
            checkpoint_revision=self.checkpoint.revision,
            device=self._resolved_device or "unknown",
            analyzed_until=max(window.scored_until for window in windows),
        )

    def release(self) -> None:
        """Release resident model memory while preserving verified disk assets."""
        if self._status == "closed":
            return
        self._model = None
        self._resolved_device = None
        self._status = "unloaded"
        if self.device.startswith("cuda") or self.device == "auto":
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    def close(self) -> None:
        """Release memory and prevent later reloads; safe to call repeatedly."""
        if self._status != "closed":
            self.release()
            self._status = "closed"

    def __enter__(self) -> WSTAGSession:
        return self.load()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def localize(
    audio_path: str | Path,
    phrases: str | list[str] | tuple[str, ...],
    *,
    threshold: float | None = None,
    device: str = "auto",
    cache_dir: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> LocalizationResult:
    """One-shot convenience call; creates and closes a fresh session each time."""
    with WSTAGSession(
        device=device,
        cache_dir=cache_dir,
        checkpoint_dir=checkpoint_dir,
        local_files_only=local_files_only,
    ) as session:
        return session.infer(audio_path, phrases, threshold=threshold)
