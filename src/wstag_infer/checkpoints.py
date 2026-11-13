"""Resolve and verify the pinned WSTAG checkpoint from one source of truth.

Reads: config/checkpoints.toml, Hugging Face snapshot_download, and SHA-256.
The session and cache_info call the same resolver so their answers cannot drift.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


@dataclass(frozen=True)
class Checkpoint:
    """Pinned source and integrity metadata for one compatible WSTAG checkpoint."""

    key: str
    repo_id: str
    revision: str
    source_repo: str
    source_revision: str
    source_file: str
    snapshot_code_sha256: str
    checkpoint_url: str
    checkpoint_license: str
    source_license: str
    verified_date: str
    files: dict[str, str]


def default_checkpoint() -> Checkpoint:
    """Read the package-owned manifest without loading model dependencies."""
    data = tomllib.loads(files("wstag_infer").joinpath("config/checkpoints.toml").read_text())
    if "model" not in data or not isinstance(data["model"], dict):
        raise ValueError("WSTAG checkpoint manifest has no [model] table")
    model = data["model"]
    required = set(Checkpoint.__dataclass_fields__) - {"files"}
    missing = required - model.keys()
    if missing:
        raise ValueError(f"WSTAG checkpoint manifest missing: {sorted(missing)}")
    assets = model.get("files")
    if not isinstance(assets, dict) or set(assets) != {
        "config.json", "model.safetensors", "vocab.json"
    }:
        raise ValueError("WSTAG checkpoint manifest must name config, safetensors, and vocab")
    for name, digest in assets.items():
        if Path(name).name != name or len(digest) != 64 or any(
            ch not in "0123456789abcdef" for ch in digest
        ):
            raise ValueError(f"Invalid WSTAG checkpoint file or SHA-256: {name}")
    return Checkpoint(**{key: model[key] for key in required}, files=dict(assets))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_snapshot(
    checkpoint: Checkpoint,
    *,
    cache_dir: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> Path:
    """Return a complete, hash-verified local snapshot or fail before model work."""
    if checkpoint_dir is not None:
        directory = Path(checkpoint_dir).expanduser().resolve()
    else:
        from huggingface_hub import snapshot_download

        directory = Path(
            snapshot_download(
                checkpoint.repo_id,
                revision=checkpoint.revision,
                allow_patterns=list(checkpoint.files),
                cache_dir=str(cache_dir) if cache_dir is not None else None,
                local_files_only=local_files_only,
            )
        )
    for name, expected in checkpoint.files.items():
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(f"WSTAG snapshot missing {name}: {directory}")
        if _sha256(path) != expected:
            raise ValueError(f"WSTAG snapshot SHA-256 mismatch: {path}")
    return directory
