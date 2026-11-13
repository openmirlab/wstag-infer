"""Verify fixed-asset integrity and shared cache/load resolution semantics.

Reads: the public checkpoint manifest and local snapshot resolver.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from wstag_infer import WSTAGSession, default_checkpoint
from wstag_infer.checkpoints import resolve_snapshot


def test_manifest_names_the_complete_pinned_snapshot():
    checkpoint = default_checkpoint()
    assert len(checkpoint.revision) == 40
    assert set(checkpoint.files) == {"config.json", "model.safetensors", "vocab.json"}
    assert checkpoint.source_file == "hf_modeling_grounding.py"


def test_local_snapshot_is_hash_checked_before_load(tmp_path):
    files = {"config.json": b"config", "model.safetensors": b"weights", "vocab.json": b"vocab"}
    for name, content in files.items():
        (tmp_path / name).write_bytes(content)
    checkpoint = replace(
        default_checkpoint(),
        files={name: hashlib.sha256(content).hexdigest() for name, content in files.items()},
    )
    assert resolve_snapshot(checkpoint, checkpoint_dir=tmp_path, local_files_only=True) == tmp_path
    session = WSTAGSession(checkpoint=checkpoint, checkpoint_dir=tmp_path)
    assert session.cache_info()["cached"] is True

    (tmp_path / "vocab.json").write_bytes(b"changed")
    assert session.cache_info()["cached"] is False
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        resolve_snapshot(checkpoint, checkpoint_dir=tmp_path)

    (tmp_path / "vocab.json").unlink()
    with pytest.raises(FileNotFoundError, match="missing vocab.json"):
        resolve_snapshot(checkpoint, checkpoint_dir=tmp_path)
