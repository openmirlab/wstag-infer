# Compatibility tracking

The pinned author inference code and checkpoint were verified with Python 3.12,
Torch 2.15.0, TorchAudio 2.11.1, and Transformers 4.57.6 on CPU. A 10-second
offline CUDA smoke test also passed on an RTX 4090 with the same environment.
The installed wheel's import, manifest, and semantics API passed on Python
3.10.18; full model inference on 3.10 and the remaining declared Python versions
has not been verified. The package
currently declares `transformers>=4.57.6,<5` because Transformers 5 has not
passed author-output parity. Removing the upper bound requires a pinned
checkpoint run against `tests/fixtures/upstream_cpu.json`, short/tail time-map
tests, and an installed-wheel inference check. Until a public issue tracker
exists, this file is the explicit compatibility tracker.

TorchAudio is an optional `model` extra and its binary must match Torch.
WAV/FLAC input decoding uses SoundFile; TorchCodec is not required. CPU and CUDA
paths are implemented by the session; explicit `mps` is rejected. CUDA numeric
parity and the full Python-version matrix must be recorded before a public release.
