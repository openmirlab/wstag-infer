# Changelog

## Unreleased

- Gate pushes and pull requests with Python 3.10–3.15 offline tests, lint, and a clean wheel-from-sdist import check; keep real-checkpoint parity separate.

## 0.1.0 — 2026-11-13

- First inference-only package for the small WSTAG Cnn8Rnn + word-embedding checkpoint.
- Verify the fixed config, safetensors weights, and vocabulary by SHA-256 before load.
- Return dense source-clock scores and optional caller-thresholded intervals for short and long WAV/FLAC files.
- Provide a reusable session, offline/manual checkpoint loading, CPU/CUDA device selection, and a JSON CLI.
- Publish the standalone source repository and Traditional Chinese usage guide; maintained installation is from GitHub, not PyPI.
