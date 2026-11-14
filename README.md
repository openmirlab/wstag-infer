# wstag-infer

Locate a described sound in WAV or FLAC audio with the small WSTAG model.

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE) [![Python: 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

## Why this exists

[WSTAG](https://github.com/wsntxxn/TextToAudioGrounding) produces dense audio–text match scores, but the author's Hugging Face loader fetches its vocabulary without a fixed revision. This inference-only package keeps the author's model implementation locally, verifies the fixed configuration, weights, and vocabulary, and maps scores from overlapping windows to the input-file clock. The original research and model belong to the WSTAG authors.

**Status:** public inference-only source repository; no PyPI release. The included model implementation is copied from the author's fixed [Hugging Face snapshot](https://huggingface.co/wsntxxn/cnn8rnn-w2vmean-audiocaps-grounding/tree/a9eb20c3e2135e0ca82d747bb99ae10ee4eaec66), with only a provenance docstring added. The model card declares Apache-2.0; the separate GitHub research tree has an MIT license. See [source provenance](docs/source-provenance.md) and [NOTICE](NOTICE). Checkpoint weights are downloaded to the user's cache and are never bundled.

## Acknowledgments

- Original research and model: Xuenan Xu, Ziyang Ma, Mengyue Wu, and Kai Yu ([paper](https://arxiv.org/abs/2401.02584)).
- Original research source: [wsntxxn/TextToAudioGrounding](https://github.com/wsntxxn/TextToAudioGrounding); adapted inference code and checkpoint: [author's Hugging Face model](https://huggingface.co/wsntxxn/cnn8rnn-w2vmean-audiocaps-grounding).
- OpenMIRLab did not train this checkpoint.

## Citation

Please cite the [WSTAG paper](https://arxiv.org/abs/2401.02584) when using the model:

```bibtex
@article{xu2024towards,
  title={Towards Weakly Supervised Text-to-Audio Grounding},
  author={Xu, Xuenan and Ma, Ziyang and Wu, Mengyue and Yu, Kai},
  journal={arXiv preprint arXiv:2401.02584},
  year={2024}
}
```

## Features

- Per-phrase score tracks on nominal 40 ms bins, plus optional threshold-derived event intervals.
- Source-time mapping for long files through overlapping 10-second windows.
- Hash-verified checkpoint loading, offline cache or local-snapshot use, explicit CPU/CUDA selection, and JSON CLI output.

## Scope

This package only performs text-to-audio localization with the small `cnn8rnn-w2vmean-audiocaps` checkpoint. It does not produce retrieval embeddings, train or fine-tune models, or include the larger LAION-CLAP WSTAG checkpoint.

## Install

Python 3.10+ is declared. The recorded CPU parity environment is Python 3.12, Torch 2.15.0, TorchAudio 2.11.1, and Transformers 4.57.6. Python 3.10 has passed an installed-wheel import and metadata check; full model inference has been verified on 3.12. Install from public GitHub:

```bash
uv pip install 'wstag-infer[model] @ git+https://github.com/openmirlab/wstag-infer.git'
```

For a local development checkout:

```bash
uv sync --extra model
```

The `model` extra supplies TorchAudio for the author's mel frontend and for resampling; its binary must match the installed Torch. SoundFile decodes WAV/FLAC. No TorchCodec or remote-code execution is used. Transformers 5 is currently excluded pending parity testing; see [compatibility tracking](docs/compatibility.md). The base package remains importable without TorchAudio, and `load()` gives an install hint when it is missing.

## Quick start

```python
from wstag_infer import WSTAGSession

with WSTAGSession(device="cpu") as model:
    result = model.infer("recording.wav", ["the sound of glass breaking"], threshold=0.5)
    print(result.events)   # threshold-derived, approximate intervals
    print(result.frames)   # nominal 40 ms score bins on the input-file clock
```

Run this using `uv run --extra model python example.py` from a checkout, or call `wstag-infer recording.wav 'the sound of glass breaking' --threshold 0.5` for JSON after installation. `localize(path, phrases, threshold=...)` provides a one-shot call. For repeated files, reuse a `WSTAGSession`: `load()` loads once, `infer()` requires a ready session, `release()` removes the resident model, and `close()` ends the session. `status` and `cache_info()` expose lifecycle and offline asset state. A session is not safe for concurrent `infer()` calls. A [Traditional Chinese usage guide](docs/usage.zh-TW.md) includes installation, offline setup, and result interpretation.

## Usage and output

Input is a SoundFile-readable audio path, principally WAV or FLAC. Stereo or multichannel input is averaged to mono, then resampled to 32 kHz. The package reads at most 10 seconds per model call, advancing by 8 seconds on long files. A real input shorter than 0.32 seconds is zero-padded for model execution, and scores beyond the actual audio are discarded. Overlapping predictions for the same source-time bin are combined by maximum score. The result includes `windows` and `analyzed_until` to show effective coverage. Memory use is one audio window at a time, plus the returned score tracks.

Scores are the author's **uncalibrated sigmoid audio–text matches**, not event probabilities. `threshold=None` returns only frame scores; a caller-supplied threshold in `[0,1]` groups adjacent passing bins into approximate `[start,end)` intervals. The 40 ms grid is a **nominal score grid, not guaranteed acoustic boundary precision**: the CNN and bidirectional GRU use surrounding context. The result carries `semantics_version="wstag-localization-v1"`, checkpoint revision, device, window map, threshold, and the meaning of every field through `result_semantics()` or `result.to_dict()`. For decisions near event edges, inspect the score curve and original audio.

Text uses the checkpoint's English whitespace-token vocabulary. Unrecognized words map to `<unk>`; short, lowercase sound descriptions similar to the author's examples are a sensible starting point. Phrase wording can substantially affect scores.

## Checkpoint and cache

The pinned revision and SHA-256 values for `config.json`, `model.safetensors`, and `vocab.json` are in [the checkpoint manifest](src/wstag_infer/config/checkpoints.toml). `load()` downloads these files into the Hugging Face cache on first use, normally `~/.cache/huggingface/hub`, and verifies all three before constructing the model. `cache_info()` checks the same resolver without downloading. Set `cache_dir=` for a custom Hugging Face cache and `local_files_only=True` to prohibit network access. For a manually prepared directory containing the three verified files, pass `checkpoint_dir=`. The [usage guide](docs/usage.zh-TW.md#離線使用) gives the exact manual-download command. A mismatch fails before model construction.

The author model code is in the package, so `trust_remote_code` is never needed. `device` accepts `auto`, `cpu`, `cuda`, and `cuda:N`; MPS is currently rejected. `auto` chooses CUDA when available, else CPU.

## What this project will never bundle

The wheel contains the pinned inference model implementation, localization adapter, checkpoint metadata, and result types. It never bundles checkpoint weights, training loops, fine-tuning, dataset loaders, or evaluation harnesses.

## Development

GitHub CI runs offline tests (`pytest -m 'not model'`) and lint on Python
3.10–3.15, then checks a wheel built from its sdist in a clean Python 3.12
environment. The author-output comparison remains a separate checkpoint-backed
verification step.

The baseline fixture records the author's direct model output before packaging. Model-marked tests compare real checkpoint scores to that baseline; other tests cover asset integrity, lifecycle, short and long audio mapping, and threshold semantics. Run from the checkout:

```bash
uv sync --extra model --extra dev
uv run --extra model --extra dev pytest -q
uv run --extra model --extra dev pytest -q -m model
uv run --extra dev ruff check src tests
uv run --extra dev python -m build
```

This package is independent of Tactus and Phonon. Any future adapter should preserve the revision, source-time window map, raw score meaning, and threshold provenance.

## License

Package source is Apache-2.0. The copied model implementation comes from the Apache-2.0-tagged author Hugging Face model card; the author's separate research repository is MIT licensed. See [LICENSE](LICENSE), [upstream MIT license](LICENSE.UPSTREAM-MIT), [NOTICE](NOTICE), and [source provenance](docs/source-provenance.md). The checkpoint is fetched from the author and is not redistributed here.

## Support

Report package issues in [openmirlab/wstag-infer issues](https://github.com/openmirlab/wstag-infer/issues). For model or research questions, consult the [original WSTAG repository](https://github.com/wsntxxn/TextToAudioGrounding).
