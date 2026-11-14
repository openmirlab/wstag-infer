# wstag-infer maintainer guide

This is a public standalone, inference-only package for the small, author-published
WSTAG Cnn8Rnn + word-embedding checkpoint. Xuenan Xu, Ziyang Ma, Mengyue Wu,
and Kai Yu created the original research and model. The package carries only
their fixed HF inference implementation plus OpenMIRLab's verified loading,
audio normalization, long-file localization, result meaning, and lifecycle.
Do not add training, fine-tuning, dataset preparation, evaluation frameworks,
search embeddings, or the larger LAION-CLAP checkpoint without a separately
verified profile and parity baseline.

The public API is `WSTAGSession`, `localize()`, and typed localization results.
Users install the maintained source from `openmirlab/wstag-infer` on GitHub;
there is no PyPI release. The Traditional Chinese usage guide in
`docs/usage.zh-TW.md` must remain consistent with README.md and the manifest.
`load()` must verify fixed config/weights/vocab before model construction and
reuse the same model for repeated `infer()` calls. `infer()` requires ready
status. `release()` retains disk assets, `close()` is terminal. Score bins are
nominal 40 ms source-clock intervals; scores are uncalibrated sigmoid matches;
thresholds are caller-selected. See `result_semantics()` and README for exact
window, padding, overlap, and coverage policies.

Keep the `_impl/modeling_wstag.py` source provenance in `docs/source-provenance.md`.
Every load-bearing source file starts with a title, rationale, and `Reads:` line.
The checkpoint manifest in `src/wstag_infer/config/checkpoints.toml` is the only
runtime owner of artifact hashes and revision. Check README and CLAUDE.md
together for any API, checkpoint, install, lifecycle, or semantics change; update
CHANGELOG.md for user-visible changes. No weights enter Git or wheels.

Verification from this checkout:

`.github/workflows/ci.yml` gates offline tests and lint on Python 3.10–3.15,
plus a clean installed-wheel smoke on Python 3.12. It excludes `model` tests
from default CI because they require the pinned author checkpoint. Run them
separately for release parity.

```bash
uv sync --extra model --extra dev
uv run --extra model --extra dev pytest -q
uv run --extra model --extra dev pytest -q -m model
uv run --extra dev ruff check src tests
uv run --extra dev python -m build
```

Before a tagged release, additionally install the wheel built from the sdist
into a clean environment outside this checkout and exercise offline inference,
repeat calls, release/reload, and terminal close. Verify the Python-version
matrix and CUDA behavior. Repository creation, weight rehosting, and release
are separate outward-facing actions.
