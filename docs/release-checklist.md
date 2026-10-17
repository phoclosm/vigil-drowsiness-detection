# Vigil v0.1.0 Release Checklist

This checklist prepares the repository for maintainer review. Completing the
code and documentation items does not authorize creating a Git tag, GitHub
release, or downloadable model artifact.

## Repository preparation

- [x] Package version is `0.1.0`.
- [x] `python -m vigil` and the `vigil` console script target the live demo.
- [x] A separate `vigil-benchmark` console script targets bounded benchmarking.
- [x] README setup, demo controls, and safety limitations are documented.
- [x] Architecture and evidence-status documents are linked from the README.
- [x] Datasets, recordings, checkpoints, and generated demo media are excluded
  from normal version control.

## Validation before review

Run these commands from a clean checkout with Python 3.10 or newer:

```powershell
python -m pip install -e ".[dev,ml]"
python -m pytest
python -m ruff check .
python -m compileall -q src
python -m vigil --help
python -m vigil.benchmark --help
```

Build a wheel and source archive in a temporary directory or clean build
environment. Inspect their contents before publication so that local data,
models, recordings, caches, and instructions such as `AGENTS.md` are absent.

## Manual demo gates

- [ ] Obtain an approved, provenance-complete trained model bundle.
- [ ] Confirm the live feature settings match the bundle's training settings.
- [ ] Run a bounded webcam or licensed-video demo on identified hardware.
- [ ] Verify the model window warms up and all three overlay colors render.
- [ ] Verify `q`, `Q`, Escape, video end, and `--max-frames` close cleanly.
- [ ] Review any screenshot or video for consent, privacy, licensing, and
  truthful model output before adding it to a release.

## Evidence and release gates

- [ ] Attach a reviewed model card before publishing model-quality claims.
- [ ] Add only measured performance values with revision, hardware, input,
  configuration, frame count, and measurement method.
- [ ] Select and add a project license with maintainer approval.
- [ ] Obtain maintainer approval for the final diff and release notes.
- [ ] Create the `v0.1.0` tag only after approval.
- [ ] Publish the GitHub release only after the tag and artifacts are verified.

Unchecked items remain explicit release blockers or limitations.
