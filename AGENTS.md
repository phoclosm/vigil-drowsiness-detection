# Vigil Repository Instructions

## Scope

These instructions apply to the entire repository. More specific `AGENTS.md`
files may refine them for a subdirectory, but they must not weaken the project
safety, evidence, data-handling, or collaboration rules below.

## Project Context

Vigil 2.0 is a from-scratch, collaborative AI/ML learning project maintained
by two contributors, M1 and M2. Its goal is a portfolio-quality real-time
driver drowsiness detection prototype built through understandable,
well-tested implementation.

Vigil is not a production safety system. Do not describe or present it as a
substitute for attentive driving, certified vehicle safety equipment, or a
medical diagnosis.

The intended real-time pipeline is:

`Webcam/Video -> Face & Eye Landmarks -> Eye/Blink Features -> Temporal Window -> Fatigue Classification -> ACTIVE / DROWSY / SLEEPING -> Alert -> Live UI`

The separate model-development pipeline is:

`Recorded Features -> Preprocessing -> PyTorch Dataset/DataLoader -> Training -> Validation -> Evaluation -> Saved Model -> Real-Time Inference`

The Week 1 target is a functional Vigil v0.1 prototype. Build only the current
approved milestone; do not implement future days or milestones early.

## Project Objectives

- Learn computer vision, machine learning, and deep learning through direct
  implementation and clear explanations.
- Deliver a working, reproducible, portfolio-quality GitHub project.
- Maintain meaningful branches, commits, pull requests, tests, and
  documentation.
- Establish a simple classical or rule-based baseline before adding learned
  ML/DL approaches.
- Compare approaches using measured, reproducible evidence.
- Keep the code modular enough that capture, feature extraction,
  classification, training, evaluation, and presentation can be tested and
  changed independently.

## Ownership

M1 owns:

- project and runtime architecture
- OpenCV video capture
- facial and eye landmark detection
- blink and eye-aspect-ratio (EAR) feature extraction
- runtime benchmarking
- real-time model integration
- overlays, demo preparation, and release work

M2 owns:

- data schema and feature recording
- baseline fatigue classification
- PyTorch `Dataset` and `DataLoader`
- neural model training
- evaluation
- checkpoint export
- model-quality testing

Ownership identifies the primary implementer and reviewer, not a reason to
create tightly coupled silos. Changes that cross an ownership boundary should
make the interface explicit and request review from the other owner. Do not
silently change another owner's public schema, feature meaning, model input or
output contract, thresholds, or runtime integration contract.

## Engineering Rules

- Prefer the simplest understandable solution that satisfies the current
  milestone.
- Do not add technologies, services, abstractions, or dependencies merely for
  novelty or buzzword value.
- Keep modules focused and interfaces explicit. Separate I/O from pure
  feature, classification, and evaluation logic where practical.
- Avoid hidden global state and machine-specific paths. Put configuration in
  explicit arguments or documented configuration once it is needed.
- Add type hints and concise docstrings where they improve clarity; comments
  should explain intent or non-obvious decisions rather than restate code.
- Handle missing cameras, unreadable frames, absent landmarks, malformed
  records, and unavailable model files explicitly when those cases enter the
  approved milestone.
- Preserve the three public fatigue-state names exactly as `ACTIVE`, `DROWSY`,
  and `SLEEPING` unless an approved design change says otherwise.
- Do not begin the next day's work until the maintainers explicitly request
  it.

## Evidence and Evaluation Integrity

- Never fabricate or estimate claimed accuracy, precision, recall, F1,
  confusion matrices, FPS, latency, resource use, dataset size, or other
  experimental results.
- Label illustrative or hypothetical values clearly; do not present them as
  measurements.
- Record enough context to reproduce every reported result: code revision,
  data split or source, hardware when relevant, configuration, thresholds,
  sample count, and measurement method.
- Keep training, validation, and test data logically separate. Prevent subject
  or session leakage when the available data makes that distinction possible.
- Compare the rule-based baseline and learned approaches under compatible
  inputs, splits, metrics, and operating conditions. Document limitations and
  failed experiments as honestly as successes.
- Do not tune against the final test set or select only favorable runs.

## Tests and Quality Checks

- Add or update focused tests with each behavior change. Prefer deterministic
  unit tests for pure transformations and small contract tests at module
  boundaries.
- Keep hardware-dependent camera, UI, and timing checks separate from the
  default deterministic test suite.
- Use synthetic fixtures only when they test code behavior; never imply that
  synthetic fixtures demonstrate real-world model quality.
- Run the relevant tests and formatting or lint checks before requesting
  review. Report exactly what was run and disclose anything that could not be
  run.
- Do not weaken, skip, or delete a failing test solely to make a change pass.

## Data, Models, Secrets, and Generated Files

- Never commit secrets, credentials, API keys, local environment files, or
  personally identifying recordings.
- Do not commit generated datasets, raw webcam/video recordings, checkpoints,
  exported models, benchmark dumps, or other large artifacts without explicit
  maintainer approval.
- Store only small, license-compatible test fixtures when they are necessary,
  documented, and intentionally reviewed for inclusion.
- Document how external data was obtained, its license or consent status, and
  the preprocessing applied before using it in reported experiments.
- Load paths and secrets from documented configuration or environment
  variables when such configuration becomes necessary.

## Git and Collaboration Workflow

- Never work directly on `main`. Use a focused branch for each approved unit
  of work.
- Keep changes scoped to the requested milestone and avoid mixing unrelated
  cleanup with feature work.
- Use meaningful Conventional Commit messages such as `feat:`, `fix:`,
  `test:`, `docs:`, `refactor:`, or `chore:`.
- Do not commit, push, merge, tag, publish a release, or open a pull request
  unless the maintainer has requested that action.
- Pull requests should explain the motivation and design, list validation
  performed, identify limitations, and include measured results only when
  measurements were actually run.
- Preserve other contributors' work. Do not rewrite shared history or discard
  unrelated local changes.

## Documentation Expectations

- Keep setup and usage steps reproducible for a new contributor.
- Document public data fields, units, coordinate conventions, temporal-window
  semantics, class mappings, model input/output shapes, and threshold choices
  when those contracts are introduced.
- Distinguish current behavior from planned work. Do not document planned
  functionality as already implemented.
- Include the prototype safety limitation in user-facing demo and release
  documentation.

## Before Finishing Any Task

- Confirm the work is within the explicitly approved day or milestone.
- Check ownership boundaries and compatibility with both intended pipelines.
- Run relevant automated checks and any approved manual check.
- Review the diff for secrets, generated data, large binaries, accidental
  artifacts, fabricated claims, and unrelated changes.
- Summarize changed files, validation performed, known limitations, and the
  next step without starting that next step.
