# M2 D2 Compact Temporal Model Candidates

Day 2 adds compact causal temporal candidates without changing the existing
GRU baseline or the runtime data contract.

## Candidates

- `GRUTemporalClassifier` remains the mandatory baseline.
- `LSTMTemporalClassifier` is a single-direction LSTM candidate.
- `TCNTemporalClassifier` is a dilated, left-padded 1-D convolutional
  candidate with default kernel size 3 and dilations `(1, 2, 4, 8, 16)`.
  Its theoretical receptive field is 63 frames, covering the complete
  60-frame window. Left padding preserves causality: each timestep depends
  only on the current and preceding observations.

All candidates accept `(batch, 60, 8)` `float32` windows and emit raw
`(batch, 3)` logits in canonical order `ACTIVE=0`, `DROWSY=1`, and
`SLEEPING=2`. `AMBIGUOUS=-1` remains an annotation status and is not a model
prediction. `predict_proba()` and `predict()` provide the same convenience
interface across candidates.

## Shared training path

`train_temporal_classifier(..., model_type="gru")` preserves the existing GRU
default. The explicit values `"gru"`, `"lstm"`, and `"tcn"` select the
corresponding candidate through one shared pipeline: grouped splitting,
training-only normalization, locked window extraction, deterministic seeded
batches, cross-entropy optimization, validation-loss checkpoint selection, and
training provenance. Candidate checkpoints use
`best_<model_type>_checkpoint.pt`; the default GRU filename remains
`best_gru_checkpoint.pt`.

The current bundle exporter and live inference loader remain GRU-specific.
LSTM and TCN training mechanics are verified here, but their checkpoints are
not claimed to be exportable or deployable through that existing runtime path.

The implementation is limited to architecture and contract tests. No model
sweep, hyperparameter ranking, checkpoint selection, calibration, or
real-world performance claim is made. Authorized labelled development data and
frozen subject/session groups remain prerequisites for meaningful comparison.
