# M2 D2 Compact Temporal Model Candidates

Day 2 adds compact causal temporal candidates without changing the existing
GRU baseline or the runtime data contract.

## Candidates

- `GRUTemporalClassifier` remains the mandatory baseline.
- `LSTMTemporalClassifier` is a single-direction LSTM candidate.
- `TCNTemporalClassifier` is a dilated, left-padded 1-D convolutional
  candidate. Left padding preserves causality: each timestep depends only on
  the current and preceding observations.

All candidates accept `(batch, 60, 8)` `float32` windows and emit raw
`(batch, 3)` logits in canonical order `ACTIVE=0`, `DROWSY=1`, and
`SLEEPING=2`. `AMBIGUOUS=-1` remains an annotation status and is not a model
prediction. `predict_proba()` and `predict()` provide the same convenience
interface across candidates.

The implementation is limited to architecture and contract tests. No model
sweep, hyperparameter ranking, checkpoint selection, calibration, or
real-world performance claim is made. Authorized labelled development data and
frozen subject/session groups remain prerequisites for meaningful comparison.
