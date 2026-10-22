"""Reproducible evaluation and head-to-head model comparison for Vigil M2."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vigil.ml.baseline.classifier import RuleBasedBaseline
from vigil.ml.baseline.evaluation import (
    BaselineEvaluationReport,
    compute_metrics,
    evaluate_baseline,
)
from vigil.ml.data.schema import SampleRecord
from vigil.ml.dataset.dataset import FatigueWindowDataset
from vigil.ml.dataset.normalizer import FeatureNormalizer
from vigil.ml.dataset.windowing import extract_windows
from vigil.ml.evaluation.input_audit import (
    EvaluationInputAudit,
    audit_evaluation_input,
    validate_audit_metadata,
)
from vigil.ml.models.export import LoadedModelBundle, load_model_bundle
from vigil.ml.models.gru import GRUTemporalClassifier
from vigil.ml.provenance import (
    validate_evaluation_group_separation,
    validate_training_provenance,
)

try:
    import torch
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    HAS_TORCH = False


@dataclass(frozen=True)
class IndependentEvaluationMetadata:
    """Metadata describing direct evaluation of an independent record set."""

    development_strategy: str
    evaluation_subjects: list[str]
    evaluation_sessions: list[str]
    secondary_split_performed: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert independent-evaluation metadata to a dictionary."""
        return {
            "development_strategy": self.development_strategy,
            "evaluation_subjects": self.evaluation_subjects,
            "evaluation_sessions": self.evaluation_sessions,
            "secondary_split_performed": self.secondary_split_performed,
        }


@dataclass(frozen=True)
class ModelComparisonReport:
    """Comparative evaluation report contrasting the learned GRU model and D3 rule baseline.

    Attributes:
        gru_report: Evaluation metrics of the learned GRU sequence classifier on held-out windows.
        baseline_matched_report: Evaluation metrics of the D3 rule baseline on the identical window endpoint samples.
        baseline_full_report: Frame-by-frame evaluation metrics of the D3 rule baseline across all non-ambiguous samples.
        evaluation_metadata: Metadata distinguishing model development from direct evaluation.
        window_count: Number of evaluation windows evaluated.
        is_synthetic: Whether the evaluation was executed on synthetic test fixtures.
    """

    gru_report: BaselineEvaluationReport
    baseline_matched_report: BaselineEvaluationReport
    baseline_full_report: BaselineEvaluationReport
    evaluation_metadata: IndependentEvaluationMetadata
    window_count: int
    is_synthetic: bool | None = None
    input_audit: EvaluationInputAudit | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the comparative report into a nested dictionary."""
        result = {
            "is_synthetic": self.is_synthetic if self.is_synthetic is not None else "unknown",
            "window_count": self.window_count,
            "gru_model": self.gru_report.to_dict(),
            "rule_baseline_matched": self.baseline_matched_report.to_dict(),
            "rule_baseline_full": self.baseline_full_report.to_dict(),
            "evaluation_metadata": self.evaluation_metadata.to_dict(),
        }
        if self.input_audit is not None:
            result["input_audit"] = self.input_audit.to_dict()
        return result

    def to_markdown(self) -> str:
        """Render a formatted markdown comparison table."""
        lines = [
            "### M2 Model Comparison Report: Learned GRU vs. D3 Rule Baseline",
            "",
        ]

        if self.is_synthetic is True:
            lines.extend([
                "> [!IMPORTANT]",
                (
                    "> **Disclaimer on Synthetic Fixtures**: This evaluation was executed using synthetic "
                    "unit-test fixtures because no real labeled driver recordings are present in the repository. "
                    "These metrics validate code mechanics, tensor contracts, and metric pipelines; they "
                    "do NOT establish real-world driver drowsiness detection accuracy."
                ),
                "",
            ])

        lines.extend([
            f"- **Model-Development Split Strategy**: `{self.evaluation_metadata.development_strategy}`",
            f"- **Independent Evaluation Subjects**: {self.evaluation_metadata.evaluation_subjects}",
            f"- **Independent Evaluation Sessions**: {self.evaluation_metadata.evaluation_sessions}",
            "- **Secondary Train/Validation Split**: No",
            f"- **Evaluated Windows ($W=60$)**: {self.window_count}",
            f"- **Synthetic Fixtures**: {str(self.is_synthetic).lower() if self.is_synthetic is not None else 'unknown'}",
            "",
            "#### Head-to-Head Performance Comparison",
            "| Metric | Learned GRU (Windows) | Rule Baseline (Matched Timesteps) | Rule Baseline (All Frames) |",
            "|---|---|---|---|",
            f"| **Overall Accuracy** | {self.gru_report.accuracy:.4f} | {self.baseline_matched_report.accuracy:.4f} | {self.baseline_full_report.accuracy:.4f} |",
            f"| **Macro Precision** | {self.gru_report.macro_precision:.4f} | {self.baseline_matched_report.macro_precision:.4f} | {self.baseline_full_report.macro_precision:.4f} |",
            f"| **Macro Recall** | {self.gru_report.macro_recall:.4f} | {self.baseline_matched_report.macro_recall:.4f} | {self.baseline_full_report.macro_recall:.4f} |",
            f"| **Macro F1 Score** | {self.gru_report.macro_f1:.4f} | {self.baseline_matched_report.macro_f1:.4f} | {self.baseline_full_report.macro_f1:.4f} |",
            f"| **Prediction Coverage** | {self.gru_report.coverage:.4f} ({self.gru_report.evaluated_samples}/{self.gru_report.eligible_samples}) | {self.baseline_matched_report.coverage:.4f} ({self.baseline_matched_report.evaluated_samples}/{self.baseline_matched_report.eligible_samples}) | {self.baseline_full_report.coverage:.4f} ({self.baseline_full_report.evaluated_samples}/{self.baseline_full_report.eligible_samples}) |",
            f"| **Skipped Predictions** | {self.gru_report.skipped_predictions} | {self.baseline_matched_report.skipped_predictions} | {self.baseline_full_report.skipped_predictions} |",
            "",
            "#### GRU Confusion Matrix (Canonical Order: 0: ACTIVE, 1: DROWSY, 2: SLEEPING)",
            "| True \\ Pred | ACTIVE (0) | DROWSY (1) | SLEEPING (2) |",
            "|---|---|---|---|",
            f"| **ACTIVE (0)** | {self.gru_report.confusion_matrix[0][0]} | {self.gru_report.confusion_matrix[0][1]} | {self.gru_report.confusion_matrix[0][2]} |",
            f"| **DROWSY (1)** | {self.gru_report.confusion_matrix[1][0]} | {self.gru_report.confusion_matrix[1][1]} | {self.gru_report.confusion_matrix[1][2]} |",
            f"| **SLEEPING (2)** | {self.gru_report.confusion_matrix[2][0]} | {self.gru_report.confusion_matrix[2][1]} | {self.gru_report.confusion_matrix[2][2]} |",
            "",
            "#### Rule Baseline (Matched) Confusion Matrix",
            "| True \\ Pred | ACTIVE (0) | DROWSY (1) | SLEEPING (2) |",
            "|---|---|---|---|",
            f"| **ACTIVE (0)** | {self.baseline_matched_report.confusion_matrix[0][0]} | {self.baseline_matched_report.confusion_matrix[0][1]} | {self.baseline_matched_report.confusion_matrix[0][2]} |",
            f"| **DROWSY (1)** | {self.baseline_matched_report.confusion_matrix[1][0]} | {self.baseline_matched_report.confusion_matrix[1][1]} | {self.baseline_matched_report.confusion_matrix[1][2]} |",
            f"| **SLEEPING (2)** | {self.baseline_matched_report.confusion_matrix[2][0]} | {self.baseline_matched_report.confusion_matrix[2][1]} | {self.baseline_matched_report.confusion_matrix[2][2]} |",
        ])

        if self.input_audit is not None:
            lines.extend(["", self.input_audit.to_markdown()])

        return "\n".join(lines)


def compare_models(
    records: list[SampleRecord],
    model: GRUTemporalClassifier,
    normalizer: FeatureNormalizer,
    baseline_classifier: RuleBasedBaseline | None = None,
    train_ratio: float = 0.75,
    seed: int = 42,
    is_synthetic: bool | None = None,
    training_provenance: dict[str, Any] | None = None,
    recording_ids: set[str] | None = None,
    dataset_name: str | None = None,
    dataset_release: str | None = None,
    manifest_id: str | None = None,
    manifest_exclusions: int | None = None,
) -> ModelComparisonReport:
    """Compare a learned GRU and D3 rule baseline on an independent evaluation set.

    Data Hygiene:
        - Verify explicit model-development provenance before windowing.
        - Evaluate all eligible windows from the supplied independent records.
        - Evaluate the rule baseline both on all eligible frames and matched to exact window endpoints.

    Args:
        records: List of SampleRecords to partition and evaluate.
        model: Trained GRUTemporalClassifier instance.
        normalizer: Fitted FeatureNormalizer instance.
        baseline_classifier: Optional RuleBasedBaseline (default uses standard thresholds).
        train_ratio: Retained for API compatibility; independent records are not split.
        seed: Retained for API compatibility; independent records are not split.
        is_synthetic: Flag indicating whether records are synthetic test fixtures.
        training_provenance: Group identities used to train the supplied model.

    Returns:
        ModelComparisonReport detailing performance metrics and confusion matrices.

    Raises:
        ImportError: If PyTorch is unavailable.
        ValueError: If records are empty or partitions lack usable evaluation windows.
    """
    validate_audit_metadata(
        dataset_name=dataset_name,
        dataset_release=dataset_release,
        manifest_id=manifest_id,
        manifest_exclusions=manifest_exclusions,
    )
    if is_synthetic is not None and not isinstance(is_synthetic, bool):
        raise TypeError("is_synthetic must be a boolean or None.")
    if not HAS_TORCH or torch is None:
        raise ImportError("PyTorch is required for model comparison. Install it with 'pip install torch'.")

    if not records:
        raise ValueError("No sample records provided for model comparison.")

    if not normalizer.is_fitted:
        raise ValueError("FeatureNormalizer must be fitted prior to model comparison.")

    provenance = training_provenance
    if provenance is None:
        provenance = getattr(model, "training_provenance", None)
    if provenance is None:
        raise TypeError(
            "Training provenance is required for independent evaluation. "
            "Pass training_provenance or evaluate a model produced by train_temporal_classifier."
        )
    validated_provenance = validate_training_provenance(provenance)
    validate_evaluation_group_separation(records, validated_provenance)

    if baseline_classifier is None:
        baseline_classifier = RuleBasedBaseline()

    # Extract every eligible window from the supplied independent dataset.
    evaluation_windows = extract_windows(records, normalizer=normalizer)
    if not evaluation_windows:
        raise ValueError(
            "Independent evaluation records yielded 0 usable windows (segments may be shorter than W=60 "
            "or overlap ambiguous annotations)."
        )

    # Evaluate the model on every independent evaluation window.
    model.eval()
    evaluation_dataset = FatigueWindowDataset(evaluation_windows)

    gru_y_true: list[int] = []
    gru_y_pred: list[int | None] = []

    with torch.no_grad():
        for i in range(len(evaluation_dataset)):
            features, target = evaluation_dataset[i]
            batch_x = features.unsqueeze(0)  # (1, 60, 8)
            logits = model(batch_x)
            pred_id = int(torch.argmax(logits, dim=-1).item())
            gru_y_true.append(int(target.item()))
            gru_y_pred.append(pred_id)

    gru_report = compute_metrics(
        y_true=gru_y_true,
        y_pred=gru_y_pred,
        total_samples=len(gru_y_true),
        ambiguous_count=0,
    )

    # Evaluate D3 on the same window endpoints.
    rec_index: dict[tuple[str, int], SampleRecord] = {
        (r.session_id, r.frame_index): r for r in records
    }

    matched_base_true: list[int] = []
    matched_base_pred: list[int | None] = []

    for w in evaluation_windows:
        endpoint_key = (w.session_id, w.end_frame_index)
        sample = rec_index.get(endpoint_key)
        if sample is None:
            continue

        matched_base_true.append(w.target)
        rule_res = baseline_classifier.classify(sample)
        matched_base_pred.append(rule_res.label_id if rule_res.has_prediction else None)

    baseline_matched_report = compute_metrics(
        y_true=matched_base_true,
        y_pred=matched_base_pred,
        total_samples=len(matched_base_true),
        ambiguous_count=0,
    )

    baseline_full_report = evaluate_baseline(records, baseline_classifier)
    evaluation_metadata = IndependentEvaluationMetadata(
        development_strategy=validated_provenance["strategy"],
        evaluation_subjects=sorted({record.subject_id for record in records}),
        evaluation_sessions=sorted({record.session_id for record in records}),
    )
    input_audit = audit_evaluation_input(
        records,
        is_synthetic=is_synthetic,
        recording_ids=recording_ids,
        dataset_name=dataset_name,
        dataset_release=dataset_release,
        manifest_id=manifest_id,
        manifest_exclusions=manifest_exclusions,
    )

    return ModelComparisonReport(
        gru_report=gru_report,
        baseline_matched_report=baseline_matched_report,
        baseline_full_report=baseline_full_report,
        evaluation_metadata=evaluation_metadata,
        window_count=len(evaluation_windows),
        is_synthetic=is_synthetic,
        input_audit=input_audit,
    )


def evaluate_recording_directory(
    directory_path: str | Path,
    bundle_path: str | Path | LoadedModelBundle | None = None,
    *,
    dataset_name: str | None = None,
    dataset_release: str | None = None,
    manifest_id: str | None = None,
    manifest_exclusions: int | None = None,
) -> ModelComparisonReport | None:
    """Evaluate and compare models on recordings located in a directory.

    If the directory does not exist or contains no `.jsonl` session files, this function
    returns None and reports the absence of real recording data cleanly without fabricating results.

    Args:
        directory_path: Directory containing recorded session `.jsonl` files.
        bundle_path: Optional model bundle or path to bundle.

    Returns:
        ModelComparisonReport if valid sessions exist, else None.
    """
    validate_audit_metadata(
        dataset_name=dataset_name,
        dataset_release=dataset_release,
        manifest_id=manifest_id,
        manifest_exclusions=manifest_exclusions,
    )
    dir_path = Path(directory_path)
    if not dir_path.exists():
        return None

    # Collect root-level and nested recordings once, in deterministic order.
    sample_files = sorted(
        set(dir_path.glob("*.jsonl")) | set(dir_path.glob("**/samples.jsonl")),
        key=lambda path: path.as_posix(),
    )
    if not sample_files:
        return None

    records: list[SampleRecord] = []
    for f in sample_files:
        with open(f, encoding="utf-8") as fh:
            for line_number, line in enumerate(fh, start=1):
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    data = json.loads(line_str)
                    records.append(SampleRecord.from_dict(data))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in recording file '{f}' at line {line_number}: {exc}"
                    ) from exc
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid SampleRecord in recording file '{f}' at line "
                        f"{line_number}: {exc}"
                    ) from exc

    if not records:
        return None

    if bundle_path is None:
        raise ValueError("A trained model bundle is required to evaluate recordings.")

    if isinstance(bundle_path, LoadedModelBundle):
        bundle = bundle_path
    else:
        bundle = load_model_bundle(bundle_path)

    return compare_models(
        records=records,
        model=bundle.model,
        normalizer=bundle.normalizer,
        training_provenance=bundle.metadata.get("training_provenance"),
        is_synthetic=False,
        recording_ids={path.as_posix() for path in sample_files},
        dataset_name=dataset_name,
        dataset_release=dataset_release,
        manifest_id=manifest_id,
        manifest_exclusions=manifest_exclusions,
    )
