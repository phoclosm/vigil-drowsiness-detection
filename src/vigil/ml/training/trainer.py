"""Reproducible training and validation pipeline for Vigil M2 temporal sequence classifiers."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any

from vigil.ml.baseline.evaluation import BaselineEvaluationReport, compute_metrics
from vigil.ml.data.schema import SampleRecord
from vigil.ml.dataset.dataset import FatigueWindowDataset
from vigil.ml.dataset.normalizer import FeatureNormalizer
from vigil.ml.dataset.splitter import DatasetGroupSplitResult, split_records_by_group
from vigil.ml.models.gru import GRUConfig, GRUTemporalClassifier
from vigil.ml.training.config import TrainingConfig

try:
    import torch
    from torch.utils.data import DataLoader
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    DataLoader = object  # type: ignore[misc,assignment]
    HAS_TORCH = False


def seed_everything(seed: int) -> None:
    """Set deterministic seeds for Python, NumPy, and PyTorch.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass

    if HAS_TORCH and torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover
            torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class TrainingResult:
    """Outcome and artifacts of a temporal classifier training run.

    Attributes:
        model: Trained GRUTemporalClassifier loaded with best validation weights.
        best_epoch: 1-indexed epoch number that achieved the lowest validation loss.
        best_val_loss: Lowest validation cross-entropy loss recorded.
        train_losses: Epoch-by-epoch training loss trajectory.
        val_losses: Epoch-by-epoch validation loss trajectory.
        best_checkpoint_state: State dictionary containing the best model weights.
        evaluation_report: Canonical evaluation report on the held-out validation set.
        split_result: Metadata describing the subject/session partition.
        normalizer: Fitted FeatureNormalizer used for feature standardization.
        checkpoint_path: Path to disk checkpoint if output_dir was specified, else None.
    """

    model: GRUTemporalClassifier
    best_epoch: int
    best_val_loss: float
    train_losses: list[float]
    val_losses: list[float]
    best_checkpoint_state: dict[str, Any]
    evaluation_report: BaselineEvaluationReport
    split_result: DatasetGroupSplitResult
    normalizer: FeatureNormalizer
    checkpoint_path: str | None = None


def train_temporal_classifier(
    records: list[SampleRecord],
    config: TrainingConfig | None = None,
) -> TrainingResult:
    """Execute reproducible temporal classifier training from canonical SampleRecord data.

    Pipeline Integrity:
        1. Split records using D4 subject-first/session-fallback grouped splitting before windowing.
        2. Fit FeatureNormalizer strictly on training partition records.
        3. Extract training and validation windows separately conforming to D4 contract.
        4. Deterministically seed Python, NumPy, PyTorch, and training DataLoader.
        5. Shuffle training batches only; validation order remains deterministic.
        6. Optimize cross-entropy loss with Adam optimizer.
        7. Record epoch losses and select the best checkpoint by validation loss.
        8. Retain checkpoint in memory or designated output_dir (never commit weights).
        9. Reject runs where either partition lacks usable windows.

    Args:
        records: List of SampleRecord instances.
        config: TrainingConfig specifying hyperparameters. Default uses sensible defaults.

    Returns:
        TrainingResult with the fitted model, loss histories, best checkpoint, and evaluation report.

    Raises:
        ImportError: If PyTorch is unavailable.
        ValueError: If partitions or windows are insufficient for training and validation.
    """
    if not HAS_TORCH or torch is None:
        raise ImportError(
            "PyTorch is required for temporal classifier training. "
            "Install PyTorch with 'pip install torch'."
        )

    if config is None:
        config = TrainingConfig()

    if not records:
        raise ValueError("No sample records provided for training.")

    # 1. Deterministic seeding
    seed_everything(config.seed)

    # 2. Split records by group (subject-first, session-fallback)
    train_records, val_records, split_result = split_records_by_group(
        records,
        train_ratio=config.train_ratio,
        seed=config.seed,
    )

    if not train_records or not val_records:
        raise ValueError(
            f"Grouped splitting yielded insufficient partitions: "
            f"train_records={len(train_records)}, val_records={len(val_records)}. "
            f"Note: {split_result.limitation_note}"
        )

    # 3. Fit FeatureNormalizer exclusively on training records
    normalizer = FeatureNormalizer().fit(train_records)

    # 4. Extract windows separately for each partition
    train_dataset = FatigueWindowDataset.from_records(train_records, normalizer=normalizer)
    val_dataset = FatigueWindowDataset.from_records(val_records, normalizer=normalizer)

    if len(train_dataset) == 0:
        raise ValueError(
            "Training partition yielded 0 usable windows. Segments may be shorter than W=60 "
            "or overlap ambiguous annotations. Cannot train model."
        )

    if len(val_dataset) == 0:
        raise ValueError(
            "Validation partition yielded 0 usable windows. An independent held-out evaluation "
            "partition is required."
        )

    # 5. DataLoaders (seeded shuffle for train, deterministic for val)
    train_gen = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=train_gen,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )

    # 6. Instantiate model, loss, and optimizer
    model_config = GRUConfig(hidden_dim=config.hidden_dim)
    model = GRUTemporalClassifier(model_config)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val_loss = float("inf")
    best_epoch = 1
    best_state_dict: dict[str, Any] = {}

    # 7. Training loop
    for epoch in range(1, config.max_epochs + 1):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * len(y_batch)

        avg_train_loss = epoch_train_loss / len(train_dataset)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                val_logits = model(x_val)
                val_loss = criterion(val_logits, y_val)
                epoch_val_loss += val_loss.item() * len(y_val)

        avg_val_loss = epoch_val_loss / len(val_dataset)
        val_losses.append(avg_val_loss)

        # 8. Checkpoint preservation (best validation loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            best_state_dict = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    # Load best checkpoint weights into model
    model.load_state_dict(best_state_dict)
    model.eval()

    checkpoint_path: str | None = None
    if config.output_dir is not None:
        os.makedirs(config.output_dir, exist_ok=True)
        checkpoint_path = os.path.join(config.output_dir, "best_gru_checkpoint.pt")
        torch.save(best_state_dict, checkpoint_path)

    # 9. Compute validation metrics using canonical evaluation helper
    y_true: list[int] = [w.target for w in val_dataset.windows]
    y_pred: list[int] = []

    with torch.no_grad():
        for x_val, _ in val_loader:
            preds = model.predict(x_val).tolist()
            y_pred.extend(preds)

    eval_report = compute_metrics(
        y_true=y_true,
        y_pred=y_pred,  # type: ignore[arg-type]
        total_samples=len(y_true),
        ambiguous_count=0,
    )

    return TrainingResult(
        model=model,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        train_losses=train_losses,
        val_losses=val_losses,
        best_checkpoint_state=best_state_dict,
        evaluation_report=eval_report,
        split_result=split_result,
        normalizer=normalizer,
        checkpoint_path=checkpoint_path,
    )
