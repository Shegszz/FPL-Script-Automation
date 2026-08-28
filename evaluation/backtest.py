"""
FPL Model Backtesting Framework

This module evaluates forecasting models against historical
Gameweek results and compares them with simple baselines.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from evaluation.baselines import BASELINES
from evaluation.metrics import evaluate_predictions


RESULTS_FILE = Path("evaluation/results.json")


REQUIRED_COLUMNS = {
    "target",
    "pts_lag1",
    "form_3gw",
    "form_5gw",
    "exp_form",
}


def validate_dataset(df: pd.DataFrame) -> None:
    """Validate that the dataset contains the fields required by baselines."""

    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(sorted(missing))
        )


def evaluate_baselines(df: pd.DataFrame) -> dict:
    """Evaluate all configured baseline models."""

    validate_dataset(df)

    evaluation_df = df.dropna(
        subset=list(REQUIRED_COLUMNS)
    ).copy()

    if evaluation_df.empty:
        raise ValueError(
            "No valid rows are available for baseline evaluation."
        )

    y_true = evaluation_df["target"]

    results = {}

    for name, predictor in BASELINES.items():
        predictions = predictor(evaluation_df)

        results[name] = evaluate_predictions(
            y_true,
            predictions,
        )

    return results


def save_results(results: dict) -> None:
    """Save evaluation results as JSON."""

    RESULTS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with RESULTS_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
        )


def run_backtest(df: pd.DataFrame) -> dict:
    """Run the complete baseline evaluation."""

    results = {
        "dataset": {
            "rows": len(df),
        },
        "baselines": evaluate_baselines(df),
    }

    save_results(results)

    return results


def main() -> None:
    """
    Entry point.

    Dataset integration will be connected after we inspect
    the existing FPL model training pipeline.
    """

    print(
        "Backtesting framework installed successfully."
    )

    print(
        "Dataset integration will be added in the next step."
    )


if __name__ == "__main__":
    main()
