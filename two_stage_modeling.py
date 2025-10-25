"""
Two-stage modeling experiment to improve playoff round predictions.

Stage 1: Logistic regression on team net rating only.
Stage 2: Multinomial model on full topology feature set + Stage 1 outputs.

Run:
    python two_stage_modeling.py
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from poster_figures import FEATURE_LABELS, load_datasets, _prepare_error_dataframe


def _full_feature_columns(df: pd.DataFrame) -> List[str]:
    exclude = {"season", "team", "playoff_round", "pred_round", "prob_ge2"}
    return [c for c in df.columns if c not in exclude]


def evaluate_two_stage_model() -> Tuple[pd.DataFrame, pd.DataFrame]:
    data = load_datasets()
    df = data.df.dropna(subset=["team_net_rating", "playoff_round"]).reset_index(drop=True)
    seasons = sorted(df["season"].unique())

    feature_cols = _full_feature_columns(df)

    stage1_records = []
    stage2_records = []
    stage1_preds = []
    stage2_preds = []

    for holdout in seasons:
        train = df[df["season"] != holdout].dropna(subset=feature_cols).reset_index(drop=True)
        test = df[df["season"] == holdout].dropna(subset=feature_cols).reset_index(drop=True)
        if test.empty or train.empty:
            continue

        # Stage 1: Net rating only
        stage1 = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=1.2,
                        penalty="l2",
                        solver="lbfgs",
                        multi_class="multinomial",
                        max_iter=5000,
                        random_state=42,
                    ),
                ),
            ]
        )
        stage1.fit(train[["team_net_rating"]], train["playoff_round"])
        train_base_proba = stage1.predict_proba(train[["team_net_rating"]])
        test_base_proba = stage1.predict_proba(test[["team_net_rating"]])
        train_base_pred = stage1.predict(train[["team_net_rating"]])
        test_base_pred = stage1.predict(test[["team_net_rating"]])

        classes = stage1.named_steps["clf"].classes_

        def augment(df_subset: pd.DataFrame, base_pred: np.ndarray, base_proba: np.ndarray) -> pd.DataFrame:
            df_aug = df_subset.copy()
            for idx, cls in enumerate(classes):
                df_aug[f"baseline_prob_{cls}"] = base_proba[:, idx]
            df_aug["baseline_pred_round"] = base_pred
            return df_aug

        train_aug = augment(train, train_base_pred, train_base_proba)
        test_aug = augment(test, test_base_pred, test_base_proba)
        aug_features = feature_cols + [col for col in train_aug.columns if col.startswith("baseline_prob_")] + [
            "baseline_pred_round"
        ]

        # Stage 2: Full topology + baseline outputs
        from sklearn.ensemble import HistGradientBoostingClassifier

        stage2 = HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_depth=None,
            max_iter=400,
            min_samples_leaf=8,
            random_state=42,
        )
        stage2.fit(train_aug[aug_features], train_aug["playoff_round"])

        # Predictions
        pred1 = stage1.predict(test[["team_net_rating"]])
        pred2 = stage2.predict(test_aug[aug_features])

        stage1_preds.append(
            pd.DataFrame(
                {
                    "season": test["season"],
                    "team": test["team"],
                    "playoff_round": test["playoff_round"],
                    "pred_round": pred1,
                }
            )
        )

        stage2_preds.append(
            pd.DataFrame(
                {
                    "season": test["season"],
                    "team": test["team"],
                    "playoff_round": test["playoff_round"],
                    "pred_round": pred2,
                }
            )
        )

        stage1_records.append(
            {
                "season": holdout,
                "accuracy": accuracy_score(test["playoff_round"], pred1),
                "macro_f1": f1_score(test["playoff_round"], pred1, average="macro"),
            }
        )
        stage2_records.append(
            {
                "season": holdout,
                "accuracy": accuracy_score(test["playoff_round"], pred2),
                "macro_f1": f1_score(test["playoff_round"], pred2, average="macro"),
            }
        )

    stage1_df = pd.DataFrame(stage1_records)
    stage2_df = pd.DataFrame(stage2_records)

    preds_stage1 = pd.concat(stage1_preds, ignore_index=True)
    preds_stage2 = pd.concat(stage2_preds, ignore_index=True)

    # Compute average absolute error
    actual = data.df[["season", "team", "playoff_round"]]
    merged_stage1 = actual.merge(preds_stage1, on=["season", "team", "playoff_round"], how="inner")
    merged_stage1["error"] = merged_stage1["pred_round"] - merged_stage1["playoff_round"]
    merged_stage2 = actual.merge(preds_stage2, on=["season", "team", "playoff_round"], how="inner")
    merged_stage2["error"] = merged_stage2["pred_round"] - merged_stage2["playoff_round"]
    stage1_abs = merged_stage1["error"].abs().mean()
    stage2_abs = merged_stage2["error"].abs().mean()

    print("Stage1 (net rating) CV metrics:")
    print(stage1_df[['accuracy', 'macro_f1']].mean())
    print(f"avg |error| = {stage1_abs:.3f}")
    print("\nStage2 (two-stage) CV metrics:")
    print(stage2_df[['accuracy', 'macro_f1']].mean())
    print(f"avg |error| = {stage2_abs:.3f}")

    # Save predictions
    preds_stage2.to_csv(Path("poster_figures/two_stage_predictions.csv"), index=False)

    return stage1_df, stage2_df


if __name__ == "__main__":
    evaluate_two_stage_model()
