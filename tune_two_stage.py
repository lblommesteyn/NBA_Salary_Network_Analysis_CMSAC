"""
Hyperparameter tuning for two-stage playoff prediction.

Stage 1: logistic regression on net rating.
Stage 2: HistGradientBoostingClassifier on topology + stage1 outputs.

Searches multiple parameter combos and blend weights, reporting
accuracy, macro-F1, and average absolute error.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier

from poster_figures import load_datasets


PARAM_GRID = [
    {"learning_rate": 0.05, "max_depth": None, "max_iter": 400, "min_samples_leaf": 6, "l2_regularization": 0.0},
    {"learning_rate": 0.07, "max_depth": None, "max_iter": 500, "min_samples_leaf": 8, "l2_regularization": 0.0},
    {"learning_rate": 0.10, "max_depth": 6, "max_iter": 400, "min_samples_leaf": 6, "l2_regularization": 0.0},
    {"learning_rate": 0.08, "max_depth": None, "max_iter": 600, "min_samples_leaf": 4, "l2_regularization": 0.01},
    {"learning_rate": 0.06, "max_depth": 8, "max_iter": 600, "min_samples_leaf": 6, "l2_regularization": 0.0},
]

BLEND_WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]


def _full_feature_columns(df: pd.DataFrame) -> List[str]:
    exclude = {"season", "team", "playoff_round", "pred_round", "prob_ge2"}
    return [c for c in df.columns if c not in exclude]


def evaluate_params() -> pd.DataFrame:
    data = load_datasets()
    df = data.df.dropna(subset=["team_net_rating", "playoff_round"]).reset_index(drop=True)
    seasons = sorted(df["season"].unique())
    feature_cols = _full_feature_columns(df)

    results: List[Dict[str, float]] = []

    for param_idx, params in enumerate(PARAM_GRID, start=1):
        stage1_records = []
        stage2_records: Dict[str, List[Dict[str, float]]] = {
            "classifier": [],
            "blend": {w: [] for w in BLEND_WEIGHTS},
        }
        stage1_preds = []
        stage2_preds: Dict[str, List[pd.DataFrame]] = {"classifier": [], "blend": {w: [] for w in BLEND_WEIGHTS}}

        for holdout in seasons:
            train = df[df["season"] != holdout].dropna(subset=feature_cols).reset_index(drop=True)
            test = df[df["season"] == holdout].dropna(subset=feature_cols).reset_index(drop=True)
            if train.empty or test.empty:
                continue

            # Stage 1 logistic on net rating
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
            train_proba = stage1.predict_proba(train[["team_net_rating"]])
            test_proba = stage1.predict_proba(test[["team_net_rating"]])
            train_pred = stage1.predict(train[["team_net_rating"]])
            test_pred = stage1.predict(test[["team_net_rating"]])
            classes = stage1.named_steps["clf"].classes_

            def augment(df_subset: pd.DataFrame, base_pred: np.ndarray, base_proba: np.ndarray) -> pd.DataFrame:
                df_aug = df_subset.copy()
                for idx_cls, cls in enumerate(classes):
                    df_aug[f"baseline_prob_{cls}"] = base_proba[:, idx_cls]
                df_aug["baseline_pred_round"] = base_pred
                return df_aug

            train_aug = augment(train, train_pred, train_proba)
            test_aug = augment(test, test_pred, test_proba)
            aug_features = feature_cols + [col for col in train_aug.columns if col.startswith("baseline_prob_")] + [
                "baseline_pred_round"
            ]

            # Stage 2 classifier with params
            stage2 = HistGradientBoostingClassifier(
                learning_rate=params["learning_rate"],
                max_depth=params["max_depth"],
                max_iter=params["max_iter"],
                min_samples_leaf=params["min_samples_leaf"],
                l2_regularization=params["l2_regularization"],
                random_state=42,
            )
            stage2.fit(train_aug[aug_features], train_aug["playoff_round"])

            pred1 = stage1.predict(test[["team_net_rating"]])
            pred2 = stage2.predict(test_aug[aug_features])
            proba2 = stage2.predict_proba(test_aug[aug_features])

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

            stage2_preds["classifier"].append(
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

            stage2_records["classifier"].append(
                {
                    "season": holdout,
                    "accuracy": accuracy_score(test["playoff_round"], pred2),
                    "macro_f1": f1_score(test["playoff_round"], pred2, average="macro"),
                }
            )

            stage1_exp = np.sum(test_proba * classes, axis=1)
            stage2_exp = np.sum(proba2 * stage2.classes_, axis=1)

            for blend_weight in BLEND_WEIGHTS:
                combined = np.clip(
                    np.rint((1 - blend_weight) * stage1_exp + blend_weight * stage2_exp),
                    0,
                    4,
                ).astype(int)
                stage2_preds["blend"][blend_weight].append(
                    pd.DataFrame(
                        {
                            "season": test["season"],
                            "team": test["team"],
                            "playoff_round": test["playoff_round"],
                            "pred_round": combined,
                        }
                    )
                )
                stage2_records["blend"][blend_weight].append(
                    {
                        "season": holdout,
                        "accuracy": accuracy_score(test["playoff_round"], combined),
                        "macro_f1": f1_score(test["playoff_round"], combined, average="macro"),
                    }
                )

        def summarize(pred_list: List[pd.DataFrame], record_list: List[Dict[str, float]]) -> Tuple[float, float, float]:
            merged = data.df[["season", "team", "playoff_round"]].merge(
                pd.concat(pred_list, ignore_index=True),
                on=["season", "team", "playoff_round"],
                how="inner",
            )
            merged["error"] = merged["pred_round"] - merged["playoff_round"]
            abs_err = merged["error"].abs().mean()
            records_df = pd.DataFrame(record_list)
            accuracy = records_df["accuracy"].mean()
            macro = records_df["macro_f1"].mean()
            return accuracy, macro, abs_err

        stage1_acc, stage1_macro, stage1_abs = summarize(stage1_preds, stage1_records)
        clf_acc, clf_macro, clf_abs = summarize(
            stage2_preds["classifier"],
            stage2_records["classifier"],
        )

        results.append(
            {
                "params": param_idx,
                "method": "stage1_net_rating",
                "blend_weight": None,
                "accuracy": stage1_acc,
                "macro_f1": stage1_macro,
                "abs_error": stage1_abs,
            }
        )
        results.append(
            {
                "params": param_idx,
                "method": "stage2_classifier",
                "blend_weight": None,
                "accuracy": clf_acc,
                "macro_f1": clf_macro,
                "abs_error": clf_abs,
            }
        )

        for blend_weight in BLEND_WEIGHTS:
            blend_acc, blend_macro, blend_abs = summarize(
                stage2_preds["blend"][blend_weight],
                stage2_records["blend"][blend_weight],
            )
            results.append(
                {
                    "params": param_idx,
                    "method": "stage2_blend",
                    "blend_weight": blend_weight,
                    "accuracy": blend_acc,
                    "macro_f1": blend_macro,
                    "abs_error": blend_abs,
                }
            )

    results_df = pd.DataFrame(results)
    results_df.sort_values(["macro_f1", "accuracy"], ascending=False, inplace=True)

    top_macro = results_df.head(5)
    top_abs = results_df.sort_values("abs_error").head(5)

    print("\nTop by Macro-F1:")
    print(top_macro)
    print("\nTop by Lowest Avg |error|:")
    print(top_abs)

    return results_df


if __name__ == "__main__":
    results = evaluate_params()
    results.to_csv(Path("poster_figures/two_stage_tuning_results.csv"), index=False)
