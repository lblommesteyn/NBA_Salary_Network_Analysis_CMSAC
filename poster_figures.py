"""
Generate poster-ready figures for the roster geometry project.

Run from the project root:

    python poster_figures.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize


RANDOM_STATE = 42
FIG_DIR = Path("poster_figures")
FIG_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")


FEATURE_LABELS: Dict[str, str] = {
    "team_net_rating": "Team Net Rating",
    "minutes_gini": "Minutes Inequality (Gini)",
    "minutes_assortativity": "Minutes Assortativity",
    "degree_centralization": "Degree Centralization",
    "edge_concentration_top5": "Top-5 Edge Concentration",
    "edge_concentration_top10": "Top-10 Edge Concentration",
    "modularity_Q": "Modularity Q",
    "community_size_cv": "Community Size CV",
    "salary_gini": "Salary Inequality (Gini)",
    "salary_top1_share": "Top Salary Share",
    "salary_top2_share": "Top 2 Salary Share",
    "salary_top3_share": "Top 3 Salary Share",
    "salary_assortativity": "Salary Assortativity",
    "RRS": "Roster Resilience Score",
    "shock_star": "Shock: Remove Max Salary",
    "shock_role": "Shock: Remove Rotation",
    "shock_connector": "Shock: Remove Connector",
    "shock_star_norm": "Shock/Net Rating",
    "shock_role_norm": "Role Shock/Net Rating",
    "shock_connector_norm": "Connector Shock/Net Rating",
    "salary_minutes_gap_top1": "Top Salary - Minutes Gap (1)",
    "salary_minutes_gap_top2": "Top Salary - Minutes Gap (2)",
    "salary_minutes_gap_top3": "Top Salary - Minutes Gap (3)",
    "minutes_to_salary_ratio_top1": "Minutes/Salaries Ratio (Top 1)",
    "minutes_to_salary_ratio_top2": "Minutes/Salaries Ratio (Top 2)",
    "minutes_to_salary_ratio_top3": "Minutes/Salaries Ratio (Top 3)",
}


@dataclass
class ModelingData:
    df: pd.DataFrame
    perm: pd.DataFrame
    rrs_full: pd.DataFrame


def load_datasets() -> ModelingData:
    base = pd.read_csv("vis_clean_data/features_from_pos_lineups.csv")
    base = (
        base.sort_values(["season", "team", "n_players"], ascending=[True, True, False])
        .drop_duplicates(subset=["season", "team"], keep="first")
        .reset_index(drop=True)
    )

    salary = pd.read_csv("vis_clean_data/salary_features.csv")
    rrs = pd.read_csv("vis_clean_data/rrs_from_pos_lineups.csv")
    rrs = (
        rrs.sort_values("RRS", ascending=False)
        .drop_duplicates(subset=["season", "team"], keep="first")
        .reset_index(drop=True)
    )
    playoff = pd.read_csv("vis_clean_data/playoff_rounds.csv")
    perm = pd.read_csv("vis_clean_data/salary_assortativity_perm_test.csv")

    df = base.merge(salary, on=["season", "team"], how="inner")
    df = df.merge(rrs, on=["season", "team"], how="left", suffixes=("", "_rrs"))
    df = df.merge(playoff, on=["season", "team"], how="inner")

    for col in ["shock_star", "shock_role", "shock_connector"]:
        df[col] = np.nan
    mask = df[["nr_hat_intact", "drop_star", "drop_role", "drop_connector"]].notna().all(axis=1)
    df.loc[mask, "shock_star"] = df.loc[mask, "nr_hat_intact"] - df.loc[mask, "drop_star"]
    df.loc[mask, "shock_role"] = df.loc[mask, "nr_hat_intact"] - df.loc[mask, "drop_role"]
    df.loc[mask, "shock_connector"] = df.loc[mask, "nr_hat_intact"] - df.loc[mask, "drop_connector"]
    for col in ["shock_star", "shock_role", "shock_connector"]:
        df[f"{col}_norm"] = df[col] / (df["team_net_rating"].abs() + 1e-6)

    df["salary_minutes_gap_top1"] = df["salary_top1_share"] - df["minutes_top1_share"]
    df["salary_minutes_gap_top2"] = df["salary_top2_share"] - df["minutes_top2_share"]
    df["salary_minutes_gap_top3"] = df["salary_top3_share"] - df["minutes_top3_share"]
    df["minutes_to_salary_ratio_top1"] = df["minutes_top1_share"] / (df["salary_top1_share"] + 1e-6)
    df["minutes_to_salary_ratio_top2"] = df["minutes_top2_share"] / (df["salary_top2_share"] + 1e-6)
    df["minutes_to_salary_ratio_top3"] = df["minutes_top3_share"] / (df["salary_top3_share"] + 1e-6)

    return ModelingData(df=df, perm=perm, rrs_full=rrs)



def evaluate_models(model_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Pipeline, Dict[str, pd.DataFrame]]:
    """Train baseline models with leave-one-season-out evaluation."""

    feature_cols = [
        "team_net_rating",
        "minutes_gini",
        "minutes_assortativity",
        "degree_centralization",
        "edge_concentration_top5",
        "modularity_Q",
        "community_size_cv",
        "salary_gini",
        "salary_assortativity",
        "salary_top1_share",
        "salary_top2_share",
        "salary_top3_share",
        "minutes_top1_share",
        "minutes_top2_share",
        "minutes_top3_share",
        "RRS",
        "shock_star",
        "shock_role",
        "shock_connector",
        "shock_star_norm",
        "shock_role_norm",
        "shock_connector_norm",
        "salary_minutes_gap_top1",
        "salary_minutes_gap_top2",
        "salary_minutes_gap_top3",
        "minutes_to_salary_ratio_top1",
        "minutes_to_salary_ratio_top2",
        "minutes_to_salary_ratio_top3",
    ]

    available = [c for c in feature_cols if c in model_df.columns]
    df = model_df.dropna(subset=available + ["playoff_round"]).reset_index(drop=True)
    X = df[available]
    y = df["playoff_round"].to_numpy()
    seasons = df["season"].to_numpy()

    models: Dict[str, Pipeline] = {
        "L2 Logistic": Pipeline(
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
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "HistGradientBoosting": Pipeline(
            [
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        learning_rate=0.08,
                        max_depth=4,
                        max_iter=400,
                        min_samples_leaf=8,
                        random_state=RANDOM_STATE,
                    ),
                )
            ]
        ),
        "Random Forest": Pipeline(
            [
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=400,
                        max_depth=6,
                        min_samples_leaf=3,
                        random_state=RANDOM_STATE,
                    ),
                )
            ]
        ),
    }

    fold_records: Dict[str, List[Dict[str, float]]] = {name: [] for name in models}
    logistic_predictions: List[Dict[str, float]] = []
    classes = np.sort(np.unique(y))

    for model_name, pipeline in models.items():
        for holdout in sorted(np.unique(seasons)):
            train_mask = seasons != holdout
            test_mask = ~train_mask
            if not np.any(test_mask):
                continue

            pipeline.fit(X.loc[train_mask], y[train_mask])
            y_pred = pipeline.predict(X.loc[test_mask])
            y_test = y[test_mask]

            record = {
                "season": holdout,
                "accuracy": accuracy_score(y_test, y_pred),
                "macro_f1": f1_score(y_test, y_pred, average="macro"),
                "roc_auc": np.nan,
                "brier": np.nan,
            }

            if hasattr(pipeline, "predict_proba"):
                proba = pipeline.predict_proba(X.loc[test_mask])
                try:
                    y_bin = label_binarize(y_test, classes=classes)
                    if proba.shape[1] == y_bin.shape[1]:
                        record["roc_auc"] = roc_auc_score(y_bin, proba, average="macro", multi_class="ovo")
                    record["brier"] = np.mean(np.sum((y_bin - proba) ** 2, axis=1))
                except ValueError:
                    pass

                if model_name == "L2 Logistic":
                    clf = pipeline.named_steps["clf"]
                    ge2_idx = np.where(clf.classes_ >= 2)[0]
                    prob_ge2 = proba[:, ge2_idx].sum(axis=1) if len(ge2_idx) else np.zeros(len(proba))
                    for team, season_val, round_val, pred_val, prob_val in zip(
                        df.loc[test_mask, "team"],
                        df.loc[test_mask, "season"],
                        y_test,
                        y_pred,
                        prob_ge2,
                    ):
                        logistic_predictions.append(
                            {
                                "team": team,
                                "season": season_val,
                                "playoff_round": round_val,
                                "pred_round": pred_val,
                                "prob_ge2": prob_val,
                            }
                        )

            fold_records[model_name].append(record)

    metrics_rows = []
    fold_map: Dict[str, pd.DataFrame] = {}
    for name, records in fold_records.items():
        if not records:
            continue
        fold_df = pd.DataFrame(records)
        fold_map[name] = fold_df
        metrics_rows.append(
            {
                "model": name,
                "accuracy_mean": fold_df["accuracy"].mean(),
                "accuracy_std": fold_df["accuracy"].std(ddof=0),
                "macro_f1_mean": fold_df["macro_f1"].mean(),
                "macro_f1_std": fold_df["macro_f1"].std(ddof=0),
                "roc_auc_mean": fold_df["roc_auc"].mean(),
                "brier_mean": fold_df["brier"].mean(),
            }
        )

    metrics_df = pd.DataFrame(metrics_rows)

    logistic_pipeline = models["L2 Logistic"]
    logistic_pipeline.fit(X, y)

    preds_df = pd.DataFrame(logistic_predictions)

    return metrics_df, preds_df, logistic_pipeline, fold_map

def plot_model_comparison(metrics_df: pd.DataFrame, fold_map: Dict[str, pd.DataFrame]) -> None:
    df = metrics_df.sort_values("macro_f1_mean", ascending=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
    palette = sns.color_palette("viridis", n_colors=len(df))
    for ax, metric, label in zip(
        axes,
        ["macro_f1_mean", "roc_auc_mean", "brier_mean"],
        ["Macro-F1", "ROC-AUC (OVO)", "Brier Score (lower is better)"],
    ):
        sns.barplot(data=df, x=metric, y="model", ax=ax, palette=palette, errorbar=None)
        metric_name = metric.replace("_mean", "")
        for idx, (model_name, row) in enumerate(zip(df["model"], df.itertuples())):
            fold_df = fold_map.get(model_name)
            if fold_df is None or metric_name not in fold_df:
                continue
            values = fold_df[metric_name].dropna()
            if len(values) > 1:
                err = values.std(ddof=1)
                if np.isfinite(err) and err > 0:
                    ax.errorbar(
                        getattr(row, metric),
                        idx,
                        xerr=err,
                        fmt="none",
                        ecolor="black",
                        capsize=4,
                        alpha=0.8,
                    )
        ax.set_xlabel(label)
        ax.set_ylabel("")
    fig.suptitle("Model Stack Performance (mean � fold std)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "poster_model_comparison.png", dpi=300)
    plt.close(fig)


def metrics_table_figure(metrics_df: pd.DataFrame) -> None:
    display_df = metrics_df[["model", "accuracy_mean", "macro_f1_mean", "roc_auc_mean", "brier_mean"]].copy()
    display_df.columns = ["Model", "Accuracy", "Macro-F1", "ROC-AUC", "Brier"]
    display_df = display_df.round(3)
    fig, ax = plt.subplots(figsize=(8, 2 + 0.35 * len(display_df)))
    ax.axis("off")
    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    ax.set_title("Model Summary Statistics", fontweight="bold", fontsize=16, pad=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "poster_metrics_table.png", dpi=300)
    plt.close(fig)


def headline_scorecard(metrics_df: pd.DataFrame, lift_df: pd.DataFrame) -> None:
    logistic_row = metrics_df.loc[metrics_df["model"] == "L2 Logistic"].squeeze()
    best_row = metrics_df.sort_values("macro_f1_mean", ascending=False).iloc[0]
    mean_lift = lift_df["lift"].mean()
    positive = (lift_df["lift"] > 0).sum()
    total = len(lift_df)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    y = 0.9
    lines = [
        "Headline Results",
        f"Logistic Macro-F1: {logistic_row['macro_f1_mean']:.3f} (Brier {logistic_row['brier_mean']:.3f})",
        f"Best Model: {best_row['model']} (Macro-F1 {best_row['macro_f1_mean']:.3f})",
        f"Average Topology Lift: {mean_lift:+.3f}",
        f"Positive Lift Seasons: {positive}/{total}",
    ]
    for idx, text in enumerate(lines):
        ax.text(0.02, y, text, fontsize=16 if idx == 0 else 14, fontweight="bold" if idx == 0 else "normal")
        y -= 0.12
    fig.tight_layout()
    fig.savefig(FIG_DIR / "poster_headline_scorecard.png", dpi=300)
    plt.close(fig)


def playoff_confusion_matrix(logistic_predictions: pd.DataFrame) -> None:
    if logistic_predictions.empty:
        return
    y_true = logistic_predictions["playoff_round"].astype(int).to_numpy()
    y_pred = logistic_predictions["pred_round"].astype(int).to_numpy()
    labels = np.sort(np.unique(np.concatenate([y_true, y_pred])))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_pct = np.divide(cm, cm.sum(axis=1, keepdims=True), where=cm.sum(axis=1, keepdims=True) != 0)
    cm_pct = np.nan_to_num(cm_pct)
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]} ({cm_pct[i, j]*100:.1f}%)"
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm_pct, annot=annot, fmt="", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted Playoff Round")
    ax.set_ylabel("Actual Playoff Round")
    ax.set_title("Confusion Matrix (row-normalised % shown)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "poster_confusion_matrix.png", dpi=300)
    plt.close(fig)


def logistic_hyperparameter_surface(model_df: pd.DataFrame) -> None:
    feature_cols = [
        "team_net_rating",
        "minutes_gini",
        "minutes_assortativity",
        "degree_centralization",
        "edge_concentration_top5",
        "modularity_Q",
        "community_size_cv",
        "salary_gini",
        "salary_assortativity",
        "salary_top1_share",
        "salary_top2_share",
        "salary_top3_share",
        "minutes_top1_share",
        "minutes_top2_share",
        "minutes_top3_share",
        "RRS",
        "shock_star",
        "shock_role",
        "shock_connector",
        "shock_star_norm",
        "shock_role_norm",
        "shock_connector_norm",
        "salary_minutes_gap_top1",
        "salary_minutes_gap_top2",
        "salary_minutes_gap_top3",
        "minutes_to_salary_ratio_top1",
        "minutes_to_salary_ratio_top2",
        "minutes_to_salary_ratio_top3",
    ]
    df = model_df.dropna(subset=feature_cols + ["playoff_round"]).reset_index(drop=True)
    X = df[feature_cols]
    y = df["playoff_round"].to_numpy()
    seasons = df["season"].to_numpy()
    C_values = np.round(np.logspace(-2, 1, 8), 4)
    l1_values = np.linspace(0.0, 1.0, 6)
    heat = np.full((len(l1_values), len(C_values)), np.nan)
    for i, l1_ratio in enumerate(l1_values):
        for j, C in enumerate(C_values):
            model = Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            penalty="elasticnet",
                            solver="saga",
                            l1_ratio=l1_ratio,
                            C=C,
                            max_iter=5000,
                            multi_class="multinomial",
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            )
            scores = []
            for holdout in sorted(np.unique(seasons)):
                train_mask = seasons != holdout
                test_mask = ~train_mask
                if not np.any(test_mask):
                    continue
                model.fit(X.loc[train_mask], y[train_mask])
                preds = model.predict(X.loc[test_mask])
                scores.append(f1_score(y[test_mask], preds, average="macro"))
            if scores:
                heat[i, j] = np.mean(scores)
    mask = np.isnan(heat)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        heat,
        annot=True,
        fmt=".2f",
        cmap="crest",
        mask=mask,
        xticklabels=[str(v) for v in C_values],
        yticklabels=[f"{v:.2f}" for v in l1_values],
        ax=ax,
    )
    ax.set_xlabel("Inverse Regularisation Strength (C)")
    ax.set_ylabel("Elastic-Net Mixing (L1 Ratio)")
    ax.set_title("Macro-F1 Surface for Elastic-Net Logistic Regression")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "poster_logistic_hyperparameter_surface.png", dpi=300)
    plt.close(fig)

def season_lift_chart(model_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    control_features = [
        "team_net_rating",
        "minutes_gini",
        "minutes_top1_share",
        "minutes_top2_share",
        "minutes_top3_share",
    ]
    topo_features = [
        "team_net_rating",
        "minutes_gini",
        "minutes_assortativity",
        "degree_centralization",
        "edge_concentration_top5",
        "modularity_Q",
        "community_size_cv",
        "salary_gini",
        "salary_assortativity",
        "salary_top1_share",
        "RRS",
        "shock_star",
        "shock_role",
        "shock_connector",
    ]
    df = model_df.dropna(subset=set(control_features + topo_features + ["playoff_round"])).reset_index(drop=True)
    seasons = sorted(df["season"].unique())
    rows = []
    for holdout in seasons:
        train_mask = df["season"] != holdout
        test_mask = ~train_mask
        if not np.any(test_mask):
            continue
        ctrl = Pipeline(
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
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
        topo = Pipeline(
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
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
        ctrl.fit(df.loc[train_mask, control_features], df.loc[train_mask, "playoff_round"])
        topo.fit(df.loc[train_mask, topo_features], df.loc[train_mask, "playoff_round"])
        y_test = df.loc[test_mask, "playoff_round"]
        ctrl_pred = ctrl.predict(df.loc[test_mask, control_features])
        topo_pred = topo.predict(df.loc[test_mask, topo_features])
        rows.append(
            {
                "season": holdout,
                "spec": "Control",
                "macro_f1": f1_score(y_test, ctrl_pred, average="macro"),
                "accuracy": accuracy_score(y_test, ctrl_pred),
            }
        )
        rows.append(
            {
                "season": holdout,
                "spec": "Topology",
                "macro_f1": f1_score(y_test, topo_pred, average="macro"),
                "accuracy": accuracy_score(y_test, topo_pred),
            }
        )
    season_df = pd.DataFrame(rows)
    lift_df = (
        season_df.pivot(index="season", columns="spec", values="macro_f1")
        .assign(lift=lambda d: d["Topology"] - d["Control"])
        .reset_index()
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(data=season_df, x="season", y="macro_f1", hue="spec", ax=axes[0])
    axes[0].set_title("Per-season Macro-F1")
    axes[0].set_ylabel("Macro-F1")
    axes[0].legend(title="Model")
    colors = lift_df["lift"].apply(lambda v: "#2ca02c" if v >= 0 else "#d62728")
    sns.barplot(data=lift_df, x="season", y="lift", palette=colors.tolist(), ax=axes[1])
    axes[1].axhline(0, color="black", linewidth=1)
    for idx, row in lift_df.iterrows():
        va = "bottom" if row["lift"] >= 0 else "top"
        offset = 0.002 if row["lift"] >= 0 else -0.002
        axes[1].text(idx, row["lift"] + offset, f"{row['lift']:+.3f}", ha="center", va=va, fontsize=10)
    axes[1].set_title("Macro-F1 Lift (Topology - Control)")
    axes[1].set_ylabel("Macro-F1 Lift")
    fig.suptitle("Seasonal Contribution of Topology Features")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "poster_season_lift.png", dpi=300)
    plt.close(fig)
    return season_df, lift_df


def topology_permutation_importance(logistic_model: Pipeline, model_df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        "team_net_rating",
        "minutes_gini",
        "minutes_assortativity",
        "degree_centralization",
        "edge_concentration_top5",
        "modularity_Q",
        "community_size_cv",
        "salary_gini",
        "salary_assortativity",
        "salary_top1_share",
        "salary_top2_share",
        "salary_top3_share",
        "minutes_top1_share",
        "minutes_top2_share",
        "minutes_top3_share",
        "RRS",
        "shock_star",
        "shock_role",
        "shock_connector",
        "shock_star_norm",
        "shock_role_norm",
        "shock_connector_norm",
        "salary_minutes_gap_top1",
        "salary_minutes_gap_top2",
        "salary_minutes_gap_top3",
        "minutes_to_salary_ratio_top1",
        "minutes_to_salary_ratio_top2",
        "minutes_to_salary_ratio_top3",
    ]
    df = model_df.dropna(subset=feature_cols + ["playoff_round"]).reset_index(drop=True)
    X = df[feature_cols]
    y = df["playoff_round"].to_numpy()
    result = permutation_importance(
        logistic_model,
        X,
        y,
        n_repeats=200,
        random_state=RANDOM_STATE,
        scoring="f1_macro",
    )
    imp = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": result.importances_mean,
            "std": result.importances_std,
        }
    ).sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(data=imp, x="importance", y="feature", palette="viridis", ax=ax, errorbar=None)
    ax.errorbar(imp["importance"], np.arange(len(imp)), xerr=imp["std"], fmt="none", ecolor="black", capsize=4)
    ax.set_xlabel("Permutation Importance (? Macro-F1)")
    ax.set_ylabel("")
    ax.set_yticklabels([FEATURE_LABELS.get(f, f) for f in imp["feature"]])
    ax.set_title("Permutation Importance - Multinomial Logistic")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "poster_permutation_importance.png", dpi=300)
    plt.close(fig)
    return imp


def effect_size_odds_ratios(model_df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        "team_net_rating",
        "minutes_gini",
        "minutes_assortativity",
        "degree_centralization",
        "edge_concentration_top5",
        "modularity_Q",
        "community_size_cv",
        "salary_gini",
        "salary_assortativity",
        "salary_top1_share",
        "salary_top2_share",
        "salary_top3_share",
        "minutes_top1_share",
        "minutes_top2_share",
        "minutes_top3_share",
        "RRS",
        "shock_star",
        "shock_role",
        "shock_connector",
        "shock_star_norm",
        "shock_role_norm",
        "shock_connector_norm",
        "salary_minutes_gap_top1",
        "salary_minutes_gap_top2",
        "salary_minutes_gap_top3",
        "minutes_to_salary_ratio_top1",
        "minutes_to_salary_ratio_top2",
        "minutes_to_salary_ratio_top3",
    ]
    df = model_df.dropna(subset=feature_cols + ["playoff_round"]).reset_index(drop=True)
    X = df[feature_cols]
    y = (df["playoff_round"] >= 2).astype(int)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_design = sm.add_constant(X_scaled)
    model = sm.Logit(y, X_design)
    result = model.fit(disp=False)
    coefs = result.params[1:]
    odds_ratios = np.exp(coefs)
    conf = result.conf_int().iloc[1:]
    or_low = np.exp(conf[0])
    or_high = np.exp(conf[1])
    effect_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "odds_ratio": odds_ratios,
            "or_ci_low": or_low,
            "or_ci_high": or_high,
        }
    )
    effect_df["feature_label"] = effect_df["feature"].map(FEATURE_LABELS).fillna(effect_df["feature"])
    effect_df = effect_df.sort_values("odds_ratio", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.barplot(data=effect_df, x="odds_ratio", y="feature_label", palette="viridis", ax=ax, errorbar=None)
    xerr = [effect_df["odds_ratio"] - effect_df["or_ci_low"], effect_df["or_ci_high"] - effect_df["odds_ratio"]]
    ax.errorbar(effect_df["odds_ratio"], np.arange(len(effect_df)), xerr=xerr, fmt="none", ecolor="black", capsize=4)
    ax.axvline(1.0, color="black", linestyle="--")
    ax.set_xlabel("Odds Ratio per 1 SD (Advance >= 2 Rounds)")
    ax.set_ylabel("")
    ax.set_title("Effect Size Summary (Binary Logistic)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "poster_effect_sizes.png", dpi=300)
    plt.close(fig)
    return effect_df


def _prepare_error_dataframe(model_df: pd.DataFrame, logistic_predictions: pd.DataFrame) -> pd.DataFrame:
    merged = model_df.merge(
        logistic_predictions[["season", "team", "playoff_round", "pred_round", "prob_ge2"]],
        on=["season", "team", "playoff_round"],
        how="left",
    ).dropna(subset=["pred_round"])
    merged["error"] = merged["pred_round"] - merged["playoff_round"]
    merged["error_cat"] = merged["error"].apply(lambda x: "Under" if x < 0 else ("Over" if x > 0 else "Exact"))
    for col in ["shock_star", "shock_role", "shock_connector"]:
        merged[f"{col}_norm"] = merged[col] / (merged["team_net_rating"].abs() + 1e-6)
    return merged


def partial_dependence_curves(logistic_model: Pipeline, model_df: pd.DataFrame) -> None:
    feature_cols = [
        "team_net_rating",
        "minutes_gini",
        "minutes_assortativity",
        "degree_centralization",
        "edge_concentration_top5",
        "modularity_Q",
        "community_size_cv",
        "salary_gini",
        "salary_assortativity",
        "salary_top1_share",
        "salary_top2_share",
        "salary_top3_share",
        "minutes_top1_share",
        "minutes_top2_share",
        "minutes_top3_share",
        "RRS",
        "shock_star",
        "shock_role",
        "shock_connector",
        "shock_star_norm",
        "shock_role_norm",
        "shock_connector_norm",
        "salary_minutes_gap_top1",
        "salary_minutes_gap_top2",
        "salary_minutes_gap_top3",
        "minutes_to_salary_ratio_top1",
        "minutes_to_salary_ratio_top2",
        "minutes_to_salary_ratio_top3",
    ]
    df = model_df.dropna(subset=feature_cols + ["playoff_round"]).reset_index(drop=True)
    X = df[feature_cols]
    clf = logistic_model.named_steps["clf"]
    ge2_idx = np.where(clf.classes_ >= 2)[0]

    def _pdp(feature: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        grid = np.linspace(X[feature].quantile(0.05), X[feature].quantile(0.95), 40)
        pd_values = []
        ice = []
        base = X.copy()
        for value in grid:
            modified = base.copy()
            modified[feature] = value
            proba = logistic_model.predict_proba(modified)
            prob_ge2 = proba[:, ge2_idx].sum(axis=1) if len(ge2_idx) else np.zeros(len(proba))
            pd_values.append(prob_ge2.mean())
            ice.append(prob_ge2)
        return grid, np.array(pd_values), np.array(ice)

    focus = ["salary_assortativity", "edge_concentration_top5"]
    fig, axes = plt.subplots(1, len(focus), figsize=(14, 6))
    rng = np.random.default_rng(RANDOM_STATE)
    for axis, feature in zip(np.atleast_1d(axes), focus):
        grid, mean_curve, ice = _pdp(feature)
        lower = np.percentile(ice, 5, axis=1)
        upper = np.percentile(ice, 95, axis=1)
        axis.fill_between(grid, lower, upper, color="#1f77b4", alpha=0.2, label="5th-95th percentile")
        sample_idx = rng.choice(ice.shape[1], size=min(20, ice.shape[1]), replace=False)
        axis.plot(grid, ice[:, sample_idx], color="#1f77b4", alpha=0.1)
        axis.plot(grid, mean_curve, color="#1f77b4", linewidth=3, label="Average PD")
        axis.set_xlabel(FEATURE_LABELS.get(feature, feature))
        axis.set_ylabel("Pr(Advancing >= 2 Rounds)")
        axis.set_ylim(0, 1)
        axis.set_title(f"Effect of {FEATURE_LABELS.get(feature, feature)}")
        axis.legend(loc="best")
    fig.suptitle("Partial Dependence & ICE - Probability of Deep Playoff Run")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "poster_partial_dependence.png", dpi=300)
    plt.close(fig)


def bootstrap_macro_f1_gain(lift_df: pd.DataFrame) -> None:
    diffs = lift_df["lift"].to_numpy()
    rng = np.random.default_rng(RANDOM_STATE)
    boot = np.array([diffs[rng.integers(0, len(diffs), size=len(diffs))].mean() for _ in range(1000)])
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(boot, bins=30, kde=True, ax=ax, color="#1f77b4")
    ax.axvline(boot.mean(), color="black", linestyle="--", label=f"Mean = {boot.mean():.3f}")
    ax.axvline(ci_low, color="red", linestyle=":", label="95% CI")
    ax.axvline(ci_high, color="red", linestyle=":")
    ax.set_xlabel("Macro-F1 Gain (Topology - Control)")
    ax.set_ylabel("Bootstrap Frequency")
    ax.set_title("Season-Level Bootstrap of Macro-F1 Lift")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "poster_bootstrap_macro_f1.png", dpi=300)
    plt.close(fig)

def calibration_curve(model_df: pd.DataFrame) -> None:
    feature_cols = [
        "team_net_rating",
        "minutes_gini",
        "minutes_assortativity",
        "degree_centralization",
        "edge_concentration_top5",
        "modularity_Q",
        "community_size_cv",
        "salary_gini",
        "salary_assortativity",
        "salary_top1_share",
        "salary_top2_share",
        "salary_top3_share",
        "minutes_top1_share",
        "minutes_top2_share",
        "minutes_top3_share",
        "RRS",
        "shock_star",
        "shock_role",
        "shock_connector",
        "shock_star_norm",
        "shock_role_norm",
        "shock_connector_norm",
        "salary_minutes_gap_top1",
        "salary_minutes_gap_top2",
        "salary_minutes_gap_top3",
        "minutes_to_salary_ratio_top1",
        "minutes_to_salary_ratio_top2",
        "minutes_to_salary_ratio_top3",
    ]
    df = model_df.dropna(subset=feature_cols + ["playoff_round"]).reset_index(drop=True)
    if df.empty:
        return
    seasons = df["season"].to_numpy()
    X = df[feature_cols]
    y_round = df["playoff_round"].to_numpy()
    y_binary = (y_round >= 2).astype(int)
    logistic_probs: List[np.ndarray] = []
    iso_probs: List[np.ndarray] = []
    y_holdout: List[np.ndarray] = []
    for holdout in sorted(np.unique(seasons)):
        train_mask = seasons != holdout
        test_mask = ~train_mask
        if not np.any(test_mask):
            continue
        model = Pipeline(
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
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
        model.fit(X.loc[train_mask], y_round[train_mask])
        prob_train = model.predict_proba(X.loc[train_mask])
        prob_test = model.predict_proba(X.loc[test_mask])
        classes = model.named_steps["clf"].classes_
        ge2_idx = np.where(classes >= 2)[0]
        if len(ge2_idx) == 0:
            continue
        prob_train_ge2 = prob_train[:, ge2_idx].sum(axis=1)
        prob_test_ge2 = prob_test[:, ge2_idx].sum(axis=1)
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(prob_train_ge2, y_binary[train_mask])
        iso_test = iso.transform(prob_test_ge2)
        logistic_probs.append(prob_test_ge2)
        iso_probs.append(iso_test)
        y_holdout.append(y_binary[test_mask])
    if not logistic_probs:
        return
    prob = np.concatenate(logistic_probs)
    prob_iso = np.concatenate(iso_probs)
    y_true = np.concatenate(y_holdout)
    plot_df = pd.DataFrame({"prob": prob, "prob_iso": prob_iso, "y": y_true})
    plot_df["bin"] = pd.cut(plot_df["prob"], bins=np.linspace(0, 1, 9), include_lowest=True)
    grouped = plot_df.groupby("bin").agg(
        predicted=("prob", "mean"),
        observed=("y", "mean"),
        count=("prob", "size"),
    )
    plot_df["bin_iso"] = pd.cut(plot_df["prob_iso"], bins=np.linspace(0, 1, 9), include_lowest=True)
    grouped_iso = plot_df.groupby("bin_iso").agg(
        predicted=("prob_iso", "mean"),
        observed=("y", "mean"),
    )
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    ax.scatter(
        grouped["predicted"],
        grouped["observed"],
        s=np.clip(grouped["count"], 1, None) * 5,
        color="#1f77b4",
        label="Logistic (cross-val)",
    )
    ax.plot(
        grouped_iso["predicted"],
        grouped_iso["observed"],
        color="#ff7f0e",
        marker="o",
        label="Isotonic (train calibrated)",
    )
    for (_, row) in grouped.iterrows():
        ax.text(row["predicted"], row["observed"], str(int(row["count"])), fontsize=9, ha="center", va="bottom")
    ax.set_xlabel("Predicted Probability (Advance >= 2 Rounds)")
    ax.set_ylabel("Observed Frequency")
    ax.set_title("Cross-Validated Calibration of Deep-Run Probabilities")
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "poster_calibration_curve.png", dpi=300)
    plt.close(fig)


def error_feature_summary(model_df: pd.DataFrame, logistic_predictions: pd.DataFrame) -> None:
    if logistic_predictions.empty:
        return
    merged = _prepare_error_dataframe(model_df, logistic_predictions)
    features = [
        "team_net_rating",
        "salary_assortativity",
        "edge_concentration_top5",
        "RRS",
        "shock_star",
        "shock_role",
        "shock_connector",
        "shock_star_norm",
        "shock_role_norm",
        "shock_connector_norm",
    ]
    merged = merged.dropna(subset=features)
    overall = merged[features].mean()
    deltas = merged.groupby("error_cat")[features].mean() - overall
    deltas = deltas.reindex(["Under", "Exact", "Over"])
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(deltas, annot=True, fmt="+.2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Feature Deviations by Prediction Error Class")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "poster_error_feature_deltas.png", dpi=300)
    plt.close(fig)


def glossary_figure() -> None:
    entries = [
        ("Roster Resilience Score (RRS)", "Average predicted net-rating drop after removing star, role, and connector players."),
        ("NR_hat_intact", "Predicted net rating with the full roster before stress tests."),
        ("Salary Assortativity", "Correlation between salaries of teammates who share minutes (negative = more mixing)."),
        ("Edge Concentration", "Share of total shared minutes captured by the strongest lineup links."),
        ("Degree Centralization", "Extent to which the network revolves around a few players."),
        ("Shock Scenarios", "Synthetic removals of stars/roles/connectors used to score resilience."),
    ]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")
    y = 0.9
    for title, desc in entries:
        ax.text(0.02, y, title, fontsize=16, fontweight="bold", ha="left", va="center")
        y -= 0.06
        ax.text(0.04, y, desc, fontsize=14, ha="left", va="center", wrap=True)
        y -= 0.08
    ax.set_title("Key Terms at a Glance", fontsize=18, fontweight="bold", loc="left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "poster_glossary.png", dpi=300)
    plt.close(fig)


def salary_assortativity_violin(perm: pd.DataFrame) -> None:
    tidy = perm.melt(
        id_vars=["season", "team"],
        value_vars=["assort_salary_obs", "assort_mu_null"],
        var_name="metric",
        value_name="assortativity",
    )
    tidy["metric"] = tidy["metric"].map(
        {"assort_salary_obs": "Observed", "assort_mu_null": "Permutation Null Mean"}
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.violinplot(data=tidy, x="metric", y="assortativity", palette="viridis", inner="quartile", ax=ax)
    sns.swarmplot(data=tidy, x="metric", y="assortativity", color="k", size=2, alpha=0.35, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Salary Assortativity")
    ax.set_title("Observed Salary Mixing vs. Null Expectation")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "poster_salary_assortativity_violin.png", dpi=300)
    plt.close(fig)


def salary_assortativity_null_panels(perm: pd.DataFrame) -> None:
    perm = perm.dropna(subset=["assort_salary_obs", "assort_mu_null", "assort_sd_null"])
    if perm.empty:
        return
    selections = [
        perm.nsmallest(1, "assort_z").iloc[0],
        perm.iloc[len(perm) // 2],
        perm.nlargest(1, "assort_z").iloc[0],
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, row in zip(axes, selections):
        mu = row["assort_mu_null"]
        sd = row["assort_sd_null"]
        obs = row["assort_salary_obs"]
        ax.axvspan(mu - sd, mu + sd, color="#1f77b4", alpha=0.2, label="Null +/- 1 SD")
        ax.axvline(mu, color="#1f77b4", linestyle="-", linewidth=2, label="Null mean")
        ax.axvline(obs, color="#d62728", linestyle="--", linewidth=2, label="Observed")
        ax.set_title(f"{row['team']} {row['season']} (z={row['assort_z']:.2f})")
        ax.set_xlabel("Salary Assortativity")
        ax.set_xlim(min(mu - 2 * sd, obs - 0.05), max(mu + 2 * sd, obs + 0.05))
        if ax is axes[0]:
            ax.set_ylabel("Density Proxy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.suptitle("Observed Salary Mixing vs. Permutation Null (Mean +/- 1 SD)")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(FIG_DIR / "poster_salary_null_panels.png", dpi=300)
    plt.close(fig)

def ablation_waterfall() -> pd.DataFrame:
    path = Path("vis_clean_data/table_ablations_playoff.csv")
    if not path.exists():
        return pd.DataFrame()
    table = pd.read_csv(path)
    order = ["A: Controls only", "B: + Dispersion", "C: + Connectivity", "D: + Full topology"]
    table = table.set_index("Model").loc[order].reset_index()
    table["increment"] = table["Macro-F1"].diff().fillna(table["Macro-F1"])
    colors = table["increment"].apply(lambda v: "#2ca02c" if v >= 0 else "#d62728")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=table, x="Model", y="increment", palette=colors.tolist(), ax=ax, dodge=False)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("Macro-F1 Gain vs. Previous Step")
    ax.set_xlabel("Feature Set")
    ax.set_title("Feature Group Contributions to Macro-F1")
    for idx, row in table.iterrows():
        va = "bottom" if row["increment"] >= 0 else "top"
        offset = 0.002 if row["increment"] >= 0 else -0.002
        ax.text(
            idx,
            row["increment"] + offset,
            f"?{row['increment']:+.3f}\nTotal {row['Macro-F1']:.3f}",
            ha="center",
            va=va,
            fontsize=10,
        )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "poster_ablation_waterfall.png", dpi=300)
    plt.close(fig)
    return table


def rrs_shock_heatmap(rrs: pd.DataFrame) -> None:
    data = rrs.copy()
    for col, new_col in [
        ("drop_star", "shock_star"),
        ("drop_role", "shock_role"),
        ("drop_connector", "shock_connector"),
    ]:
        data[new_col] = data["nr_hat_intact"] - data[col]
    summary = data.groupby("season")[["shock_star", "shock_role", "shock_connector", "RRS"]].mean().sort_index()
    heat = summary[["shock_star", "shock_role", "shock_connector"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(heat, annot=True, fmt=".2f", cmap="mako", ax=ax)
    ax.set_xlabel("Removal Scenario")
    ax.set_ylabel("Season")
    ax.set_title("Average Net-Rating Drop After Player Removal")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "poster_rrs_shock_heatmap.png", dpi=300)
    plt.close(fig)


def case_study_shock_profiles(rrs: pd.DataFrame) -> None:
    data = rrs.copy()
    for col, new_col in [
        ("drop_star", "shock_star"),
        ("drop_role", "shock_role"),
        ("drop_connector", "shock_connector"),
    ]:
        data[new_col] = data["nr_hat_intact"] - data[col]
    data["avg_drop"] = data[["shock_star", "shock_role", "shock_connector"]].mean(axis=1)
    resilient = data.nsmallest(1, "avg_drop").iloc[0]
    fragile = data.nlargest(1, "avg_drop").iloc[0]
    selected = pd.DataFrame(
        [
            {"profile": f"{resilient['team']} {resilient['season']} (Resilient)", "shock_star": resilient["shock_star"], "shock_role": resilient["shock_role"], "shock_connector": resilient["shock_connector"]},
            {"profile": f"{fragile['team']} {fragile['season']} (Fragile)", "shock_star": fragile["shock_star"], "shock_role": fragile["shock_role"], "shock_connector": fragile["shock_connector"]},
        ]
    )
    tidy = selected.melt(id_vars="profile", var_name="scenario", value_name="rating_drop")
    tidy["scenario"] = tidy["scenario"].map(
        {
            "shock_star": "Remove Max Salary",
            "shock_role": "Remove Rotation Player",
            "shock_connector": "Remove Connector",
        }
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(data=tidy, x="scenario", y="rating_drop", hue="profile", ax=ax)
    ax.set_xlabel("Shock Scenario")
    ax.set_ylabel("Predicted Net Rating Drop")
    ax.set_title("Shock Response Case Study")
    ax.legend(title="Team & Season")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "poster_case_study_shocks.png", dpi=300)
    plt.close(fig)


def topology_cluster_map(model_df: pd.DataFrame) -> None:
    feature_cols = [
        "degree_centralization",
        "edge_concentration_top5",
        "modularity_Q",
        "community_size_cv",
        "salary_gini",
        "salary_assortativity",
        "RRS",
        "shock_star",
        "shock_role",
        "shock_connector",
    ]
    df = model_df.dropna(subset=feature_cols + ["team_net_rating"]).reset_index(drop=True)
    X = StandardScaler().fit_transform(df[feature_cols])
    kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=20).fit(X)
    labels = kmeans.labels_
    df_clusters = df.assign(cluster=labels)
    cluster_means = df_clusters.groupby("cluster")[feature_cols + ["team_net_rating"]].mean()
    star_cluster = cluster_means["degree_centralization"].idxmax()
    resilient_cluster = cluster_means["RRS"].idxmin()
    cluster_names = {star_cluster: "Star-Centered", resilient_cluster: "Resilient Mesh"}
    for idx in cluster_means.index:
        cluster_names.setdefault(idx, "Connector Hubs")
    coords = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=df_clusters["cluster"],
        cmap="Set2",
        s=80,
        alpha=0.85,
    )
    handles, legend_labels = scatter.legend_elements()
    readable = []
    for label in legend_labels:
        match = re.search(r"\d+", label)
        label_idx = int(match.group()) if match else 0
        readable.append(cluster_names.get(label_idx, f"Cluster {label_idx}"))
    ax.legend(handles, readable, title="Archetype", loc="best")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title("Roster Topology Archetypes")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "poster_cluster_map.png", dpi=300)
    plt.close(fig)


def cross_validation_schematic(seasons: Iterable[str]) -> None:
    seasons = list(seasons)
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.axis("off")
    width, height, y = 1.0, 0.8, 0.1
    for idx, season in enumerate(seasons):
        x = idx * (width + 0.1)
        rect = plt.Rectangle((x, y), width, height, edgecolor="black", facecolor="#5dade2", alpha=0.7)
        ax.add_patch(rect)
        ax.text(x + width / 2, y + height / 2, season, ha="center", va="center", fontsize=12, fontweight="bold")
        ax.text(
            x + width / 2,
            y + height + 0.15,
            "Hold-out Fold" if idx == len(seasons) // 2 else "",
            ha="center",
            va="bottom",
            fontsize=11,
        )
    ax.set_xlim(-0.1, len(seasons) * (width + 0.1))
    ax.set_ylim(0, 1.5)
    ax.set_title("Leave-One-Season-Out Cross Validation")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "poster_cv_schematic.png", dpi=300)
    plt.close(fig)


def main() -> None:
    data = load_datasets()
    metrics_df, logistic_preds, logistic_model, fold_map = evaluate_models(data.df)
    season_df, lift_df = season_lift_chart(data.df)
    plot_model_comparison(metrics_df, fold_map)
    metrics_table_figure(metrics_df)
    headline_scorecard(metrics_df, lift_df)
    playoff_confusion_matrix(logistic_preds)
    logistic_hyperparameter_surface(data.df)
    topology_permutation_importance(logistic_model, data.df)
    effect_size_odds_ratios(data.df)
    partial_dependence_curves(logistic_model, data.df)
    bootstrap_macro_f1_gain(lift_df)
    calibration_curve(data.df)
    error_feature_summary(data.df, logistic_preds)
    ordinal_error_heatmaps(data.df, logistic_preds)
    ordinal_error_distribution(data.df, logistic_preds)
    archetype_error_summary(data.df, logistic_preds)
    shock_vs_net_rating_scatter(data.df, logistic_preds)
    glossary_figure()
    salary_assortativity_violin(data.perm)
    salary_assortativity_null_panels(data.perm)
    ablation_waterfall()
    rrs_shock_heatmap(data.rrs_full)
    case_study_shock_profiles(data.rrs_full)
    topology_cluster_map(data.df)
    cross_validation_schematic(sorted(data.df["season"].unique()))


def cluster_roster_archetypes(model_df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, Dict[int, str]]:
    feature_cols = [
        "degree_centralization",
        "edge_concentration_top5",
        "modularity_Q",
        "community_size_cv",
        "salary_gini",
        "salary_assortativity",
        "RRS",
        "shock_star",
        "shock_role",
        "shock_connector",
    ]
    df_clean = model_df.dropna(subset=feature_cols + ["team_net_rating"]).reset_index(drop=True)
    X = StandardScaler().fit_transform(df_clean[feature_cols])
    kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=20).fit(X)
    labels = kmeans.labels_
    df_clusters = df_clean.assign(cluster=labels)
    cluster_means = df_clusters.groupby("cluster")[feature_cols + ["team_net_rating"]].mean()
    star_cluster = cluster_means["degree_centralization"].idxmax()
    resilient_cluster = cluster_means["RRS"].idxmin()
    cluster_names: Dict[int, str] = {star_cluster: "Star-Centered", resilient_cluster: "Resilient Mesh"}
    for idx in cluster_means.index:
        cluster_names.setdefault(idx, "Connector Hubs")
    df_clusters["archetype"] = df_clusters["cluster"].map(cluster_names)
    return df_clusters, X, cluster_names

def archetype_error_summary(model_df: pd.DataFrame, logistic_predictions: pd.DataFrame) -> None:
    if logistic_predictions.empty:
        return
    df_error = _prepare_error_dataframe(model_df, logistic_predictions)
    clusters_df, _, cluster_names = cluster_roster_archetypes(model_df)
    merged = df_error.merge(
        clusters_df[["season", "team", "archetype"]],
        on=["season", "team"],
        how="left",
    )
    summary = (
        merged.groupby(["archetype", "error_cat"]).size().unstack(fill_value=0)
    )
    summary = summary.reindex(sorted(summary.index))
    fig, ax = plt.subplots(figsize=(8, 5))
    (summary.T / summary.sum(axis=1)).T.plot(kind="bar", stacked=True, ax=ax, color=["#d62728", "#1f77b4", "#2ca02c"])
    ax.set_ylabel("Share of Teams")
    ax.set_xlabel("Archetype")
    ax.set_title("Prediction Error Breakdown by Archetype")
    ax.legend(title="Error", loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "poster_archetype_error.png", dpi=300)
    plt.close(fig)



def ordinal_error_heatmaps(model_df: pd.DataFrame, logistic_predictions: pd.DataFrame) -> None:
    if logistic_predictions.empty:
        return
    df_error = _prepare_error_dataframe(model_df, logistic_predictions)
    rounds = sorted(df_error['playoff_round'].unique())
    preds = sorted(df_error['pred_round'].unique())
    counts = df_error.pivot_table(index='playoff_round', columns='pred_round', values='error', aggfunc='size', fill_value=0).reindex(index=rounds, columns=preds, fill_value=0)
    penalty = df_error.assign(abs_error=lambda d: d['error'].abs()).pivot_table(index='playoff_round', columns='pred_round', values='abs_error', aggfunc='sum', fill_value=0).reindex(index=rounds, columns=preds, fill_value=0)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    sns.heatmap(counts, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_xlabel('Predicted Round')
    axes[0].set_ylabel('Actual Round')
    axes[0].set_title('Prediction Counts')
    sns.heatmap(penalty, annot=True, fmt='.1f', cmap='Reds', ax=axes[1])
    axes[1].set_xlabel('Predicted Round')
    axes[1].set_title('Weighted Absolute Error (Sum of |Pred-Actual|)')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'poster_error_penalty_heatmap.png', dpi=300)
    plt.close(fig)


def ordinal_error_distribution(model_df: pd.DataFrame, logistic_predictions: pd.DataFrame) -> None:
    if logistic_predictions.empty:
        return
    df_error = _prepare_error_dataframe(model_df, logistic_predictions)
    df_error['abs_error'] = df_error['error'].abs()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=df_error, x='playoff_round', y='abs_error', ax=ax)
    sns.stripplot(data=df_error, x='playoff_round', y='abs_error', color='black', alpha=0.3, size=3, ax=ax)
    ax.set_xlabel('Actual Playoff Round')
    ax.set_ylabel('|Predicted - Actual|')
    ax.set_title('Absolute Prediction Error by Actual Round')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'poster_error_abs_distribution.png', dpi=300)
    plt.close(fig)

    abs_counts = df_error['abs_error'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(x=abs_counts.index, y=abs_counts.values, palette='viridis', ax=ax)
    ax.set_xlabel('|Predicted - Actual|')
    ax.set_ylabel('Number of Teams')
    ax.set_title('Absolute Error Distance Distribution')
    for x, y in zip(abs_counts.index, abs_counts.values):
        ax.text(x, y + 0.3, str(int(y)), ha='center')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'poster_error_distance_counts.png', dpi=300)
    plt.close(fig)

    signed_counts = df_error['error'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=signed_counts.index, y=signed_counts.values, palette='coolwarm', ax=ax)
    ax.set_xlabel('Prediction - Actual (Signed Error)')
    ax.set_ylabel('Number of Teams')
    ax.set_title('Signed Error Distance Distribution')
    for x, y in zip(signed_counts.index, signed_counts.values):
        ax.text(x, y + 0.3, str(int(y)), ha='center')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'poster_error_signed_counts.png', dpi=300)
    plt.close(fig)

    summary = abs_counts.rename('count').reset_index().rename(columns={'index': 'abs_error'})
    summary.to_csv(FIG_DIR / 'error_distance_counts.csv', index=False)


def shock_vs_net_rating_scatter(model_df: pd.DataFrame, logistic_predictions: pd.DataFrame) -> None:
    if logistic_predictions.empty:
        return
    df_error = _prepare_error_dataframe(model_df, logistic_predictions)
    palette = {"Under": "#d62728", "Exact": "#1f77b4", "Over": "#2ca02c"}
    combos = [
        ("shock_star_norm", "Star Shock / |Net Rating|", "Team Net Rating", "poster_shock_vs_net_rating.png"),
        ("shock_role_norm", "Role Shock / |Net Rating|", "Team Net Rating", "poster_role_shock_vs_net_rating.png"),
        ("shock_connector_norm", "Connector Shock / |Net Rating|", "Team Net Rating", "poster_connector_shock_vs_net_rating.png"),
    ]
    for y_col, y_label, x_label, filename in combos:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            data=df_error,
            x="team_net_rating",
            y=y_col,
            hue="error_cat",
            palette=palette,
            ax=ax,
        )
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.axvline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"{y_label} vs. {x_label}")
        fig.tight_layout()
        fig.savefig(FIG_DIR / filename, dpi=300)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
    for ax, (y_col, y_label, _, _) in zip(axes, combos):
        sns.scatterplot(
            data=df_error,
            x="team_net_rating",
            y=y_col,
            hue="error_cat",
            palette=palette,
            ax=ax,
            legend=False,
        )
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.axvline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("Team Net Rating")
        ax.set_ylabel(y_label)
        ax.set_title(y_label)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Error", loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "poster_shock_vs_net_rating_grid.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=df_error,
        x="salary_assortativity",
        y="shock_star_norm",
        hue="error_cat",
        palette=palette,
        ax=ax,
    )
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Salary Assortativity")
    ax.set_ylabel("Star Shock / |Net Rating|")
    ax.set_title("Star Shock vs. Salary Assortativity")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "poster_shock_vs_assortativity.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()

