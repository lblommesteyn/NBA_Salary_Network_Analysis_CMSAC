"""
Generate poster-ready figures using enhanced predictions:
- Confusion matrix from enhanced predictions
- Confusion-by-archetype bar chart (Under/Exact/Over by star-centric flag)
- Calibrated reliability curve (saved under poster naming)

Run: python enhanced_poster_plots.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.isotonic import IsotonicRegression

RANDOM_STATE = 42
FIG_DIR = Path("poster_figures")
FIG_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")

# Conference membership (include alternate codes where applicable)
EAST = {
    "ATL", "BOS", "BRK", "BKN", "CHA", "CHO", "CHI", "CLE", "DET", "IND",
    "MIA", "MIL", "NYK", "ORL", "PHI", "TOR", "WAS",
}
WEST = {
    "DAL", "DEN", "GSW", "HOU", "LAC", "LAL", "MEM", "MIN", "NOP", "OKC",
    "PHX", "PHO", "POR", "SAC", "SAS", "UTA",
}


def _load_primary_predictions() -> pd.DataFrame:
    # Prefer Two-Stage, else HGB, else Logistic
    candidates = [
        "enhanced_predictions_Two-Stage_HGB.csv",
        "enhanced_predictions_HistGradientBoosting.csv",
        "enhanced_predictions_L2_Logistic.csv",
    ]
    for name in candidates:
        path = FIG_DIR / name
        if path.exists():
            df = pd.read_csv(path)
            if not df.empty:
                return df
    return pd.DataFrame()


essential_cols = [
    "season", "team", "n_players", "degree_centralization", "edge_concentration_top5",
    "modularity_Q", "community_size_cv", "minutes_gini", "minutes_assortativity",
    "salary_gini", "salary_assortativity", "salary_top1_share", "RRS",
]


def _load_feature_frame_for_archetypes() -> pd.DataFrame:
    base = (
        pd.read_csv("vis_clean_data/features_from_pos_lineups.csv")
        .sort_values(["season", "team", "n_players"], ascending=[True, True, False])
        .drop_duplicates(subset=["season", "team"], keep="first")
    )
    salary = pd.read_csv("vis_clean_data/salary_features.csv")
    rrs = (
        pd.read_csv("vis_clean_data/rrs_from_pos_lineups.csv")
        .sort_values("RRS", ascending=False)
        .drop_duplicates(subset=["season", "team"], keep="first")
    )
    df = base.merge(salary, on=["season", "team"], how="inner").merge(rrs, on=["season", "team"], how="left")

    # Recompute shocks needed for clustering parity
    for col, new_col in [("drop_star", "shock_star"), ("drop_role", "shock_role"), ("drop_connector", "shock_connector")]:
        df[new_col] = df["nr_hat_intact"] - df[col]

    # KMeans to get star-centric cluster (same features and random_state as in enhanced_modeling)
    cluster_features = [
        "degree_centralization", "edge_concentration_top5", "modularity_Q", "community_size_cv",
        "salary_gini", "salary_assortativity", "RRS", "shock_star", "shock_role", "shock_connector",
    ]
    mask = df[cluster_features].notna().all(axis=1)
    df["cluster_id"] = -1
    df["is_star_cluster"] = 0
    if mask.any():
        Xk = StandardScaler().fit_transform(df.loc[mask, cluster_features])
        km = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=20).fit(Xk)
        df.loc[mask, "cluster_id"] = km.labels_
        profile = df.loc[mask, cluster_features].copy()
        profile["cluster"] = km.labels_
        summary = profile.groupby("cluster").mean(numeric_only=True)
        star_cluster = int(summary["degree_centralization"].idxmax())
        df["is_star_cluster"] = (df["cluster_id"] == star_cluster).astype(int)
    return df[["season", "team", "is_star_cluster"]]


def _load_seed_proxy_frame() -> pd.DataFrame:
    base = (
        pd.read_csv("vis_clean_data/features_from_pos_lineups.csv")
        .sort_values(["season", "team", "n_players"], ascending=[True, True, False])
        .drop_duplicates(subset=["season", "team"], keep="first")
    )
    df = base[["season", "team", "team_net_rating"]].copy()
    df["conference"] = np.select(
        [df["team"].isin(list(EAST)), df["team"].isin(list(WEST))],
        ["East", "West"],
        default="Unknown",
    )
    # Seed proxy: rank within season × conference (1 is best)
    df["seed_proxy"] = (
        df.groupby(["season", "conference"])  # type: ignore
        ["team_net_rating"].rank(ascending=False, method="first")
    )
    # Deciles within season × conference (1 best seeds)
    grp = df.groupby(["season", "conference"])  # type: ignore
    sizes = grp["team"].transform("size")
    df["seed_decile"] = 1 + np.floor((df["seed_proxy"] - 1) * 10 / sizes)
    df["seed_decile"] = df["seed_decile"].clip(1, 10).astype(int)
    df["seed_group"] = np.where(df["seed_decile"] <= 3, "Top", np.where(df["seed_decile"] >= 8, "Bottom", "Middle"))
    return df[["season", "team", "seed_proxy", "seed_decile", "seed_group"]]


def plot_confusion_from_preds(preds: pd.DataFrame, out_name: str) -> None:
    if preds.empty:
        return
    y_true = preds["playoff_round"].astype(int).to_numpy()
    y_pred = preds["pred_round"].astype(int).to_numpy()
    labels = np.sort(np.unique(np.concatenate([y_true, y_pred])))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted Playoff Round")
    ax.set_ylabel("Actual Playoff Round")
    ax.set_title("Enhanced Model Confusion Matrix (Counts shown, color = row-normalised)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / out_name, dpi=300)
    plt.close(fig)


def plot_calibration_by_archetype(preds: pd.DataFrame, out_name: str) -> None:
    if preds.empty:
        return
    feat = _load_feature_frame_for_archetypes()
    df = preds.merge(feat, on=["season", "team"], how="left")
    prob_col = "prob_ge2_iso" if ("prob_ge2_iso" in df.columns and df["prob_ge2_iso"].notna().any()) else "prob_ge2"
    df = df.dropna(subset=[prob_col]).copy()
    if df.empty:
        return

    groups = {0: "Other", 1: "Star"}
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
    for key, label in groups.items():
        sub = df[df["is_star_cluster"] == key].copy()
        if sub.empty:
            continue
        sub["bin"] = pd.cut(sub[prob_col], bins=np.linspace(0, 1, 9), include_lowest=True)
        grouped = sub.groupby("bin").agg(
            predicted=(prob_col, "mean"),
            observed=("playoff_round", lambda s: (s >= 2).mean()),
            count=(prob_col, "size"),
        )
        ax.plot(grouped["predicted"], grouped["observed"], marker="o", label=label)
    ax.set_xlabel("Predicted Probability (Advance >= 2 Rounds)")
    ax.set_ylabel("Observed Frequency")
    ax.set_title("Calibration by Archetype")
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIG_DIR / out_name, dpi=300)
    plt.close(fig)


def plot_seed_decile_error_chart(preds: pd.DataFrame, out_name: str) -> None:
    if preds.empty:
        return
    seed = _load_seed_proxy_frame()
    df = preds.merge(seed, on=["season", "team"], how="left")
    df["err"] = np.sign(df["pred_round"].astype(int) - df["playoff_round"].astype(int))
    df["err_cat"] = df["err"].map({-1: "Under", 0: "Exact", 1: "Over"}).fillna("Exact")
    tally = df.groupby(["seed_decile", "err_cat"]).size().reset_index(name="count")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=tally, x="seed_decile", y="count", hue="err_cat", palette="Set2", ax=ax)
    ax.set_xlabel("Seed Decile (1 = Top Seeds)")
    ax.set_ylabel("Team-Seasons")
    ax.set_title("Prediction Error by Seed Decile")
    ax.legend(title="Category")
    fig.tight_layout()
    fig.savefig(FIG_DIR / out_name, dpi=300)
    plt.close(fig)


def plot_calibration_by_seed_group(preds: pd.DataFrame, out_name: str) -> None:
    if preds.empty:
        return
    seed = _load_seed_proxy_frame()
    df = preds.merge(seed, on=["season", "team"], how="left")
    prob_col = "prob_ge2_iso" if ("prob_ge2_iso" in df.columns and df["prob_ge2_iso"].notna().any()) else "prob_ge2"
    df = df.dropna(subset=[prob_col]).copy()
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
    for group in ["Top", "Middle", "Bottom"]:
        sub = df[df["seed_group"] == group].copy()
        if sub.empty:
            continue
        sub["bin"] = pd.cut(sub[prob_col], bins=np.linspace(0, 1, 9), include_lowest=True)
        grouped = sub.groupby("bin").agg(
            predicted=(prob_col, "mean"),
            observed=("playoff_round", lambda s: (s >= 2).mean()),
            count=(prob_col, "size"),
        )
        ax.plot(grouped["predicted"], grouped["observed"], marker="o", label=group)
    ax.set_xlabel("Predicted Probability (Advance >= 2 Rounds)")
    ax.set_ylabel("Observed Frequency")
    ax.set_title("Calibration by Seed Group")
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIG_DIR / out_name, dpi=300)
    plt.close(fig)


def plot_confusion_by_archetype(preds: pd.DataFrame, out_name: str) -> None:
    if preds.empty:
        return
    feat = _load_feature_frame_for_archetypes()
    df = preds.merge(feat, on=["season", "team"], how="left")
    df["err"] = np.sign(df["pred_round"].astype(int) - df["playoff_round"].astype(int))
    df["err_cat"] = df["err"].map({-1: "Under", 0: "Exact", 1: "Over"}).fillna("Exact")

    tally = df.groupby(["is_star_cluster", "err_cat"]).size().reset_index(name="count")
    # Ensure all combinations exist
    idx = pd.MultiIndex.from_product([[0, 1], ["Under", "Exact", "Over"]], names=["is_star_cluster", "err_cat"])
    tally = tally.set_index(["is_star_cluster", "err_cat"]).reindex(idx, fill_value=0).reset_index()

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(data=tally, x="err_cat", y="count", hue="is_star_cluster", palette="Set2", ax=ax)
    ax.set_xlabel("Prediction Category")
    ax.set_ylabel("Team-Seasons")
    ax.set_title("Under/Exact/Over by Archetype (0=Balanced, 1=Star-Centered)")
    ax.legend(title="Star-Centered")
    fig.tight_layout()
    fig.savefig(FIG_DIR / out_name, dpi=300)
    plt.close(fig)


def plot_calibration_from_preds(preds: pd.DataFrame, out_name: str) -> None:
    df = preds.dropna(subset=["prob_ge2"]).copy()
    if df.empty:
        return
    prob_col = "prob_ge2_iso" if ("prob_ge2_iso" in df.columns and df["prob_ge2_iso"].notna().any()) else "prob_ge2"
    y_true = (df["playoff_round"] >= 2).astype(int).to_numpy()
    prob = df[prob_col].to_numpy()

    # Bin
    df["bin"] = pd.cut(df[prob_col], bins=np.linspace(0, 1, 9), include_lowest=True)
    grouped = df.groupby("bin").agg(
        predicted=(prob_col, "mean"),
        observed=("playoff_round", lambda s: (s >= 2).mean()),
        count=(prob_col, "size"),
    )

    # Fit post-hoc isotonic for a smooth curve
    try:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(prob, y_true)
        df["prob_iso"] = iso.transform(prob)
        df["bin_iso"] = pd.cut(df["prob_iso"], bins=np.linspace(0, 1, 9), include_lowest=True)
        grouped_iso = df.groupby("bin_iso").agg(
            predicted=("prob_iso", "mean"),
            observed=("playoff_round", lambda s: (s >= 2).mean()),
        )
    except Exception:
        grouped_iso = None

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    ax.scatter(grouped["predicted"], grouped["observed"], s=grouped["count"] * 5, color="#1f77b4", label="Enhanced Model")
    if grouped_iso is not None:
        ax.plot(grouped_iso["predicted"], grouped_iso["observed"], color="#ff7f0e", marker="o", label="Isotonic (viz)")
    for (_, row) in grouped.iterrows():
        ax.text(row["predicted"], row["observed"], str(int(row["count"])), fontsize=10, ha="center", va="bottom")
    ax.set_xlabel("Predicted Probability (Advance >= 2 Rounds)")
    ax.set_ylabel("Observed Frequency")
    ax.set_title("Enhanced Model Reliability")
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIG_DIR / out_name, dpi=300)
    plt.close(fig)


def main() -> None:
    preds = _load_primary_predictions()
    if preds.empty:
        print("No enhanced predictions found.")
        return
    # Confusion
    plot_confusion_from_preds(preds, out_name="poster_confusion_matrix_enhanced.png")
    # Confusion by archetype
    plot_confusion_by_archetype(preds, out_name="poster_confusion_by_archetype.png")
    # Calibration under poster naming
    plot_calibration_from_preds(preds, out_name="poster_enhanced_calibration.png")
    # New diagnostics
    plot_calibration_by_archetype(preds, out_name="poster_archetype_calibration.png")
    plot_seed_decile_error_chart(preds, out_name="poster_seed_decile_error.png")
    plot_calibration_by_seed_group(preds, out_name="poster_seed_group_calibration.png")


if __name__ == "__main__":
    main()
