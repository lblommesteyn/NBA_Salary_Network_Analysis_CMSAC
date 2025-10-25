"""
Enhanced modeling script implementing data-driven fixes:
- Engineered features (normalized shocks, interactions, mesh surplus, archetype)
- HistGradientBoosting as primary with isotonic calibration
- Two-stage modeling (baseline ability -> topology uplift)
- Class/sample weighting to reduce underprediction of deep runs
- Monitoring: error summaries and calibration plots

Outputs written to poster_figures/.
Run: python enhanced_modeling.py
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

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


@dataclass
class EnhancedData:
    df: pd.DataFrame


def load_datasets_enhanced() -> EnhancedData:
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

    playoff = pd.read_csv("vis_clean_data/playoff_rounds.csv")

    df = (
        base.merge(salary, on=["season", "team"], how="inner")
        .merge(rrs, on=["season", "team"], how="left")
        .merge(playoff, on=["season", "team"], how="inner")
    )
    df["playoff_round"] = df["playoff_round"].astype(int)

    # Shock magnitudes from drops
    for col, new_col in [
        ("drop_star", "shock_star"),
        ("drop_role", "shock_role"),
        ("drop_connector", "shock_connector"),
    ]:
        df[new_col] = df["nr_hat_intact"] - df[col]

    # Merge salary assortativity permutation stats (and controls if present)
    perm = pd.read_csv("vis_clean_data/salary_assortativity_perm_test.csv")
    cols_keep = ["season", "team", "assort_z", "assort_salary_obs", "assort_mu_null"]
    for extra in ["n_nodes", "n_edges"]:
        if extra in perm.columns:
            cols_keep.append(extra)
    df = df.merge(
        perm[cols_keep],
        on=["season", "team"],
        how="left",
    )
    df["salary_assort_z"] = df["assort_z"]
    df["mesh_surplus"] = df["assort_salary_obs"] - df["assort_mu_null"]
    if "n_nodes" in df.columns:
        df["assort_n_nodes"] = df["n_nodes"]
    if "n_edges" in df.columns:
        df["assort_n_edges"] = df["n_edges"]

    # Shock normalizations and transforms
    denom_nr = (df["team_net_rating"].abs() + 1.0).clip(lower=1.0)
    df["shock_star_per_nr"] = df["shock_star"] / denom_nr
    df["shock_star_per_salary_top1"] = df["shock_star"] / (df["salary_top1_share"].clip(lower=0.05))
    df["shock_star_log"] = np.log1p(np.maximum(df["shock_star"], 0))
    df["shock_star_cap"] = df["shock_star"].clip(upper=df["shock_star"].quantile(0.95))
    # Mirror transforms for role and connector shocks
    df["shock_role_per_nr"] = df["shock_role"] / denom_nr
    df["shock_role_log"] = np.log1p(np.maximum(df["shock_role"], 0))
    df["shock_role_cap"] = df["shock_role"].clip(upper=df["shock_role"].quantile(0.95))
    df["shock_connector_per_nr"] = df["shock_connector"] / denom_nr
    df["shock_connector_log"] = np.log1p(np.maximum(df["shock_connector"], 0))
    df["shock_connector_cap"] = df["shock_connector"].clip(upper=df["shock_connector"].quantile(0.95))

    # Interactions capturing superstar punch
    df["nr_x_top1_salary"] = df["team_net_rating"] * df["salary_top1_share"]
    df["nr_x_one_minus_assort"] = df["team_net_rating"] * (1 - df["salary_assortativity"])
    if "minutes_top1_share" in df.columns:
        df["nr_x_minutes_top1"] = df["team_net_rating"] * df["minutes_top1_share"]
    else:
        df["nr_x_minutes_top1"] = np.nan
    # Multi-star interactions
    if "salary_top2_share" in df.columns:
        df["nr_x_top2_salary"] = df["team_net_rating"] * df["salary_top2_share"]
    else:
        df["nr_x_top2_salary"] = np.nan
    if "salary_top3_share" in df.columns:
        df["nr_x_top3_salary"] = df["team_net_rating"] * df["salary_top3_share"]
    else:
        df["nr_x_top3_salary"] = np.nan
    # Minutes depth interactions (if available)
    if "minutes_top2_share" in df.columns:
        df["nr_x_minutes_top2"] = df["team_net_rating"] * df["minutes_top2_share"]
    else:
        df["nr_x_minutes_top2"] = np.nan
    if "minutes_top3_share" in df.columns:
        df["nr_x_minutes_top3"] = df["team_net_rating"] * df["minutes_top3_share"]
    else:
        df["nr_x_minutes_top3"] = np.nan

    # Star availability proxy and injury-blended expected rating
    if "minutes_top1_share" in df.columns and "nr_hat_intact" in df.columns and "drop_star" in df.columns:
        # Season-specific healthy baseline for top-1 minutes share (85th percentile)
        try:
            baseline = df.groupby("season")["minutes_top1_share"].transform(lambda s: s.quantile(0.85))
        except Exception:
            baseline = 0.22  # fallback constant if grouping fails
        denom = pd.Series(baseline).replace(0, np.nan)
        p_avail = (df["minutes_top1_share"] / denom).clip(0, 1)
        p_avail = p_avail.fillna(p_avail.median() if np.isfinite(p_avail.median()) else 0.6)
        df["star_p_avail_proxy"] = p_avail
        df["nr_expected_star_availability"] = p_avail * df["nr_hat_intact"] + (1 - p_avail) * df["drop_star"]
    else:
        df["star_p_avail_proxy"] = np.nan
        df["nr_expected_star_availability"] = np.nan

    # Seed proxy by conference (rank by net rating within season + conference)
    df["conference"] = np.select(
        [df["team"].isin(list(EAST)), df["team"].isin(list(WEST))],
        ["East", "West"],
        default="Unknown",
    )
    try:
        df["seed_proxy"] = (
            df.groupby(["season", "conference"])  # type: ignore
            ["team_net_rating"].rank(ascending=False, method="first")
        )
    except Exception:
        df["seed_proxy"] = np.nan

    # Depth ratio: top10 vs top5 edge concentration
    if "edge_concentration_top10" in df.columns and "edge_concentration_top5" in df.columns:
        denom = df["edge_concentration_top5"].replace(0, np.nan)
        df["edge_concentration_top10_to_top5"] = df["edge_concentration_top10"] / denom

    # Archetype via KMeans clustering (3 clusters), plus star-centric flag
    cluster_features = [
        "degree_centralization",
        "edge_concentration_top5",
        "edge_concentration_top10",
        "modularity_Q",
        "community_size_cv",
        "salary_gini",
        "salary_assortativity",
        "RRS",
        "shock_star",
        "shock_role",
        "shock_connector",
    ]
    cluster_mask = df[cluster_features].notna().all(axis=1)
    df["cluster_id"] = -1
    if cluster_mask.any():
        Xk = StandardScaler().fit_transform(df.loc[cluster_mask, cluster_features])
        km = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=20).fit(Xk)
        df.loc[cluster_mask, "cluster_id"] = km.labels_
        profile_df = df.loc[cluster_mask, cluster_features].copy()
        profile_df["cluster"] = km.labels_
        cluster_summary = profile_df.groupby("cluster").mean(numeric_only=True)
        star_cluster = int(cluster_summary["degree_centralization"].idxmax())
        df["is_star_cluster"] = (df["cluster_id"] == star_cluster).astype(int)
    else:
        df["is_star_cluster"] = 0

    # High shock & high talent flag
    q70 = df["shock_star_per_nr"].quantile(0.70)
    df["high_shock_high_talent"] = ((df["team_net_rating"] > 5) & (df["shock_star_per_nr"] > q70)).astype(int)

    return EnhancedData(df=df)


def _calc_fold_metrics(y_test: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro"),
    }


def evaluate_models_enhanced(model_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Leave-one-season-out across candidate models; returns metrics and predictions.

    Returns:
      metrics_df: summary across folds (rows per model per fold)
      predictions_by_model: model -> DataFrame(season, team, playoff_round, pred_round, prob_ge2, prob_ge2_iso)
      fold_details: model -> per-fold metrics DataFrame
    """
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
        # engineered
        "salary_assort_z",
        "mesh_surplus",
        "assort_n_nodes",
        "assort_n_edges",
        "shock_star_per_nr",
        "shock_star_per_salary_top1",
        "shock_star_log",
        "shock_star_cap",
        "shock_role_per_nr",
        "shock_role_log",
        "shock_role_cap",
        "shock_connector_per_nr",
        "shock_connector_log",
        "shock_connector_cap",
        "nr_x_top1_salary",
        "nr_x_top2_salary",
        "nr_x_top3_salary",
        "nr_x_one_minus_assort",
        "nr_x_minutes_top1",
        "nr_x_minutes_top2",
        "nr_x_minutes_top3",
        "edge_concentration_top10",
        "edge_concentration_top10_to_top5",
        "seed_proxy",
        "star_p_avail_proxy",
        "nr_expected_star_availability",
        "cluster_id",
        "is_star_cluster",
        "high_shock_high_talent",
    ]

    available = [c for c in feature_cols if c in model_df.columns]
    df = model_df.dropna(subset=available + ["playoff_round"]).reset_index(drop=True)
    X = df[available]
    y = df["playoff_round"].to_numpy()
    seasons = df["season"].to_numpy()

    models: Dict[str, object] = {
        "L2 Logistic": (
            "logit",
            LogisticRegression(
                penalty="l2",
                C=1.2,
                solver="lbfgs",
                max_iter=5000,
                multi_class="multinomial",
                random_state=RANDOM_STATE,
            ),
        ),
        "HistGradientBoosting": (
            "hgb",
            HistGradientBoostingClassifier(
                learning_rate=0.08,
                max_depth=4,
                max_iter=400,
                min_samples_leaf=8,
                random_state=RANDOM_STATE,
            ),
        ),
    }

    records: List[Dict[str, float]] = []
    predictions_by_model_raw: Dict[str, List[pd.DataFrame]] = {name: [] for name in models}
    fold_details: Dict[str, List[Dict[str, float]]] = {name: [] for name in models}

    seasons_unique = sorted(pd.unique(seasons))
    classes = np.sort(pd.unique(y))

    # Helper for sample weighting
    weight_map = {0: 1.0, 1: 1.2, 2: 1.6, 3: 2.2, 4: 3.0}
    sample_weight_all = np.array([weight_map.get(int(c), 1.0) for c in y])

    for model_name, (kind, clf) in models.items():
        for holdout in seasons_unique:
            train_mask = seasons != holdout
            test_mask = ~train_mask

            X_train, X_test = X.loc[train_mask], X.loc[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]
            if X_test.empty:
                continue

            # Fit with weighting where supported
            try:
                if kind == "logit":
                    scaler = StandardScaler().fit(X_train)
                    X_train_s = scaler.transform(X_train)
                    X_test_s = scaler.transform(X_test)
                    clf.fit(X_train_s, y_train, sample_weight=sample_weight_all[train_mask])
                    y_pred = clf.predict(X_test_s)
                else:
                    clf.fit(X_train, y_train, sample_weight=sample_weight_all[train_mask])
                    y_pred = clf.predict(X_test)
            except TypeError:
                # Fall back without weights
                if kind == "logit":
                    scaler = StandardScaler().fit(X_train)
                    X_train_s = scaler.transform(X_train)
                    X_test_s = scaler.transform(X_test)
                    clf.fit(X_train_s, y_train)
                    y_pred = clf.predict(X_test_s)
                else:
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)

            # Metrics
            m = _calc_fold_metrics(y_test, y_pred)
            m.update({"season": holdout, "model": model_name})

            # Probabilities and calibration
            prob_df = df.loc[test_mask, ["season", "team", "playoff_round"]].copy()
            aligned_proba = None
            ge2_idx: List[int] = []

            if kind == "logit":
                proba = clf.predict_proba(X_test_s)
                model_classes = clf.classes_
            else:
                proba = clf.predict_proba(X_test)
                model_classes = clf.classes_

            aligned_classes = [c for c in classes if c in model_classes]
            reorder_idx = [np.where(model_classes == c)[0][0] for c in aligned_classes]
            aligned_proba = proba[:, reorder_idx]
            y_bin = label_binarize(y_test, classes=aligned_classes)
            if y_bin.shape[1] == aligned_proba.shape[1]:
                m["brier"] = np.mean(np.sum((y_bin - aligned_proba) ** 2, axis=1))
                try:
                    m["roc_auc"] = roc_auc_score(
                        y_test,
                        aligned_proba,
                        labels=aligned_classes,
                        multi_class="ovo",
                        average="macro",
                    )
                except ValueError:
                    m["roc_auc"] = np.nan

            ge2_idx = [np.where(model_classes == c)[0][0] for c in model_classes if c >= 2]
            prob_df["pred_round"] = y_pred
            prob_df["prob_ge2"] = aligned_proba[:, ge2_idx].sum(axis=1) if ge2_idx else np.nan

            # Fit isotonic on train fold and apply to test
            try:
                if kind == "logit":
                    train_proba = clf.predict_proba(X_train_s)
                else:
                    train_proba = clf.predict_proba(X_train)
                train_aligned = train_proba[:, reorder_idx]
                ge2_train = train_aligned[:, ge2_idx].sum(axis=1) if ge2_idx else np.zeros(len(train_aligned))
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(ge2_train, (y_train >= 2).astype(int))
                prob_df["prob_ge2_iso"] = iso.transform(prob_df["prob_ge2"]) if prob_df["prob_ge2"].notna().any() else np.nan
            except Exception:
                prob_df["prob_ge2_iso"] = np.nan

            records.append(m)
            fold_details[model_name].append(m)
            predictions_by_model_raw[model_name].append(prob_df)

    # Two-Stage HGB (baseline ER + uplift)
    two_stage_name = "Two-Stage HGB"
    predictions_by_model_raw.setdefault(two_stage_name, [])
    fold_details.setdefault(two_stage_name, [])

    baseline_cols = [c for c in ["team_net_rating", "minutes_gini", "minutes_top1_share"] if c in df.columns]
    if len(baseline_cols) == 3:
        for holdout in seasons_unique:
            train_mask = seasons != holdout
            test_mask = ~train_mask
            if not np.any(test_mask):
                continue

            X_train_full, X_test_full = X.loc[train_mask], X.loc[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]

            base_clf = LogisticRegression(C=1.2, penalty="l2", solver="lbfgs", multi_class="multinomial", max_iter=5000, random_state=RANDOM_STATE)
            scaler = StandardScaler().fit(df.loc[train_mask, baseline_cols])
            base_train = scaler.transform(df.loc[train_mask, baseline_cols])
            base_test = scaler.transform(df.loc[test_mask, baseline_cols])
            base_clf.fit(base_train, y_train)
            base_proba_train = base_clf.predict_proba(base_train)
            base_proba_test = base_clf.predict_proba(base_test)
            er_classes = base_clf.classes_
            er_train = (base_proba_train * er_classes).sum(axis=1)
            er_test = (base_proba_test * er_classes).sum(axis=1)

            X_train_2 = X_train_full.copy()
            X_test_2 = X_test_full.copy()
            X_train_2["baseline_er"] = er_train
            X_test_2["baseline_er"] = er_test

            hgb2 = HistGradientBoostingClassifier(learning_rate=0.08, max_depth=4, max_iter=400, min_samples_leaf=8, random_state=RANDOM_STATE)
            sw_train = sample_weight_all[train_mask]
            try:
                hgb2.fit(X_train_2, y_train, sample_weight=sw_train)
            except TypeError:
                hgb2.fit(X_train_2, y_train)

            y_pred = hgb2.predict(X_test_2)
            proba = hgb2.predict_proba(X_test_2)

            m = _calc_fold_metrics(y_test, y_pred)
            m.update({"season": holdout, "model": two_stage_name})

            aligned_classes = [c for c in classes if c in hgb2.classes_]
            reorder_idx = [np.where(hgb2.classes_ == c)[0][0] for c in aligned_classes]
            aligned_proba = proba[:, reorder_idx]
            try:
                y_bin = label_binarize(y_test, classes=aligned_classes)
                if y_bin.shape[1] == aligned_proba.shape[1]:
                    m["brier"] = np.mean(np.sum((y_bin - aligned_proba) ** 2, axis=1))
                    m["roc_auc"] = roc_auc_score(y_test, aligned_proba, labels=aligned_classes, multi_class="ovo", average="macro")
            except Exception:
                pass

            prob_df = df.loc[test_mask, ["season", "team", "playoff_round"]].copy()
            ge2_idx = [np.where(hgb2.classes_ == c)[0][0] for c in hgb2.classes_ if c >= 2]
            prob_df["pred_round"] = y_pred
            prob_df["prob_ge2"] = aligned_proba[:, ge2_idx].sum(axis=1) if ge2_idx else np.nan

            try:
                train_proba = hgb2.predict_proba(X_train_2)
                train_aligned = train_proba[:, reorder_idx]
                ge2_train = train_aligned[:, ge2_idx].sum(axis=1) if ge2_idx else np.zeros(len(train_aligned))
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(ge2_train, (y_train >= 2).astype(int))
                prob_df["prob_ge2_iso"] = iso.transform(prob_df["prob_ge2"]) if prob_df["prob_ge2"].notna().any() else np.nan
            except Exception:
                prob_df["prob_ge2_iso"] = np.nan

            records.append(m)
            fold_details[two_stage_name].append(m)
            predictions_by_model_raw[two_stage_name].append(prob_df)

    metrics_df = pd.DataFrame(records)
    predictions_by_model = {
        name: (pd.concat(items, ignore_index=True).sort_values(["season", "team"]) if items else pd.DataFrame())
        for name, items in predictions_by_model_raw.items()
    }
    fold_details_df = {name: pd.DataFrame(rows) for name, rows in fold_details.items() if rows}

    return metrics_df, predictions_by_model, fold_details_df


def plot_model_comparison(metrics_df: pd.DataFrame, out_name: str = "enhanced_model_comparison.png") -> None:
    if metrics_df.empty:
        return
    df = metrics_df.copy()
    # Aggregate per-model means
    agg = df.groupby("model", as_index=False).agg(
        accuracy_mean=("accuracy", "mean"),
        macro_f1_mean=("macro_f1", "mean"),
        roc_auc_mean=("roc_auc", "mean"),
        brier_mean=("brier", "mean"),
    )
    agg = agg.sort_values("macro_f1_mean", ascending=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
    palette = sns.color_palette("viridis", n_colors=len(agg))

    sns.barplot(data=agg, x="macro_f1_mean", y="model", ax=axes[0], palette=palette)
    axes[0].set_xlabel("Macro-F1 (Mean)")
    axes[0].set_ylabel("")

    sns.barplot(data=agg, x="roc_auc_mean", y="model", ax=axes[1], palette=palette)
    axes[1].set_xlabel("ROC-AUC (OVO)")
    axes[1].set_ylabel("")

    sns.barplot(data=agg, x="brier_mean", y="model", ax=axes[2], palette=palette)
    axes[2].set_xlabel("Brier (lower better)")
    axes[2].set_ylabel("")

    fig.suptitle("Enhanced Model Comparison (LOSO)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / out_name, dpi=300)
    plt.close(fig)


def plot_calibration(preds: pd.DataFrame, out_name: str) -> None:
    df = preds.dropna(subset=["prob_ge2"]).copy()
    if df.empty:
        return

    y_true = (df["playoff_round"] >= 2).astype(int).to_numpy()
    prob_col = "prob_ge2_iso" if "prob_ge2_iso" in df.columns and df["prob_ge2_iso"].notna().any() else "prob_ge2"
    prob = df[prob_col].to_numpy()

    # Binning
    df["bin"] = pd.cut(df[prob_col], bins=np.linspace(0, 1, 9), include_lowest=True)
    grouped = df.groupby("bin").agg(
        predicted=(prob_col, "mean"),
        observed=("playoff_round", lambda s: (s >= 2).mean()),
        count=(prob_col, "size"),
    )

    # Post-hoc isotonic (for visualization only)
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
    ax.scatter(grouped["predicted"], grouped["observed"], s=grouped["count"] * 5, color="#1f77b4", label="Model")
    if grouped_iso is not None:
        ax.plot(grouped_iso["predicted"], grouped_iso["observed"], color="#ff7f0e", marker="o", label="Isotonic (viz)")
    for (_, row) in grouped.iterrows():
        ax.text(row["predicted"], row["observed"], str(int(row["count"])), fontsize=10, ha="center", va="bottom")
    ax.set_xlabel("Predicted Probability (Advance >= 2 Rounds)")
    ax.set_ylabel("Observed Frequency")
    ax.set_title("Reliability of Deep-Run Probabilities")
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIG_DIR / out_name, dpi=300)
    plt.close(fig)


def compute_error_summary(preds: pd.DataFrame, full_df: pd.DataFrame) -> pd.DataFrame:
    if preds.empty:
        return pd.DataFrame()
    df = preds.merge(full_df[["season", "team", "team_net_rating", "shock_star", "shock_star_per_nr", "is_star_cluster"]], on=["season", "team"], how="left")
    df["err"] = np.sign(df["pred_round"].astype(int) - df["playoff_round"].astype(int))
    df["err_cat"] = df["err"].map({-1: "Under", 0: "Exact", 1: "Over"}).fillna("Exact")

    summary = df.groupby("err_cat").agg(
        count=("team", "size"),
        team_net_rating_mean=("team_net_rating", "mean"),
        shock_star_mean=("shock_star", "mean"),
        shock_star_per_nr_mean=("shock_star_per_nr", "mean"),
        star_cluster_share=("is_star_cluster", "mean"),
    ).reset_index()
    return summary


def main() -> None:
    data = load_datasets_enhanced()

    metrics_df, preds_by_model, fold_details = evaluate_models_enhanced(data.df)
    plot_model_comparison(metrics_df)

    # Choose primary (Two-Stage if available else HGB else Logistic)
    primary = None
    for name in ["Two-Stage HGB", "HistGradientBoosting", "L2 Logistic"]:
        if name in preds_by_model and not preds_by_model[name].empty:
            primary = name
            break

    if primary is None:
        print("No predictions available.")
        return

    # Calibration plot
    plot_calibration(preds_by_model[primary], f"enhanced_calibration_{primary.replace(' ', '_')}.png")

    # Error summary CSV
    err_df = compute_error_summary(preds_by_model[primary], data.df)
    if not err_df.empty:
        err_df.to_csv(FIG_DIR / f"enhanced_error_summary_{primary.replace(' ', '_')}.csv", index=False)

    # Save per-model predictions for inspection
    for name, p in preds_by_model.items():
        if p is not None and not p.empty:
            p.to_csv(FIG_DIR / f"enhanced_predictions_{name.replace(' ', '_')}.csv", index=False)


if __name__ == "__main__":
    main()
