"""
Enhanced diagnostics for poster:
- Ablation waterfall across feature groups (controls -> dispersion -> connectivity -> full)
- Partial Dependence (PDP) and ICE curves for key features

Run: python enhanced_diagnostics.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import f1_score

from enhanced_modeling import RANDOM_STATE, FIG_DIR, load_datasets_enhanced

sns.set_theme(style="whitegrid", context="talk")


def _evaluate_loso(df: pd.DataFrame, features: List[str], target: str = "playoff_round") -> Tuple[pd.DataFrame, float]:
    available = [c for c in features if c in df.columns]
    data = df.dropna(subset=available + [target]).reset_index(drop=True)
    X = data[available]
    y = data[target].to_numpy()
    seasons = data["season"].to_numpy()

    # Model
    clf = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_depth=4,
        max_iter=400,
        min_samples_leaf=8,
        random_state=RANDOM_STATE,
    )

    # Round-aware weights
    weight_map = {0: 1.0, 1: 1.2, 2: 1.6, 3: 2.2, 4: 3.0}
    sample_weight_all = np.array([weight_map.get(int(c), 1.0) for c in y])

    records: List[Dict[str, float]] = []
    seasons_unique = sorted(pd.unique(seasons))

    for holdout in seasons_unique:
        train_mask = seasons != holdout
        test_mask = ~train_mask
        if not np.any(test_mask):
            continue
        X_train, X_test = X.loc[train_mask], X.loc[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        try:
            clf.fit(X_train, y_train, sample_weight=sample_weight_all[train_mask])
        except TypeError:
            clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        records.append({"season": holdout, "macro_f1": f1_score(y_test, y_pred, average="macro")})

    fold_df = pd.DataFrame(records)
    return fold_df, float(fold_df["macro_f1"].mean() if not fold_df.empty else np.nan)


def ablation_waterfall_enhanced(df: pd.DataFrame) -> pd.DataFrame:
    # Feature groups
    controls = [
        "team_net_rating",
        "minutes_gini",
        "minutes_assortativity",
    ]
    dispersion = controls + [
        "salary_gini",
        "minutes_top1_share",
        "salary_top1_share",
    ]
    connectivity = dispersion + [
        "degree_centralization",
        "edge_concentration_top5",
        "edge_concentration_top10",
        "community_size_cv",
        "modularity_Q",
    ]
    full = connectivity + [
        # Resilience/shocks and transforms
        "RRS",
        "shock_star", "shock_star_per_nr", "shock_star_log", "shock_star_cap",
        "shock_role", "shock_role_per_nr", "shock_role_log", "shock_role_cap",
        "shock_connector", "shock_connector_per_nr", "shock_connector_log", "shock_connector_cap",
        # Mesh surplus and controls
        "salary_assortativity", "salary_assort_z", "mesh_surplus", "assort_n_nodes", "assort_n_edges",
        # Superstar punch interactions
        "nr_x_top1_salary", "nr_x_top2_salary", "nr_x_top3_salary", "nr_x_one_minus_assort",
        "nr_x_minutes_top1", "nr_x_minutes_top2", "nr_x_minutes_top3",
        # Depth
        "edge_concentration_top10_to_top5",
        # Path
        "seed_proxy",
        # Archetypes
        "cluster_id", "is_star_cluster", "high_shock_high_talent",
        # Availability blend
        "star_p_avail_proxy", "nr_expected_star_availability",
    ]

    rows = []
    for label, feat in [
        ("A: Controls", controls),
        ("B: + Dispersion", dispersion),
        ("C: + Connectivity", connectivity),
        ("D: + Full", full),
    ]:
        fold_df, macro = _evaluate_loso(df, feat)
        rows.append({"Model": label, "Macro-F1": macro})

    table = pd.DataFrame(rows)
    table["increment"] = table["Macro-F1"].diff().fillna(table["Macro-F1"])

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=table, x="Model", y="increment", color="#1f77b4", ax=ax)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("Macro-F1 Gain vs. Previous Step")
    ax.set_xlabel("Feature Set")
    ax.set_title("Feature Group Contributions to Macro-F1 (Enhanced)")
    for idx, row in table.iterrows():
        ax.text(idx, row["increment"], f"{row['Macro-F1']:.3f}", ha="center", va="bottom", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "poster_ablation_waterfall_enhanced.png", dpi=300)
    plt.close(fig)

    table.to_csv(FIG_DIR / "table_ablations_playoff_enhanced.csv", index=False)
    return table


def _fit_two_stage_full(df: pd.DataFrame, features: List[str]) -> Tuple[HistGradientBoostingClassifier, pd.DataFrame, List[str]]:
    available = [c for c in features if c in df.columns]
    # Keep only numeric columns for model input
    numeric_available = [c for c in available if pd.api.types.is_numeric_dtype(df[c])]
    data = df.dropna(subset=numeric_available + ["playoff_round"]).reset_index(drop=True)
    X = data[numeric_available].copy()
    y = data["playoff_round"].to_numpy()

    # Stage 1: baseline logistic
    baseline_cols = [c for c in ["team_net_rating", "minutes_gini", "minutes_top1_share"] if c in data.columns]
    if len(baseline_cols) == 3:
        scaler = StandardScaler().fit(data[baseline_cols])
        base_train = scaler.transform(data[baseline_cols])
        base_clf = LogisticRegression(C=1.2, penalty="l2", solver="lbfgs", multi_class="multinomial", max_iter=5000, random_state=RANDOM_STATE)
        base_clf.fit(base_train, y)
        base_proba = base_clf.predict_proba(base_train)
        er_classes = base_clf.classes_
        er_full = (base_proba * er_classes).sum(axis=1)
        X["baseline_er"] = er_full
        # Store also in data so downstream slicing by avail works
        data["baseline_er"] = er_full

    # Stage 2: HGB
    hgb = HistGradientBoostingClassifier(learning_rate=0.08, max_depth=4, max_iter=400, min_samples_leaf=8, random_state=RANDOM_STATE)
    weight_map = {0: 1.0, 1: 1.2, 2: 1.6, 3: 2.2, 4: 3.0}
    sw = np.array([weight_map.get(int(c), 1.0) for c in y])
    try:
        hgb.fit(X, y, sample_weight=sw)
    except TypeError:
        hgb.fit(X, y)
    # Return the exact feature set used by the fitted model (includes baseline_er if added)
    return hgb, data, list(X.columns)


def _predict_prob_ge2(model: HistGradientBoostingClassifier, X: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(X)
    classes = model.classes_
    ge2_idx = [np.where(classes == c)[0][0] for c in classes if c >= 2]
    return proba[:, ge2_idx].sum(axis=1) if ge2_idx else np.zeros(len(X))


def pdp_ice_enhanced(df: pd.DataFrame) -> None:
    # Key features for curves
    features = [
        "team_net_rating",  # for context
        "nr_x_top2_salary",
        "shock_star_per_nr",
        "seed_proxy",
        "mesh_surplus",
        "nr_expected_star_availability",
    ]

    # Fit two-stage on full data with the full enhanced feature set
    # Use only numeric columns for the full feature set
    full_features = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    # Remove target from features if present
    full_features = [c for c in full_features if c != "playoff_round"]
    model, data, avail = _fit_two_stage_full(df, full_features)

    # PDP: vary each feature over its 5th–95th percentiles
    n_points = 25
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()

    X_base = data[avail].copy()
    x_ref = X_base.median(numeric_only=True)

    rng = np.random.default_rng(RANDOM_STATE)
    # Sample from the exact model feature matrix to avoid missing columns like baseline_er
    ice_samples = X_base.sample(n=min(12, len(X_base)), random_state=RANDOM_STATE)

    for i, feat in enumerate(features):
        ax = axes[i]
        if feat not in X_base.columns:
            ax.axis("off")
            ax.set_title(f"{feat} (missing)")
            continue
        q = np.linspace(0.05, 0.95, n_points)
        grid = np.quantile(X_base[feat].dropna(), q) if X_base[feat].notna().any() else np.linspace(-1, 1, n_points)

        # PDP: fix all at median, vary feat
        X_tmp = pd.DataFrame(np.tile(x_ref.to_numpy(), (len(grid), 1)), columns=x_ref.index)
        X_tmp[feat] = grid
        y_pdp = _predict_prob_ge2(model, X_tmp)
        ax.plot(grid, y_pdp, color="#1f77b4", label="PDP")

        # ICE: for a few samples, vary feat around that sample's features
        for (_, row) in ice_samples.iterrows():
            # Row comes from X_base so it already has the exact feature columns/ordering
            X_row = pd.DataFrame(np.tile(row.to_numpy(), (len(grid), 1)), columns=avail)
            X_row[feat] = grid
            y_ice = _predict_prob_ge2(model, X_row)
            ax.plot(grid, y_ice, color="#1f77b4", alpha=0.2)

        ax.set_title(feat)
        ax.set_xlabel(feat)
        ax.set_ylabel("Pr(Advance ≥ R2)")

    fig.suptitle("PDP + ICE for Key Enhanced Features")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "poster_pdp_ice_enhanced.png", dpi=300)
    plt.close(fig)


def main() -> None:
    data = load_datasets_enhanced()
    df = data.df.copy()

    # Ablations
    ablation_waterfall_enhanced(df)

    # PDP/ICE
    pdp_ice_enhanced(df)


if __name__ == "__main__":
    main()
