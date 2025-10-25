from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm

# Reuse existing data loaders and error helper
from poster_figures import (
    load_datasets,
    evaluate_models,
    _prepare_error_dataframe,
)


def main() -> None:
    sns.set_theme(style="whitegrid", context="talk")

    # Load data and get logistic predictions
    data = load_datasets()
    _, logistic_preds, _, _ = evaluate_models(data.df)
    if logistic_preds.empty:
        print("No predictions available; cannot draw figure.")
        return

    df_error = _prepare_error_dataframe(data.df, logistic_preds)

    # Gradient scatter for ROLE shock vs team net rating
    fig, ax = plt.subplots(figsize=(8, 6))
    vals = df_error["error"].to_numpy()
    v = float(np.nanmax(np.abs(vals))) if len(vals) else 1.0
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-v, vmax=v)

    sc = ax.scatter(
        df_error["team_net_rating"],
        df_error["shock_role_norm"],
        c=vals,
        cmap="coolwarm",
        norm=norm,
        edgecolor="white",
        linewidth=0.5,
        alpha=0.9,
    )

    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Team Net Rating")
    ax.set_ylabel("Role Shock / |Net Rating|")
    ax.set_title("Role Shock / |Net Rating| vs. Team Net Rating")

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Prediction Error (Pred - Actual)")

    outdir = Path("poster_figures")
    outdir.mkdir(exist_ok=True)
    fig.tight_layout()
    out_path = outdir / "poster_role_shock_vs_net_rating.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
