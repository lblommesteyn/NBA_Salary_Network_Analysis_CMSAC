
import os, glob, re, math, random
import numpy as np, pandas as pd
from collections import defaultdict
from itertools import combinations

try:
    import networkx as nx
except Exception:
    nx = None

from build_features import (
    season_from_filename, parse_lineup, minutes_to_float,
    build_edges_and_minutes, build_features_for_team
)

RNG = np.random.default_rng(7)

def fit_proxy_net_rating(features_csv="/mnt/data/features_team_season.csv"):
    """Fit a simple ridge model predicting net_rating from features as a proxy."""
    from sklearn.linear_model import RidgeCV
    df = pd.read_csv(features_csv)
    # require net_rating
    if "net_rating" not in df.columns:
        raise ValueError("features CSV must include 'net_rating' (merge team_season_stats.csv first).")
    # select numeric cols
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    drop_cols = {"n_players","total_player_minutes"}
    X_cols = [c for c in num_cols if c not in drop_cols and c != "net_rating"]
    X = df[X_cols].fillna(0.0).values
    y = df["net_rating"].values
    model = RidgeCV(alphas=np.logspace(-3,3,25)).fit(X, y)
    return model, X_cols, df

def recompute_features_after_removal(team, pm, pair_m, removed):
    # Remove nodes and any edges touching them
    pm2 = {p:m for p,m in pm.items() if p not in removed}
    pair2 = {}
    for (u,v), m in pair_m.items():
        if u not in removed and v not in removed:
            pair2[(u,v)] = m
    return build_features_for_team(team, pm2, pair2, salary_share=None, min_threshold=0.0)

def run_rrs(lineup_glob, team_stats_csv, features_csv="/mnt/data/features_team_season.csv", n_role_trials=20):
    model, X_cols, feats = fit_proxy_net_rating(features_csv)
    stats = pd.read_csv(team_stats_csv)
    # Regroup inputs: we need the raw pm/pair_m for each team-season to simulate removals
    # For simplicity we expect a single lineup file per season here
    # (If you have multiple files per season, glob them with a pattern that matches each season file)
    rr_rows = []
    # Build a mapping team-season -> pm/pair_m using the same parsing as in build_features.py
    season_files = sorted(glob.glob(lineup_glob))
    for fp in season_files:
        season = season_from_filename(os.path.basename(fp)) or ""
        df = pd.read_csv(fp)
        cols = {c.lower(): c for c in df.columns}
        team_col = cols.get("team_abbreviation") or cols.get("team") or list(df.columns)[0]
        lineup_col = cols.get("group_name") or cols.get("lineup") or cols.get("players")
        minutes_col = cols.get("min") or cols.get("minutes")
        tpm, tpair = build_edges_and_minutes(df, team_col, lineup_col, minutes_col)

        # intact predicted NR for each team
        for team in tpm.keys():
            pm = tpm[team]
            pair_m = tpair[team]
            intact_feat = build_features_for_team(team, pm, pair_m, salary_share=None, min_threshold=0.0)
            if intact_feat is None:
                continue
            intact_df = pd.DataFrame([intact_feat])
            X_intact = intact_df[X_cols].fillna(0.0).values
            nr_hat_intact = float(model.predict(X_intact)[0])

            # star removal (highest degree approx: by total pair minutes)
            deg = defaultdict(float)
            for (u,v), m in pair_m.items():
                deg[u]+=m; deg[v]+=m
            if deg:
                star = max(deg, key=deg.get)
            else:
                continue

            star_feat = recompute_features_after_removal(team, pm, pair_m, {star})
            X_star = pd.DataFrame([star_feat])[X_cols].fillna(0.0).values if star_feat else X_intact
            nr_hat_star = float(model.predict(X_star)[0]) if star_feat else nr_hat_intact

            # role removal: random mid-salary proxy = mid-minutes players
            mins_sorted = sorted([(p,m) for p,m in pm.items()], key=lambda x:x[1])
            mid_slice = mins_sorted[len(mins_sorted)//3: 2*len(mins_sorted)//3] or mins_sorted
            role_deltas = []
            for _ in range(n_role_trials):
                cand = RNG.choice([p for p,_ in mid_slice])
                role_feat = recompute_features_after_removal(team, pm, pair_m, {cand})
                X_role = pd.DataFrame([role_feat])[X_cols].fillna(0.0).values if role_feat else X_intact
                nr_hat_role = float(model.predict(X_role)[0]) if role_feat else nr_hat_intact
                role_deltas.append(nr_hat_intact - nr_hat_role)
            role_drop = float(np.mean(role_deltas)) if role_deltas else 0.0

            # connector removal: proxy by betweenness requires networkx; fallback to highest edge count neighbor
            connector = None
            if nx is not None and len(pm) > 2:
                G = nx.Graph()
                for (u,v), m in pair_m.items():
                    if u in pm and v in pm and m>0:
                        G.add_edge(u,v, weight=float(m))
                if G.number_of_nodes() >= 3:
                    btw = nx.betweenness_centrality(G, weight='weight', normalized=True)
                    connector = max(btw, key=btw.get) if btw else None
            if connector is None:
                # fallback: node with highest number of distinct partners
                partners = defaultdict(set)
                for (u,v), m in pair_m.items():
                    partners[u].add(v); partners[v].add(u)
                connector = max(partners, key=lambda k: len(partners[k])) if partners else None

            conn_feat = recompute_features_after_removal(team, pm, pair_m, {connector}) if connector else None
            X_conn = pd.DataFrame([conn_feat])[X_cols].fillna(0.0).values if conn_feat else X_intact
            nr_hat_conn = float(model.predict(X_conn)[0]) if conn_feat else nr_hat_intact

            # RRS aggregation
            deltas = np.array([nr_hat_intact - nr_hat_star,
                               role_drop,
                               nr_hat_intact - nr_hat_conn], dtype=float)
            denom = abs(nr_hat_intact) + 1e-3
            RRS = 1.0 - float(np.mean(deltas) / denom)

            rr_rows.append({
                "team": team, "season": season,
                "nr_hat_intact": nr_hat_intact,
                "drop_star": float(nr_hat_intact - nr_hat_star),
                "drop_role": float(role_drop),
                "drop_connector": float(nr_hat_intact - nr_hat_conn),
                "RRS": RRS
            })
    out = pd.DataFrame(rr_rows).sort_values(["season","team"])
    out.to_csv("/mnt/data/rrs_by_team_season.csv", index=False)
    print("[OK] wrote /mnt/data/rrs_by_team_season.csv")
    return out

if __name__ == "__main__":
    # Example usage (adjust paths first)
    run_rrs(
        lineup_glob="/mnt/data/your_lineups/*.csv",
        team_stats_csv="/mnt/data/team_season_stats.csv",
        features_csv="/mnt/data/features_team_season.csv",
        n_role_trials=20
    )
