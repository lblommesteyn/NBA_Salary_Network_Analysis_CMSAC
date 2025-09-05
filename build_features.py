
import os
import re
import glob
import math
from itertools import combinations
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

# Optional imports
try:
    import networkx as nx
except Exception as e:
    nx = None

def season_from_filename(fn: str) -> str:
    m = re.search(r"(\d{4})[-_](\d{4})", fn)
    return f"{m.group(1)}-{m.group(2)}" if m else None

def parse_lineup(s: str):
    if pd.isna(s):
        return []
    # Common separators: "-" or " - " or "|" or ","
    for sep in [" - ", "-", "|", ","]:
        if sep in s:
            parts = [p.strip() for p in s.split(sep) if p.strip()]
            if len(parts) >= 2:
                return parts
    # Fallback: whitespace split
    parts = [p.strip() for p in s.split() if p.strip()]
    return parts

def minutes_to_float(v):
    if pd.isna(v):
        return 0.0
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v)
    # Formats like "123:45" (min:sec)
    if ":" in s:
        try:
            mm, ss = s.split(":")
            return float(mm) + float(ss)/60.0
        except:
            pass
    # Strip any text
    s = re.sub(r"[^\d\.]", "", s)
    try:
        return float(s)
    except:
        return 0.0

def gini(arr):
    x = np.array(arr, dtype=float)
    if np.allclose(x, 0):
        return 0.0
    x = x[x>=0]
    if x.size == 0:
        return 0.0
    x_sorted = np.sort(x)
    n = x_sorted.size
    cumx = np.cumsum(x_sorted)
    g = (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n
    return float(g)

def top_k_share(arr, k=1):
    x = np.array(arr, dtype=float)
    if x.size == 0:
        return 0.0
    tot = x.sum()
    if tot <= 0:
        return 0.0
    return float(np.sort(x)[-k:].sum() / tot)

def degree_centralization(degrees):
    """Freeman centralization index (normalized)."""
    if len(degrees) <= 2:
        return 0.0
    d = np.array(degrees, dtype=float)
    d_max = d.max() if d.size else 0.0
    num = np.sum(d_max - d)
    n = len(d)
    den = (n - 1) * (n - 2)  # max for star graph (undirected)
    return float(num / den) if den > 0 else 0.0

def assortativity(edges, node_values):
    """Weighted assortativity by edge weight using Pearson corr of endpoint node values."""
    xs, ys, ws = [], [], []
    for (u, v), w in edges.items():
        if u in node_values and v in node_values:
            xs.append(node_values[u])
            ys.append(node_values[v])
            ws.append(w)
    if len(xs) < 2:
        return np.nan
    xs, ys, ws = np.array(xs), np.array(ys), np.array(ws, dtype=float)
    # Weighted Pearson correlation
    w = ws / ws.sum()
    mx = np.sum(w * xs)
    my = np.sum(w * ys)
    cov = np.sum(w * (xs - mx) * (ys - my))
    vx = np.sum(w * (xs - mx)**2)
    vy = np.sum(w * (ys - my)**2)
    if vx <= 0 or vy <= 0:
        return np.nan
    return float(cov / np.sqrt(vx * vy))

def greedy_modularity_parts(edges):
    """Approx community detection using NetworkX greedy modularity (if available)."""
    if nx is None:
        return None, np.nan, None
    G = nx.Graph()
    for (u, v), w in edges.items():
        if w > 0:
            G.add_edge(u, v, weight=float(w))
    if G.number_of_nodes() == 0:
        return None, np.nan, None
    try:
        from networkx.algorithms.community import greedy_modularity_communities, modularity
        comms = list(greedy_modularity_communities(G, weight='weight'))
        # compute modularity
        Q = modularity(G, comms, weight='weight')
        parts = {node: i for i, c in enumerate(comms) for node in c}
        sizes = [len(c) for c in comms]
        cv = float(np.std(sizes) / (np.mean(sizes) + 1e-9)) if len(sizes) >= 2 else 0.0
        return parts, Q, cv
    except Exception as e:
        return None, np.nan, None

def build_edges_and_minutes(df, team_col, lineup_col, minutes_col):
    # aggregate per team-season
    records = []
    for _, row in df.iterrows():
        team = row.get(team_col, None)
        lineup = parse_lineup(row.get(lineup_col, ""))
        minutes = minutes_to_float(row.get(minutes_col, 0.0))
        if team is None or len(lineup) < 2 or minutes <= 0:
            continue
        records.append((team, tuple(sorted(lineup)), minutes))
    # roll up identical lineups per team
    out = defaultdict(lambda: defaultdict(float))  # team -> lineup(tuple) -> minutes
    for team, lineup, minutes in records:
        out[team][lineup] += minutes
    # compute player minutes and pair shared minutes
    team_player_minutes = defaultdict(lambda: defaultdict(float))
    team_pair_minutes = defaultdict(lambda: defaultdict(float))
    for team, lineup_dict in out.items():
        for lineup, mins in lineup_dict.items():
            for p in lineup:
                team_player_minutes[team][p] += mins
            for u, v in combinations(lineup, 2):
                key = tuple(sorted((u, v)))
                team_pair_minutes[team][key] += mins
    return team_player_minutes, team_pair_minutes

def build_features_for_team(team, player_minutes, pair_minutes, salary_share=None, min_threshold=300.0):
    """Compute features for a single team graph."""
    # filter by minutes threshold
    pm = {p: m for p, m in player_minutes.items() if m >= min_threshold}
    if len(pm) < 2:
        return None  # not enough players

    # node values: salary share if provided, else minutes share
    tot_salary = 0.0
    node_salary_share = {}
    if salary_share:
        # salary_share is already in fractions summing ~1 per team
        node_salary_share = {p: float(salary_share.get(p, 0.0)) for p in pm}
        tot_salary = sum(node_salary_share.values())

    # edges: restrict to players that passed threshold
    edges = {}
    for (u, v), m in pair_minutes.items():
        if u in pm and v in pm and m > 0:
            # co-possession intensity normalized by max individual minutes
            denom = max(pm[u], pm[v])
            w = float(m) / float(denom) if denom > 0 else 0.0
            if w > 0:
                edges[(u, v)] = w

    # basic degrees (weighted)
    deg = Counter()
    for (u, v), w in edges.items():
        deg[u] += w
        deg[v] += w

    degrees = [deg.get(p, 0.0) for p in pm]
    degree_centr = degree_centralization(degrees)
    edge_weights = np.array(list(edges.values())) if edges else np.array([])
    edge_conc_top5 = float(np.sort(edge_weights)[-5:].sum() / edge_weights.sum()) if edge_weights.size >= 5 else float(edge_weights.sum() / (edge_weights.sum() + 1e-9))
    edge_conc_top10 = float(np.sort(edge_weights)[-10:].sum() / (edge_weights.sum() + 1e-9)) if edge_weights.size >= 10 else edge_conc_top5

    # salary dispersion
    if salary_share and tot_salary > 0:
        salaries = [node_salary_share[p] for p in pm]
        gini_salary = gini(salaries)
        top1 = top_k_share(salaries, 1)
        top2 = top_k_share(salaries, 2)
        top3 = top_k_share(salaries, 3)
        assort = assortativity(edges, node_salary_share)
    else:
        # proxy with minutes share
        minutes = np.array(list(pm.values()), dtype=float)
        minutes_share = (minutes / minutes.sum()) if minutes.sum() > 0 else minutes
        gini_salary = gini(minutes_share)
        top1 = top_k_share(minutes_share, 1)
        top2 = top_k_share(minutes_share, 2)
        top3 = top_k_share(minutes_share, 3)
        assort = np.nan  # not meaningful without salary shares

    # community
    parts, Q, comm_cv = greedy_modularity_parts(edges)

    feat = {
        "team": team,
        "n_players": len(pm),
        "degree_centralization": degree_centr,
        "edge_concentration_top5": edge_conc_top5,
        "edge_concentration_top10": edge_conc_top10,
        "modularity_Q": Q,
        "community_size_cv": comm_cv if comm_cv is not None else np.nan,
        "salary_gini": gini_salary,
        "salary_top1_share": top1,
        "salary_top2_share": top2,
        "salary_top3_share": top3,
        "salary_assortativity": assort,
        "total_player_minutes": float(sum(pm.values())),
    }
    return feat

def load_salaries_csv(path):
    """Expect columns: team, season, player, salary (in dollars)."""
    if not os.path.exists(path):
        return None
    sal = pd.read_csv(path)
    # normalize column names
    sal.columns = [c.strip().lower() for c in sal.columns]
    if not {"team","season","player","salary"} <= set(sal.columns):
        raise ValueError("Salary CSV must have columns: team, season, player, salary")
    # salary share per team-season
    sal["salary"] = sal["salary"].astype(float)
    sal["team_season"] = sal["team"].astype(str) + " " + sal["season"].astype(str)
    shares = {}
    for (ts), sdf in sal.groupby("team_season"):
        tot = sdf["salary"].sum()
        d = {row["player"]: (row["salary"] / tot if tot>0 else 0.0) for _, row in sdf.iterrows()}
        shares[ts] = d
    return shares

def main(
    lineup_glob="/mnt/data/lineups_by_team_*/*.csv",
    lineup_team_col="TEAM",
    lineup_lineup_col="GROUP_NAME",
    lineup_minutes_col="MIN",
    season_label=None,
    salaries_csv=None,
    team_stats_csv=None,
    output_features="/mnt/data/features_team_season.csv",
    min_threshold=300.0
):
    # Gather all lineup files
    files = sorted(glob.glob(lineup_glob))
    if not files:
        print(f"[WARN] No lineup files found for glob: {lineup_glob}")
    all_features = []
    salaries_by_team_season = load_salaries_csv(salaries_csv) if salaries_csv else None

    for fp in files:
        df = pd.read_csv(fp)
        # Try to detect columns if not provided
        cols = {c.lower(): c for c in df.columns}
        team_col = lineup_team_col if lineup_team_col in df.columns else (cols.get("team_abbreviation") or cols.get("team") or list(df.columns)[0])
        lineup_col = lineup_lineup_col if lineup_lineup_col in df.columns else (cols.get("group_name") or cols.get("lineup") or cols.get("players"))
        minutes_col = lineup_minutes_col if lineup_minutes_col in df.columns else (cols.get("min") or cols.get("minutes") or list(df.columns)[1])

        # Infer season from filename if not provided
        season = season_label or season_from_filename(os.path.basename(fp)) or ""

        # Build player & pair minutes per team (within this file assumed one season)
        team_player_minutes, team_pair_minutes = build_edges_and_minutes(df, team_col, lineup_col, minutes_col)

        # Compute features for each team
        for team, pm in team_player_minutes.items():
            pair_m = team_pair_minutes[team]
            # salary share lookup
            ss = None
            if salaries_by_team_season is not None:
                key = f"{team} {season}"
                ss = salaries_by_team_season.get(key, None)
            feat = build_features_for_team(team, pm, pair_m, salary_share=ss, min_threshold=min_threshold)
            if feat:
                feat["season"] = season
                all_features.append(feat)

    feats = pd.DataFrame(all_features).sort_values(["season","team"])
    if team_stats_csv and os.path.exists(team_stats_csv):
        stats = pd.read_csv(team_stats_csv)
        stats.columns = [c.strip().lower() for c in stats.columns]
        # Expect: team, season/year, reg_games, reg_wins, reg_losses, reg_win_pct, total_games, pts_per_game, opp_pts_per_game, net_rating, srs
        # Harmonize columns
        if "year" in stats.columns and "season" not in stats.columns:
            stats.rename(columns={"year":"season"}, inplace=True)
        feats = feats.merge(stats, left_on=["team","season"], right_on=["team","season"], how="left")

    feats.to_csv(output_features, index=False)
    print(f"[OK] Wrote features to: {output_features}")
    return feats

if __name__ == "__main__":
    # Example call (adjust the glob/columns to your lineup exports)
    main(
        lineup_glob="/mnt/data/your_lineups/*.csv",   # TODO: replace with your lineup file path(s)
        lineup_team_col="TEAM_ABBREVIATION",          # or "TEAM"
        lineup_lineup_col="GROUP_NAME",               # or "LINEUP"
        lineup_minutes_col="MIN",                     # or "Minutes"
        season_label=None,
        salaries_csv="/mnt/data/salaries_team_season.csv",   # optional
        team_stats_csv="/mnt/data/team_season_stats.csv",    # optional
        output_features="/mnt/data/features_team_season.csv",
        min_threshold=300.0
    )
