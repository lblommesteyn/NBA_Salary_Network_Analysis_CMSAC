"""
Bracket-aware playoff projection with play-in and probability trees.

- Builds seeds per conference by season (proxy: sort by team_net_rating).
- Simulates play-in (7–10) and full bracket per conference with Monte Carlo.
- Series win probability computed from single-game p via best-of-7 formula.
- Single-game win p is logistic in net rating difference.
- Outputs per-team round probabilities and per-team opponent distributions.
- Generates probability tree figures for top-likelihood contenders.

Run: python bracket_projection.py
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from math import comb

RANDOM_STATE = 42
FIG_DIR = Path("poster_figures")
FIG_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")

# Static conference map in 3-letter codes consistent with vis_clean_data
EAST = {
    "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DET", "IND", "MIA", "MIL",
    "NYK", "ORL", "PHI", "TOR", "WAS",
}
WEST = {
    "DAL", "DEN", "GSW", "HOU", "LAC", "LAL", "MEM", "MIN", "NOP", "OKC",
    "PHX", "POR", "SAC", "SAS", "UTA",
}

SEASON_YEAR = {
    "2020-2021": 2021,
    "2021-2022": 2022,
    "2022-2023": 2023,
    "2023-2024": 2024,
    "2024-2025": 2025,
}

@dataclass
class TeamRating:
    season: str
    team: str
    net_rating: float


def load_team_ratings() -> pd.DataFrame:
    """Load team net ratings from features and keep one row per season/team."""
    df = (
        pd.read_csv("vis_clean_data/features_from_pos_lineups.csv")
        .sort_values(["season", "team", "n_players"], ascending=[True, True, False])
        .drop_duplicates(subset=["season", "team"], keep="first")
        [["season", "team", "team_net_rating"]]
    )
    return df.rename(columns={"team_net_rating": "net_rating"})


def win_prob_single_game(nr_a: float, nr_b: float, k: float = 0.14, hca: float = 0.0) -> float:
    """Logistic single-game win probability for Team A given net rating diff.

    p = 1 / (1 + exp(-k * (nr_a - nr_b + hca)))
    hca: small edge for higher seed if desired (in net rating units)
    """
    return 1.0 / (1.0 + np.exp(-k * (nr_a - nr_b + hca)))


def series_win_prob_best_of_7(p: float) -> float:
    """Probability Team A wins a best-of-7 series if each game has win prob p.

    Sum over k losses before the 4th win: sum_{k=0..3} C(3+k, k) p^4 (1-p)^k
    """
    return float(sum(comb(3 + k, k) * (p ** 4) * ((1 - p) ** k) for k in range(4)))


def seed_conferences(ratings: pd.DataFrame, season: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = ratings[ratings["season"] == season].copy()
    east = df[df["team"].isin(EAST)].sort_values("net_rating", ascending=False).reset_index(drop=True)
    west = df[df["team"].isin(WEST)].sort_values("net_rating", ascending=False).reset_index(drop=True)
    east["seed"] = np.arange(1, len(east) + 1)
    west["seed"] = np.arange(1, len(west) + 1)
    return east, west


def simulate_play_in(conf_df: pd.DataFrame, rng: np.random.Generator) -> Tuple[str, str]:
    """Simulate play-in to determine 7 and 8 seeds given conference ranking (1..N).

    Assumes seeds >= 10 exist; if not, returns top available.
    """
    sub = conf_df.copy()
    teams = {int(r.seed): r.team for r in sub.itertuples()}
    nr = {int(r.seed): float(r.net_rating) for r in sub.itertuples()}

    # If fewer than 10 teams available (e.g., toy data), fill by ordering
    if max(teams) < 10:
        available = sorted(teams)
        # Fall back: last two slots are next seeds
        seed7 = available[6] if len(available) >= 7 else available[-1]
        seed8 = available[7] if len(available) >= 8 else available[-1]
        return teams[seed7], teams[seed8]

    # 7 vs 8 (winner becomes 7 seed)
    p_7 = win_prob_single_game(nr[7], nr[8])
    seven_beats_eight = rng.random() < p_7
    if seven_beats_eight:
        seed7_team = teams[7]
        loser78 = 8
    else:
        seed7_team = teams[8]
        loser78 = 7

    # 9 vs 10 (loser eliminated)
    p_9 = win_prob_single_game(nr[9], nr[10])
    nine_beats_ten = rng.random() < p_9
    winner_910 = 9 if nine_beats_ten else 10

    # Loser of 7/8 vs winner of 9/10 for 8th seed
    p_last = win_prob_single_game(nr[loser78], nr[winner_910])
    loser78_wins = rng.random() < p_last
    seed8_team = teams[loser78] if loser78_wins else teams[winner_910]

    return seed7_team, seed8_team


def simulate_series(team_a: str, team_b: str, nr_map: Dict[str, float], rng: np.random.Generator) -> str:
    p = win_prob_single_game(nr_map[team_a], nr_map[team_b])
    p_series = series_win_prob_best_of_7(p)
    return team_a if rng.random() < p_series else team_b


def run_season_bracket(ratings: pd.DataFrame, season: str, sims: int = 5000, rng: np.random.Generator | None = None) -> Tuple[pd.DataFrame, Dict[str, Dict[str, List[str]]]]:
    """Monte Carlo bracket with play-in. Returns:
      - per-team probabilities of reaching each round (0..4)
      - routes/opponents dictionary for building probability trees
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_STATE)

    east, west = seed_conferences(ratings, season)
    nr_map = {r.team: float(r.net_rating) for r in pd.concat([east, west]).itertuples(index=False)}

    teams = list(nr_map.keys())
    counts = {t: np.zeros(5, dtype=int) for t in teams}  # reached rounds: 0..4
    routes: Dict[str, Dict[str, List[str]]] = {t: {"R1": [], "R2": [], "R3": [], "R4": []} for t in teams}

    for _ in range(sims):
        # Determine 7/8 via play-in
        seed7_e, seed8_e = simulate_play_in(east, rng)
        seed7_w, seed8_w = simulate_play_in(west, rng)

        # Build bracket per conference
        bracket_e = {
            1: east.loc[east["seed"] == 1, "team"].item(),
            2: east.loc[east["seed"] == 2, "team"].item(),
            3: east.loc[east["seed"] == 3, "team"].item(),
            4: east.loc[east["seed"] == 4, "team"].item(),
            5: east.loc[east["seed"] == 5, "team"].item(),
            6: east.loc[east["seed"] == 6, "team"].item(),
            7: seed7_e,
            8: seed8_e,
        }
        bracket_w = {
            1: west.loc[west["seed"] == 1, "team"].item(),
            2: west.loc[west["seed"] == 2, "team"].item(),
            3: west.loc[west["seed"] == 3, "team"].item(),
            4: west.loc[west["seed"] == 4, "team"].item(),
            5: west.loc[west["seed"] == 5, "team"].item(),
            6: west.loc[west["seed"] == 6, "team"].item(),
            7: seed7_w,
            8: seed8_w,
        }

        # Round 1
        r1_e_pairs = [(1, 8), (2, 7), (3, 6), (4, 5)]
        r1_w_pairs = [(1, 8), (2, 7), (3, 6), (4, 5)]
        r1_e_winners = []
        r1_w_winners = []
        for a, b in r1_e_pairs:
            ta, tb = bracket_e[a], bracket_e[b]
            winner = simulate_series(ta, tb, nr_map, rng)
            r1_e_winners.append(winner)
            routes[winner]["R1"].append(tb if winner == ta else ta)
        for a, b in r1_w_pairs:
            ta, tb = bracket_w[a], bracket_w[b]
            winner = simulate_series(ta, tb, nr_map, rng)
            r1_w_winners.append(winner)
            routes[winner]["R1"].append(tb if winner == ta else ta)

        # Update counts: reaching Round 1 means made playoffs (>=1)
        for t in r1_e_winners + r1_w_winners:
            counts[t][1] += 1

        # Round 2 (E: winner of 1/8 vs 4/5; 2/7 vs 3/6)
        r2_e_pairs = [(r1_e_winners[0], r1_e_winners[3]), (r1_e_winners[1], r1_e_winners[2])]
        r2_w_pairs = [(r1_w_winners[0], r1_w_winners[3]), (r1_w_winners[1], r1_w_winners[2])]
        r2_e_winners = []
        r2_w_winners = []
        for ta, tb in r2_e_pairs:
            winner = simulate_series(ta, tb, nr_map, rng)
            r2_e_winners.append(winner)
            routes[winner]["R2"].append(tb if winner == ta else ta)
        for ta, tb in r2_w_pairs:
            winner = simulate_series(ta, tb, nr_map, rng)
            r2_w_winners.append(winner)
            routes[winner]["R2"].append(tb if winner == ta else ta)

        for t in r2_e_winners + r2_w_winners:
            counts[t][2] += 1

        # Conference Finals
        e_final = simulate_series(r2_e_winners[0], r2_e_winners[1], nr_map, rng)
        routes[e_final]["R3"].append(r2_e_winners[1] if e_final == r2_e_winners[0] else r2_e_winners[0])
        w_final = simulate_series(r2_w_winners[0], r2_w_winners[1], nr_map, rng)
        routes[w_final]["R3"].append(r2_w_winners[1] if w_final == r2_w_winners[0] else r2_w_winners[0])

        counts[e_final][3] += 1
        counts[w_final][3] += 1

        # NBA Finals
        champ = simulate_series(e_final, w_final, nr_map, rng)
        routes[champ]["R4"].append(w_final if champ == e_final else e_final)
        counts[champ][4] += 1

        # Everyone that made playoffs at least reached R1; teams that did not qualify (lost play-in or lower) remain at 0.
        # Increment R0 for any team that did not reach R1 in this sim
        reached_r1 = set(r1_e_winners + r1_w_winners)
        for t in teams:
            if t not in reached_r1:
                counts[t][0] += 1

    # Build probabilities
    prob_rows: List[Dict[str, object]] = []
    for t in teams:
        total = float(sims)
        pr = counts[t] / total
        prob_rows.append({
            "season": season,
            "team": t,
            "pr_out_before_r1": pr[0],
            "pr_r1": pr[1],
            "pr_r2": pr[2],
            "pr_r3": pr[3],
            "pr_r4": pr[4],
        })
    probs_df = pd.DataFrame(prob_rows)
    return probs_df, routes


def summarize_opponents(routes: Dict[str, Dict[str, List[str]]]) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for team, rdict in routes.items():
        recs = []
        for rnd in ["R1", "R2", "R3", "R4"]:
            if not rdict[rnd]:
                continue
            s = pd.Series(rdict[rnd]).value_counts(normalize=True).sort_values(ascending=False)
            for opp, p in s.items():
                recs.append({"round": rnd, "opponent": opp, "prob": p})
        out[team] = pd.DataFrame(recs)
    return out


def plot_probability_tree(season: str, team: str, opp_df: pd.DataFrame, probs_row: pd.Series) -> None:
    """Simple probability tree figure: show the top likely opponent per round and team’s round probabilities."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")

    y = 0.85
    ax.text(0.02, y, f"{team} — {season}", fontsize=18, fontweight="bold")
    y -= 0.08

    # Round probabilities
    txt = (
        f"Pr(R1)={probs_row['pr_r1']:.2f}  "
        f"Pr(R2)={probs_row['pr_r2']:.2f}  "
        f"Pr(R3)={probs_row['pr_r3']:.2f}  "
        f"Pr(Title)={probs_row['pr_r4']:.2f}"
    )
    ax.text(0.02, y, txt, fontsize=14)
    y -= 0.1

    # For each round, show top 3 likely opponents and probabilities
    for rnd in ["R1", "R2", "R3", "R4"]:
        row = opp_df[opp_df["round"] == rnd].sort_values("prob", ascending=False).head(3)
        if row.empty:
            continue
        line = ",  ".join([f"{r.opponent} ({r.prob:.2f})" for r in row.itertuples(index=False)])
        ax.text(0.04, y, f"Most likely {rnd} opponents: {line}", fontsize=13)
        y -= 0.08

    ax.set_title("Probability Tree (most likely opponents by round)", loc="left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"probability_tree_{season}_{team}.png", dpi=300)
    plt.close(fig)


def plot_conference_projection(season: str, probs_df: pd.DataFrame) -> None:
    df = probs_df.copy()
    df["conference"] = np.where(df["team"].isin(list(EAST)), "East", "West")
    df = df.sort_values(["conference", "pr_r4"], ascending=[True, False])

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
    for i, conf in enumerate(["East", "West"]):
        sub = df[df["conference"] == conf].head(10)
        sns.barplot(data=sub, x="pr_r4", y="team", ax=axes[i], palette="crest")
        axes[i].set_title(f"{conf} — Title Probabilities (Top 10)")
        axes[i].set_xlabel("Title Probability")
        axes[i].set_ylabel("")
    fig.suptitle(f"{season} Bracket Projection — Monte Carlo")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"conference_projection_{season}.png", dpi=300)
    plt.close(fig)


def main() -> None:
    ratings = load_team_ratings()
    seasons = sorted(ratings["season"].unique())

    all_probs = []
    for season in seasons:
        probs_df, routes = run_season_bracket(ratings, season, sims=5000)
        all_probs.append(probs_df)
        # Save per-season details
        probs_df.to_csv(FIG_DIR / f"bracket_probabilities_{season}.csv", index=False)
        opp = summarize_opponents(routes)
        # Choose top contenders by title probability
        top = probs_df.sort_values("pr_r4", ascending=False).head(4)
        for row in top.itertuples(index=False):
            t = row.team
            tree_df = opp.get(t, pd.DataFrame())
            try:
                plot_probability_tree(season, t, tree_df, row._asdict())
            except Exception:
                pass
        plot_conference_projection(season, probs_df)

    if all_probs:
        pd.concat(all_probs, ignore_index=True).to_csv(FIG_DIR / "bracket_probabilities_all_seasons.csv", index=False)


if __name__ == "__main__":
    main()
