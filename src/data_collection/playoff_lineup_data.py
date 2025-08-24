#!/usr/bin/env python3
"""
NBA Lineup BPM Pipeline (Unified)
---------------------------------

This script consolidates the functionality of three separate scripts
(`lineup_scraper.py`, `combiner.py` and `BPM_Calculator.py`) into a
single program. It performs the following steps:

1. **Scrape** lineup statistics from the NBA Stats API for multiple
   seasons and categories.  The data are requested directly from
   ``stats.nba.com/stats/leaguedashlineups`` using asynchronous HTTP
   requests with realistic headers and retry/backoff logic to handle
   throttling.

2. **Merge and Normalize** the scraped data.  Traditional counting
   statistics (e.g. points, rebounds, assists) are merged with
   possession and efficiency metrics from the "Advanced" category.
   Counting stats are then converted to per‑100 possession rates.  The
   script also joins additional statistics from the Four Factors,
   Miscellaneous, Scoring and Opponent categories when available.

3. **Model Training**.  A Ridge, Lasso or ElasticNet regression model
   (selected via cross‑validated grid search) is trained to predict
   lineup ``NET_RATING`` using the per‑100 and percentage statistics as
   features.  Out‑of‑fold predictions and final model predictions are
   computed for each lineup.  The model coefficients are saved for
   inspection.

4. **Output** a single CSV file ``lineup_bpm.csv`` containing the
   merged dataset along with the model's out‑of‑fold and final
   predicted lineup BPM values.  A separate CSV of model coefficients
   is also written.

Usage:
    python lineup_bpm_pipeline.py

The script relies on ``pandas``, ``aiohttp``, ``numpy`` and
``scikit‑learn``.  If those libraries are not installed you can
install them with ``pip install pandas aiohttp brotli scikit‑learn``.

Note:
    The NBA stats API enforces fairly strict rate limits and may
    sometimes return ``403`` or ``429`` responses.  This program
    includes user‑agent rotation and exponential backoff to handle
    temporary errors, but scraping can still take several minutes
    depending on network conditions.  If you encounter persistent
    scraping issues, try reducing the ``CONCURRENCY`` value or
    adjusting the seasons and categories to a smaller subset while
    developing.
"""

import asyncio
import aiohttp
import os
import random
import time
import json
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import GroupKFold, GridSearchCV, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer




##############################################################################
# Configuration
##############################################################################

# Seasons to scrape (edit as needed).  New seasons can be appended to the
# list.  Note that stats.nba.com uses strings like "2024-25" to refer to
# seasons that span two calendar years.
SEASONS: List[str] = [
    "2024-25",
    "2023-24",
    "2022-23",
    "2021-22",
    "2020-21",
]

# Categories of lineup statistics to request.  Each category corresponds
# to a tab on the lineup stats page.  The mapping below converts the
# human‑friendly names into the API's ``MeasureType`` parameter.  You
# can remove or add categories here, but the merge logic later in the
# script expects at least the "Traditional" and "Advanced" categories.
CATEGORIES: List[str] = [
    "Traditional",
    "Advanced",
    "Four Factors",
    "Misc",
    "Scoring",
    "Opponent",
]

# Mapping from UI category names to the measure types required by the API.
# The key names should match those in CATEGORIES; the values are the
# strings accepted by the API.  "Opponent" covers the opponent scoring
# view on the website.
CATEGORY_MAP: Dict[str, str] = {
    "Traditional": "Base",
    "Advanced": "Advanced",
    "Four Factors": "Four Factors",
    "Misc": "Misc",
    "Scoring": "Scoring",
    "Opponent": "Opponent",
}

# Output directory to hold intermediate and final CSVs.  All CSVs
# written by this program will be placed under this directory.
SAVE_ALL_CSVS = False
SAVE_PER100_CSV = False

# Save data files to a folder inside the repo (relative path)


OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))




# Endpoint and base query parameters for the NBA lineup stats API.
BASE_URL = "https://stats.nba.com/stats/leaguedashlineups"
BASE_PARAMS: Dict[str, Any] = {
    "SeasonType": "Playoffs",
    "PerMode": "Totals",
    "PlusMinus": "N",
    "PaceAdjust": "N",
    "Rank": "N",
    "LeagueID": "00",
    "GroupQuantity": 5,
    "DateFrom": "",
    "DateTo": "",
    "GameScope": "",
    "Location": "",
    "Month": 0,
    "OpponentTeamID": 0,
    "Outcome": "",
    "SeasonSegment": "",
    "ShotClockRange": "",
    "VsConference": "",
    "VsDivision": "",
    "Conference": "",
    "Division": "",
    "LastNGames": 0,
    "Period": 0,
    "PORound": 0,
}

# Concurrency and retry settings for the async scraper.  Adjust
# ``CONCURRENCY`` down if you see frequent 403/429 errors; increasing it
# may speed up scraping but could trigger server throttling.
CONCURRENCY = 3
MAX_RETRIES = 4
REQUEST_TIMEOUT = 30  # seconds
DELAY_JITTER_RANGE = (0.05, 0.2)  # small random delay between requests


# User‑agent pool used to rotate the ``User-Agent`` header on each
# request.  Rotating agents helps to avoid some trivial blocking.  Feel
# free to add more recent UA strings here.
UA_POOL: List[str] = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
]


##############################################################################
# Helper functions for scraping
##############################################################################

def referer_for_category(category: str) -> str:
    """Return a referer URL appropriate for a category.

    The NBA stats site uses the page referer to determine which layout
    was requested.  While not strictly necessary, providing the referer
    that corresponds to the requested category helps mimic real user
    browsing and reduces the chance of blocking.
    """
    base = "https://www.nba.com/stats/lineups/"
    page = {
        "Traditional": "traditional",
        "Advanced": "advanced",
        "Four Factors": "four-factors",
        "Misc": "misc",
        "Scoring": "scoring",
        "Opponent": "opponent",
    }.get(category, "advanced")
    return f"{base}{page}"


def make_headers(category: str) -> Dict[str, str]:
    """Construct a realistic set of headers for the NBA stats API.

    The API uses headers like ``Origin``, ``Referer`` and
    ``x-nba-stats-origin`` to validate requests.  Rotating the
    ``User-Agent`` also helps distribute requests.  Pass the category
    so the referer can be set correctly.
    """
    return {
        "Accept": "application/json, text/plain, */*",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en-US,en;q=0.9",
        "Origin": "https://www.nba.com",
        "Referer": referer_for_category(category),
        "User-Agent": random.choice(UA_POOL),
        "Connection": "keep-alive",
        "x-nba-stats-origin": "stats",
        "x-nba-stats-token": "true",
        # Additional headers that a real browser would include.  While
        # optional, they help avoid certain 403 responses on some
        # systems.  The values below are typical for modern browsers.
        "Sec-Fetch-Site": "same-site",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Dest": "empty",
    }


def make_params(season: str, category: str) -> Dict[str, Any]:
    """Assemble the query parameters for a season/category pair."""
    p: Dict[str, Any] = dict(BASE_PARAMS)
    p["Season"] = season
    p["MeasureType"] = CATEGORY_MAP[category]
    return p


def parse_payload_to_df(payload: Dict[str, Any]) -> pd.DataFrame:
    """Parse the JSON payload returned by the API into a DataFrame.

    The API occasionally returns either a ``resultSets`` list or a
    single ``resultSet`` object.  This helper normalizes both shapes
    and assigns column names.  Column names are stripped of leading and
    trailing whitespace to avoid duplicates like " NET_RATING".
    """
    rs = payload.get("resultSets", [payload.get("resultSet")])[0]
    df = pd.DataFrame(rs["rowSet"], columns=[c.strip() for c in rs["headers"]])
    return df


def concat_align(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    if not dfs:
        return pd.DataFrame()

    # ✅ Always force Season column into col_order first
    col_order: List[str] = ["Season"] + [c for c in dfs[0].columns if c != "Season"]

    for df in dfs[1:]:
        for c in df.columns:
            if c not in col_order:
                col_order.append(c)

    aligned: List[pd.DataFrame] = [df.reindex(columns=col_order) for df in dfs]

    combined = pd.concat(aligned, ignore_index=True)

    # ✅ Ensure Season column isn’t all NaN
    if "Season" in combined.columns:
        combined["Season"] = combined["Season"].fillna("Unknown")

    return combined



##############################################################################
# Asynchronous scraping logic
##############################################################################

async def fetch_one(
    session: aiohttp.ClientSession,
    season: str,
    category: str,
    sem: asyncio.Semaphore,
) -> Tuple[str, Optional[pd.DataFrame]]:
    """Fetch one (season, category) dataset with retries and backoff.

    Returns a tuple of (category, DataFrame) so that downstream logic
    can group results by category.  If scraping ultimately fails, the
    DataFrame will be ``None``.  The semaphore limits the number of
    concurrent requests to the server.
    """
    params = make_params(season, category)
    for attempt in range(1, MAX_RETRIES + 1):
        # small random jitter to avoid request bursts
        await asyncio.sleep(random.uniform(*DELAY_JITTER_RANGE))
        try:
            async with sem:
                # Set a timeout on the request to avoid hanging
                timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
                async with session.get(
                    BASE_URL,
                    params=params,
                    headers=make_headers(category),
                    timeout=timeout,
                ) as resp:
                    text = await resp.text()
                    if resp.status == 200:
                        try:
                            payload = json.loads(text)
                            df = parse_payload_to_df(payload)
                            df["Season"] = season
                            df["Category"] = category
                            print(f"Fetched {season} | {category:<12}")
                            return category, df
                        except Exception as e:
                            print(f"JSON parse error {season}/{category}: {e}\nBody[:200]={text[:200]!r}")
                            # Fall through to retry
                    else:
                        # Non‑200 response; treat certain codes as retryable
                        print(f"HTTP {resp.status} for {season}/{category}")
                        if resp.status in {429, 500, 502, 503, 504, 403}:
                            raise RuntimeError(f"Retryable status {resp.status}")
                        # For other codes, don't retry
                        return category, None

        except Exception as e:
            if attempt == MAX_RETRIES:
                print(f"FAILED {season}/{category} after {attempt} attempts: {e}")
                return category, None
            backoff = 1.2 * (2 ** (attempt - 1)) + random.uniform(0.1, 0.6)
            print(f"Retry {attempt}/{MAX_RETRIES} for {season}/{category} in {backoff:.2f}s ({e})")
            await asyncio.sleep(backoff)

    return category, None  # Should not reach here


async def scrape_all() -> Dict[str, pd.DataFrame]:
    """Scrape all seasons and categories defined in the config.

    Returns a dictionary mapping category names to a single DataFrame
    containing all seasons concatenated.  Categories that fail to
    return any data will be omitted from the result.
    """
    sem = asyncio.Semaphore(CONCURRENCY)
    connector = aiohttp.TCPConnector(limit_per_host=CONCURRENCY)
    results: Dict[str, List[pd.DataFrame]] = defaultdict(list)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks: List[asyncio.Task] = []
        for season in SEASONS:
            for category in CATEGORIES:
                tasks.append(asyncio.create_task(fetch_one(session, season, category, sem)))
        responses = await asyncio.gather(*tasks)
    for category, df in responses:
        if category and df is not None and not df.empty:
            results[category].append(df)
    # Concatenate all DataFrames for each category
    combined: Dict[str, pd.DataFrame] = {}
    for category, df_list in results.items():
        combined_df = concat_align(df_list)
        combined_df = combined_df.reset_index(drop=True)
        combined[category] = combined_df
    return combined


##############################################################################
# Data merging and per‑100 possession normalization
##############################################################################

def merge_trad_and_adv(
    trad: pd.DataFrame,
    adv: pd.DataFrame,
) -> pd.DataFrame:
    """Merge Traditional and Advanced stats and compute per‑100 rates.

    The Traditional dataset contains raw counting statistics such as
    ``PTS``, ``AST``, etc., while the Advanced dataset contains
    possessions (``POSS``) and efficiency metrics like ``NET_RATING``.
    The merge is performed on season, team abbreviation and lineup
    group name.  After merging, per‑100 possession columns are
    computed for a list of candidate stats.  Rows lacking a ``POSS``
    value are dropped to avoid division by zero.
    """
    keys = ["Season", "TEAM_ABBREVIATION", "GROUP_NAME"]
    # Ensure join keys are stripped strings to avoid mismatches
    for df in (trad, adv):
        for k in keys:
            if k in df.columns and df[k].dtype == object:
                df[k] = df[k].astype(str).str.strip()
    adv_cols = keys + [c for c in ["POSS", "OFF_RATING", "DEF_RATING", "NET_RATING", "PACE"] if c in adv.columns]
    merged = pd.merge(trad, adv[adv_cols], on=keys, how="inner")
    # Compute per‑100 stats
    per100_candidates = [
        "PTS", "AST", "REB", "STL", "BLK", "TOV",
        "OREB", "DREB", "FGM", "FGA", "FG3M", "FG3A",
    ]
    for stat in per100_candidates:
        if stat in merged.columns and "POSS" in merged.columns:
            merged[f"{stat}_per100"] = (merged[stat] / merged["POSS"]) * 100
    merged = merged.dropna(subset=["POSS"])  # drop rows without possessions
    return merged


def safe_merge(left: pd.DataFrame, right: Optional[pd.DataFrame], keys: List[str]) -> pd.DataFrame:
    """Merge ``right`` into ``left`` on the given keys, avoiding column collisions.

    If ``right`` is None or empty, the left DataFrame is returned
    unmodified.  Duplicate columns (other than the join keys) in the
    right DataFrame are dropped before merging.
    """
    if right is None or right.empty:
        return left
    # Drop duplicate columns on the right (except join keys)
    dupes = [c for c in right.columns if c in left.columns and c not in keys]
    right_clean = right.drop(columns=dupes)
    return pd.merge(left, right_clean, on=keys, how="left")


##############################################################################
# Feature table construction
##############################################################################

def build_feature_table(
    category_data: Dict[str, pd.DataFrame],
    min_poss: int = 100
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, List[str], Optional[np.ndarray], Optional[np.ndarray]]:
    """Construct the modeling dataset from scraped category DataFrames."""

    # Ensure required categories exist
    if "Traditional" not in category_data or "Advanced" not in category_data:
        raise ValueError("Traditional and Advanced categories are required for merging.")

    # Merge traditional and advanced
    merged = merge_trad_and_adv(category_data["Traditional"], category_data["Advanced"])

    # Optional categories
    keys = ["Season", "TEAM_ABBREVIATION", "GROUP_NAME"]
    optional_categories = {
        "Four Factors": ["EFG_PCT", "TM_TOV_PCT", "OREB_PCT", "FTA_RATE"],
        "Misc": ["PTS_OFF_TOV", "PTS_2ND_CHANCE", "PTS_FB", "PTS_PAINT"],
        "Scoring": ["PCT_FGA_2PT", "PCT_FGA_3PT", "PCT_PTS_2PT", "PCT_PTS_3PT",
                    "PCT_PTS_FT", "PCT_AST_FGM"],
        "Opponent": ["OPP_PTS_FB", "OPP_PTS_PAINT", "OPP_PTS_OFF_TOV"],
    }
    for cat, cols in optional_categories.items():
        if cat in category_data:
            df_right = category_data[cat]
            # Strip whitespace on join keys
            for k in keys:
                if k in df_right.columns and df_right[k].dtype == object:
                    df_right[k] = df_right[k].astype(str).str.strip()
            cols_keep = keys + [c for c in cols if c in df_right.columns]
            merged = safe_merge(merged, df_right[cols_keep], keys)

    # Drop rows with no possessions or low possessions
    if "POSS" in merged.columns:
        merged["POSS"] = pd.to_numeric(merged["POSS"], errors="coerce")
        merged = merged.dropna(subset=["POSS"])
        merged = merged[merged["POSS"] > 0]
        if min_poss is not None:
            merged = merged[merged["POSS"] >= min_poss]

    # 🔥 Drop duplicate lineups BEFORE feature construction
    merged = merged.drop_duplicates(
        subset=["Season", "TEAM_ABBREVIATION", "GROUP_NAME"]
    ).reset_index(drop=True)

    # Feature selection
    per100_features = [c for c in merged.columns if c.endswith("_per100")]
    percent_features = [c for c in [
        "FG_PCT", "FG3_PCT", "FT_PCT",
        "EFG_PCT", "TM_TOV_PCT", "OREB_PCT", "FTA_RATE",
        "PCT_FGA_2PT", "PCT_FGA_3PT", "PCT_PTS_2PT", "PCT_PTS_3PT",
        "PCT_PTS_FT", "PCT_AST_FGM",
        "OPP_PTS_FB", "OPP_PTS_PAINT", "OPP_PTS_OFF_TOV",
    ] if c in merged.columns]
    feature_cols = per100_features + percent_features

    # Avoid target leakage
    leakage_cols = {"NET_RATING", "OFF_RATING", "DEF_RATING",
                    "PLUS_MINUS", "W", "L", "W_PCT"}
    feature_cols = [c for c in feature_cols if c not in leakage_cols]

    if "NET_RATING" not in merged.columns:
        raise ValueError("NET_RATING missing from merged data.")

    y = merged["NET_RATING"].astype(float)
    X = merged[feature_cols].astype(float)

    sample_weight: Optional[np.ndarray] = (
        merged["POSS"].values if "POSS" in merged.columns else None
    )
    groups: Optional[np.ndarray] = (
        merged["Season"].astype(str).values if "Season" in merged.columns else None
    )

    return merged, X, y, feature_cols, sample_weight, groups



##############################################################################
# Model training and evaluation
##############################################################################

def fit_best_model(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: Optional[np.ndarray] = None,
    groups: Optional[np.ndarray] = None,
) -> GridSearchCV:
    """Fit a tuned regression model using cross‑validated grid search.

    The model pipeline imputes missing values, scales the features and
    fits either a Ridge, Lasso or ElasticNet regression.  A grid of
    hyper‑parameters is explored and the best model is selected based
    on negative mean squared error.  Grouped cross‑validation is used
    when ``groups`` are provided.
    """
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scale", StandardScaler(with_mean=True, with_std=True)),
        ("model", Ridge()),
    ])
    param_grid = [
        {"model": [Ridge()], "model__alpha": np.logspace(-3, 3, 13)},
        {"model": [Lasso(max_iter=10000)], "model__alpha": np.logspace(-3, 1, 9)},
        {"model": [ElasticNet(max_iter=10000)], "model__alpha": np.logspace(-3, 2, 11), "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]},
    ]
    if groups is not None and len(np.unique(groups)) > 1:
        n_splits = min(len(np.unique(groups)), 5)
        cv = GroupKFold(n_splits=n_splits)
    else:
        cv = 5
    gscv = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    if sample_weight is not None:
        gscv.fit(X, y, model__sample_weight=sample_weight, groups=groups)
    else:
        gscv.fit(X, y, groups=groups)
    return gscv


def evaluate_oof(
    estimator: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: Optional[np.ndarray] = None,
    groups: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, float, float]:
    """Generate out‑of‑fold predictions and compute evaluation metrics."""
    if groups is not None and len(np.unique(groups)) > 1:
        n_splits = min(len(np.unique(groups)), 5)
        cv = GroupKFold(n_splits=n_splits)
    else:
        cv = 5
    oof_pred = cross_val_predict(
        estimator, X, y, cv=cv, n_jobs=-1, groups=groups
    )
    r2 = r2_score(y, oof_pred)
    mae = mean_absolute_error(y, oof_pred)
    rmse = np.sqrt(mean_squared_error(y, oof_pred))
    return oof_pred, r2, mae, rmse


##############################################################################
# Main driver
##############################################################################

def main() -> None:
    print("Scraping lineup data from NBA stats API...")
    category_data = asyncio.run(scrape_all())
    if not category_data:
        print("No data retrieved. Exiting.")
        return

    if SAVE_ALL_CSVS:
        for cat, df in category_data.items():
            # Drop unwanted ID columns
            drop_cols = ["GROUP_SET", "GROUP_ID", "TEAM_ID"]
            df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

            # 🔥 Ensure Season is always present
            if "Season" not in df.columns:
                df["Season"] = df.get("Season", pd.Series(["Unknown"] * len(df)))

            out_path = os.path.join(
                OUTPUT_DIR, f"lineups_{cat.lower().replace(' ', '_')}.csv"
            )
            df.to_csv(out_path, index=False)
            print(f"Saved cleaned {cat} data to {out_path} (rows={len(df)})")

    else:
        for cat, df in category_data.items():
            print(f"Loaded raw {cat} data (rows={len(df)})")

    # Step 2: build features
    df, X, y, feature_cols, sample_weight, groups = build_feature_table(category_data)
        # Optionally save the per-100 possession feature table
    if SAVE_PER100_CSV:
        drop_cols = ["GROUP_SET", "GROUP_ID", "TEAM_ID"]
        df_clean = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

        per100_path = os.path.join(OUTPUT_DIR, "lineups_per100.csv")
        df_clean.to_csv(per100_path, index=False)
        print(f"Saved per-100 possession feature table to {per100_path} (rows={len(df_clean)})")



    # Step 3: fit tuned model
    gscv = fit_best_model(X, y, sample_weight=sample_weight, groups=groups)
    best_est = gscv.best_estimator_
    oof_pred, r2, mae, rmse = evaluate_oof(
        best_est, X, y, sample_weight=sample_weight, groups=groups
    )

    # Train final model
    if sample_weight is not None:
        best_est.fit(X, y, model__sample_weight=sample_weight)
    else:
        best_est.fit(X, y)
    final_pred = best_est.predict(X)


    # Assemble output with BPM predictions
    out = df.copy()
    out["Lineup_BPM_OOF"] = oof_pred
    out["Lineup_BPM_Final"] = final_pred
    if "Season" in out.columns:
        out["Lineup_BPM_Final_Demeaned"] = (
            out["Lineup_BPM_Final"]
            - out.groupby("Season")["Lineup_BPM_Final"].transform("mean")
        )
        out["Lineup_BPM_OOF_Demeaned"] = (
            out["Lineup_BPM_OOF"]
            - out.groupby("Season")["Lineup_BPM_OOF"].transform("mean")
        )

    # 🔥 Clean final BPM file
    drop_cols = ["GROUP_SET", "GROUP_ID", "TEAM_ID"]
    out = out.drop(columns=[c for c in out.columns if c in drop_cols], errors="ignore")

    # 🔥 Ensure Season is always present
    if "Season" not in out.columns and "Season" in df.columns:
        out["Season"] = df["Season"]
    elif "Season" not in out.columns:
        out["Season"] = "Unknown"

    bpm_path = os.path.join(OUTPUT_DIR, "lineup_bpm.csv")
    out.to_csv(bpm_path, index=False)
    print(f"Pipeline completed successfully. Final BPM saved to {bpm_path}")








if __name__ == "__main__":
    main()