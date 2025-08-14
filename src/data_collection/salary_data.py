"""
Salary Data Collection Module

Collects NBA player salary data from Basketball Reference and other sources.
"""

import os
import re
import time
import random
import logging
import requests
import pandas as pd
from bs4 import BeautifulSoup, Comment
from typing import List, Optional
from io import StringIO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SalaryDataCollector:
    """Collects NBA salary data from various sources."""

    def __init__(self, cache_dir: str = "cache_bbr"):
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/115.0.0.0 Safari/537.36"
                )
            }
        )
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _fetch_html(self, url: str) -> str:
        """
        Fetch HTML content with caching and safe delays.
        """
        filename = url.replace("https://www.basketball-reference.com/teams/", "")
        filename = filename.replace("/", "_")
        filepath = os.path.join(self.cache_dir, filename)

        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()

        logger.info(f"Fetching: {url}")
        resp = self.session.get(url)
        resp.raise_for_status()
        html = resp.text

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)

        time.sleep(random.uniform(3, 6))  # Safe delay
        return html

    def collect_basketball_reference_team_salaries(
        self, season: int, teams: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Collect salary data for every player on each team for a given season.

        Args:
            season: NBA season year (e.g., 2024 for 2023-24 season)
            teams: Optional list of team abbreviations (e.g., ["LAL", "BOS"])

        Returns:
            DataFrame with player salary information.
        """
        if teams is None:
            teams = [
                "ATL",
                "BOS",
                "BRK",
                "CHO",
                "CHI",
                "CLE",
                "DAL",
                "DEN",
                "DET",
                "GSW",
                "HOU",
                "IND",
                "LAC",
                "LAL",
                "MEM",
                "MIA",
                "MIL",
                "MIN",
                "NOP",
                "NYK",
                "OKC",
                "ORL",
                "PHI",
                "PHO",
                "POR",
                "SAC",
                "SAS",
                "TOR",
                "UTA",
                "WAS",
            ]

        all_frames = []

        for team in teams:
            url = f"https://www.basketball-reference.com/teams/{team}/{season}.html"
            try:
                html = self._fetch_html(url)
                df = self._parse_salary_table(html, team, season)
                if df is not None:
                    all_frames.append(df)
            except Exception as e:
                logger.error(f"Failed to collect {team} {season} salaries: {e}")

        if not all_frames:
            return pd.DataFrame()

        final_df = pd.concat(all_frames, ignore_index=True)
        logger.info(
            f"Collected {len(final_df)} salary rows for {season - 1}-{str(season)[2:]} season"
        )
        return final_df

    def _parse_salary_table(
        self, html: str, team_abbr: str, season: int
    ) -> Optional[pd.DataFrame]:
        """
        Parse the team salary table from Basketball Reference HTML.

        Args:
            html: Raw HTML content
            team_abbr: Team abbreviation
            season: NBA season year

        Returns:
            DataFrame or None if no table found.
        """
        soup = BeautifulSoup(html, "html.parser")

        salary_table = None
        for div_id in ["all_salaries", "all_salaries2"]:
            div = soup.find("div", id=div_id)
            if div:
                comment = div.find(string=lambda text: isinstance(text, Comment))
                if comment:
                    comment_soup = BeautifulSoup(str(comment), "html.parser")
                    table = comment_soup.find("table")
                    if table:
                        salary_table = pd.read_html(StringIO(str(table)))[0]
                        break

        if salary_table is None or salary_table.empty:
            return None

        # Normalize column names
        salary_table.columns = [
            str(c).strip().lower().replace(" ", "_") for c in salary_table.columns
        ]

        # Clean salary column
        if "salary" in salary_table.columns:
            salary_table["salary"] = (
                salary_table["salary"]
                .replace(r"[\$,]", "", regex=True)
                .replace("", None)
                .astype(float)
            )

        salary_table["team_id"] = team_abbr
        salary_table["season_year"] = season

        return salary_table


def collect_salary_data_for_range(start_year: int, end_year: int) -> pd.DataFrame:
    """
    Collect salary data for all NBA teams from Basketball Reference between two seasons.

    Args:
        start_year: First season year
        end_year: Last season year

    Returns:
        Combined DataFrame with all salaries.
    """
    collector = SalaryDataCollector()
    all_data = []

    for year in range(start_year, end_year + 1):
        df = collector.collect_basketball_reference_team_salaries(year)
        if not df.empty:
            all_data.append(df)

    if not all_data:
        return pd.DataFrame()

    final_df = pd.concat(all_data, ignore_index=True)
    final_df.to_csv(f"nba_salaries_{start_year}_{end_year}.csv", index=False)
    logger.info(
        f"Saved {len(final_df)} rows to nba_salaries_{start_year}_{end_year}.csv"
    )
    return final_df


if __name__ == "__main__":
    df_all = collect_salary_data_for_range(2020, 2025)
    print(df_all.head())

