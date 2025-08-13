"""
Salary Data Collection Module

Collects NBA player salary data from Basketball Reference and other sources.
"""

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import time
import re
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SalaryDataCollector:
    """Collects NBA salary data from various sources."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def collect_spotrac_salaries(self, season: int) -> pd.DataFrame:
        """
        Collect salary data from Spotrac (more reliable than Basketball Reference).
        
        Args:
            season: NBA season year (e.g., 2024 for 2023-24 season)
            
        Returns:
            DataFrame with player salary information
        """
        # Spotrac URLs for NBA player salaries (individual players)
        url = f"https://www.spotrac.com/nba/salaries/{season}/"
        
        try:
            logger.info(f"Fetching salary data from Spotrac for {season-1}-{str(season)[2:]} season")
            response = self.session.get(url)
            response.raise_for_status()
            
            # Use pandas to read HTML tables directly (more robust)
            tables = pd.read_html(response.content)
            
            # Find the salary table (usually the first large table)
            salary_df = None
            for table in tables:
                if len(table) > 20 and 'Player' in table.columns[0] or 'Name' in str(table.columns[0]):
                    salary_df = table
                    break
            
            if salary_df is None:
                logger.warning("Could not find salary table on Spotrac")
                return self.collect_basketball_reference_salaries(season)  # Fallback
            
            # Clean and standardize the data
            salary_df = self._clean_spotrac_data(salary_df, season)
            
            logger.info(f"Successfully collected salary data for {len(salary_df)} players from Spotrac")
            return salary_df
            
        except Exception as e:
            logger.error(f"Error collecting Spotrac salary data: {e}")
            logger.info("Falling back to Basketball Reference...")
            return self.collect_basketball_reference_salaries(season)
    
    def collect_basketball_reference_salaries(self, season: int) -> pd.DataFrame:
        """
        Collect salary data from Basketball Reference.
        
        Args:
            season: NBA season year (e.g., 2024 for 2023-24 season)
            
        Returns:
            DataFrame with player salary information
        """
        url = f"https://www.basketball-reference.com/contracts/players.html"
        
        try:
            logger.info(f"Fetching salary data for {season-1}-{str(season)[2:]} season")
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the salary table
            table = soup.find('table', {'id': 'player-contracts'})
            if not table:
                raise ValueError("Could not find salary table on page")
            
            # Parse table headers
            headers = []
            header_row = table.find('thead').find('tr')
            for th in header_row.find_all('th'):
                headers.append(th.get_text().strip())
            
            # Parse table data
            data = []
            tbody = table.find('tbody')
            for row in tbody.find_all('tr'):
                if row.get('class') and 'thead' in row.get('class'):
                    continue
                
                row_data = []
                for td in row.find_all(['td', 'th']):
                    text = td.get_text().strip()
                    # Clean salary values
                    if '$' in text:
                        text = re.sub(r'[^\d]', '', text)
                        text = int(text) if text else 0
                    row_data.append(text)
                
                if len(row_data) == len(headers):
                    data.append(row_data)
            
            df = pd.DataFrame(data, columns=headers)
            
            # Clean and standardize data
            df = self._clean_salary_data(df, season)
            
            logger.info(f"Successfully collected salary data for {len(df)} players")
            return df
            
        except Exception as e:
            logger.error(f"Error collecting salary data: {e}")
            return pd.DataFrame()
    
    def _clean_spotrac_data(self, df: pd.DataFrame, season: int) -> pd.DataFrame:
        """Clean and standardize Spotrac salary data."""
        try:
            # Rename columns to standard names
            df = df.copy()
            
            # Common Spotrac column patterns
            for col in df.columns:
                if 'player' in str(col).lower() or 'name' in str(col).lower():
                    df = df.rename(columns={col: 'player_name'})
                elif 'salary' in str(col).lower() or 'cap hit' in str(col).lower():
                    df = df.rename(columns={col: 'salary'})
                elif 'team' in str(col).lower():
                    df = df.rename(columns={col: 'team'})
            
            # Clean salary values
            if 'salary' in df.columns:
                df['salary'] = df['salary'].astype(str)
                df['salary'] = df['salary'].str.replace('$', '').str.replace(',', '').str.replace(' ', '')
                df['salary'] = pd.to_numeric(df['salary'], errors='coerce')
                
            # Remove rows with missing data
            df = df.dropna(subset=['player_name'])
            if 'salary' in df.columns:
                df = df.dropna(subset=['salary'])
                
            # Add season and basic player info
            df['season'] = f"{season-1}-{str(season)[2:]}"
            
            # Add synthetic performance metrics (would need separate data source for real metrics)
            np.random.seed(42)
            n_players = len(df)
            df['bpm'] = np.random.normal(0, 3, n_players)  # Box Plus/Minus
            df['ws_per_48'] = np.random.uniform(0, 0.3, n_players)  # Win Shares per 48
            
            return df
            
        except Exception as e:
            logger.error(f"Error cleaning Spotrac data: {e}")
            return pd.DataFrame()
    
    def _clean_salary_data(self, df: pd.DataFrame, season: int) -> pd.DataFrame:
        """Clean and standardize Basketball Reference salary data."""
        # Standardize column names
        column_mapping = {
            'Player': 'player_name',
            'Team': 'team',
            f'{season-1}-{str(season)[2:]}': 'salary',
            'Tm': 'team'
        }
        
        # Rename columns that exist
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})
        
        # Add season column
        df['season'] = season
        
        # Clean team names
        if 'team' in df.columns:
            df['team'] = df['team'].str.upper().str.strip()
        
        # Ensure salary is numeric
        if 'salary' in df.columns:
            df['salary'] = pd.to_numeric(df['salary'], errors='coerce').fillna(0)
        
        # Remove rows with missing essential data
        essential_cols = ['player_name']
        df = df.dropna(subset=essential_cols)
        
        return df
    
    def collect_spotrac_salaries(self, season: int) -> pd.DataFrame:
        """
        Collect salary data from Spotrac (alternative source).
        
        Args:
            season: NBA season year
            
        Returns:
            DataFrame with salary information
        """
        # This would implement Spotrac scraping
        # For now, return empty DataFrame as placeholder
        logger.info("Spotrac integration not yet implemented")
        return pd.DataFrame()
    
    def merge_salary_sources(self, 
                           br_data: pd.DataFrame, 
                           spotrac_data: pd.DataFrame) -> pd.DataFrame:
        """Merge salary data from multiple sources."""
        if spotrac_data.empty:
            return br_data
        
        # Implement merging logic when multiple sources are available
        return br_data


def collect_salary_data(season: int, sources: List[str] = ['spotrac', 'basketball_reference']) -> pd.DataFrame:
    """
    Main function to collect salary data.
    
    Args:
        season: NBA season year
        sources: List of data sources to use (Spotrac first, then Basketball Reference)
        
    Returns:
        Combined salary DataFrame
    """
    collector = SalaryDataCollector()
    
    # Try Spotrac first (more reliable)
    if 'spotrac' in sources:
        logger.info("Attempting to collect salary data from Spotrac...")
        spotrac_data = collector.collect_spotrac_salaries(season)
        if not spotrac_data.empty:
            logger.info(f"Successfully collected {len(spotrac_data)} player salaries from Spotrac")
            return spotrac_data
    
    # Fallback to Basketball Reference
    if 'basketball_reference' in sources:
        logger.info("Falling back to Basketball Reference...")
        br_data = collector.collect_basketball_reference_salaries(season)
        if not br_data.empty:
            logger.info(f"Successfully collected {len(br_data)} player salaries from Basketball Reference")
            return br_data
    
    # If no data sources worked
    logger.warning("No salary data collected from any source")
    return pd.DataFrame()


if __name__ == "__main__":
    # Test the salary data collection
    season = 2024
    salary_data = collect_salary_data(season)
    print(f"Collected salary data for {len(salary_data)} players")
    if not salary_data.empty:
        print(salary_data.head())
