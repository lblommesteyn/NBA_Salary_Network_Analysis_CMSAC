"""
Lineup and On-Court Minutes Data Collection Module

Collects NBA lineup data and shared on-court minutes using NBA API and pbpstats.
"""

import pandas as pd
import numpy as np
from nba_api.stats.endpoints import (
    teamgamelog, leaguegamefinder, playergamelog, 
    boxscoreadvancedv2, boxscoretraditionalv2, teamplayerdashboard
)
from nba_api.stats.static import teams, players
import requests
import time
import logging
from typing import Dict, List, Tuple, Optional
from itertools import combinations

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LineupDataCollector:
    """Collects NBA lineup and on-court minutes data."""
    
    def __init__(self):
        self.teams_info = teams.get_teams()
        self.request_delay = 0.6  # NBA API rate limiting
    
    def get_team_roster(self, team_id: int, season: str) -> pd.DataFrame:
        """
        Get team roster for a given season.
        
        Args:
            team_id: NBA team ID
            season: Season string (e.g., '2023-24')
            
        Returns:
            DataFrame with team roster information
        """
        try:
            time.sleep(self.request_delay)
            
            # Convert season format: '2023-24' -> '2023-24'
            # NBA API expects this format
            
            # Get team player dashboard
            dashboard = teamplayerdashboard.TeamPlayerDashboard(
                team_id=team_id,
                season=season,
                season_type_all_star='Regular Season'
            )
            
            roster_df = dashboard.get_data_frames()[1]  # Players stats
            
            # Clean and standardize
            roster_df = roster_df.rename(columns={
                'PLAYER_NAME': 'player_name',
                'PLAYER_ID': 'player_id',
                'MIN': 'minutes_played'
            })
            
            roster_df['team_id'] = team_id
            roster_df['season'] = season
            
            return roster_df
            
        except Exception as e:
            logger.error(f"Error getting roster for team {team_id}: {e}")
            return pd.DataFrame()
    
    def get_team_games(self, team_id: int, season: str) -> List[str]:
        """Get all game IDs for a team in a season."""
        try:
            time.sleep(self.request_delay)
            
            # Use leaguegamefinder which is more reliable for team games
            gamefinder = leaguegamefinder.LeagueGameFinder(
                team_id_nullable=team_id,
                season_nullable=season,
                season_type_nullable='Regular Season'
            )
            
            games_df = gamefinder.get_data_frames()[0]
            if games_df.empty:
                logger.warning(f"No games found for team {team_id} in season {season}")
                return []
                
            # Get unique game IDs (each game appears twice - home and away)
            game_ids = games_df['GAME_ID'].unique().tolist()
            logger.info(f"Found {len(game_ids)} games for team {team_id}")
            return game_ids
            
        except Exception as e:
            logger.error(f"Error getting games for team {team_id}: {e}")
            return []
    
    def get_game_lineups(self, game_id: str) -> pd.DataFrame:
        """
        Extract lineup information from a game using boxscore data.
        This is a simplified approach - real lineup data would need play-by-play parsing.
        """
        try:
            time.sleep(self.request_delay)
            
            # Get traditional boxscore
            boxscore = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
            player_stats = boxscore.get_data_frames()[0]
            
            # Filter to only players who played
            player_stats = player_stats[player_stats['MIN'].notna() & (player_stats['MIN'] != '0:00')]
            
            # Create simplified lineup data based on minutes played
            lineup_data = []
            
            for team_id in player_stats['TEAM_ID'].unique():
                team_players = player_stats[player_stats['TEAM_ID'] == team_id]
                
                # Get all player combinations (simplified approach)
                players_list = team_players['PLAYER_ID'].tolist()
                player_names = team_players['PLAYER_NAME'].tolist()
                player_minutes = dict(zip(players_list, team_players['MIN']))
                
                # Generate all possible player pairs
                for player1, player2 in combinations(players_list, 2):
                    # Estimate shared minutes (simplified - real data would come from play-by-play)
                    min1 = self._convert_minutes(player_minutes[player1])
                    min2 = self._convert_minutes(player_minutes[player2])
                    
                    # Rough estimate: shared minutes = min(individual minutes) * overlap_factor
                    overlap_factor = 0.7  # Assumption about lineup overlap
                    shared_minutes = min(min1, min2) * overlap_factor
                    
                    lineup_data.append({
                        'game_id': game_id,
                        'team_id': team_id,
                        'player1_id': player1,
                        'player2_id': player2,
                        'player1_name': dict(zip(team_players['PLAYER_ID'], team_players['PLAYER_NAME']))[player1],
                        'player2_name': dict(zip(team_players['PLAYER_ID'], team_players['PLAYER_NAME']))[player2],
                        'shared_minutes': shared_minutes
                    })
            
            return pd.DataFrame(lineup_data)
            
        except Exception as e:
            logger.error(f"Error getting lineups for game {game_id}: {e}")
            return pd.DataFrame()
    
    def _convert_minutes(self, min_str: str) -> float:
        """Convert minutes string (MM:SS) to float."""
        try:
            if pd.isna(min_str) or min_str == '0:00':
                return 0.0
            parts = str(min_str).split(':')
            return float(parts[0]) + float(parts[1]) / 60.0
        except:
            return 0.0
    
    def collect_team_lineup_data(self, team_id: int, season: str, max_games: Optional[int] = None) -> pd.DataFrame:
        """
        Collect lineup data for an entire team's season.
        
        Args:
            team_id: NBA team ID
            season: Season string
            max_games: Maximum number of games to process (for testing)
            
        Returns:
            DataFrame with all lineup combinations and shared minutes
        """
        logger.info(f"Collecting lineup data for team {team_id}, season {season}")
        
        # Get all games for the team
        game_ids = self.get_team_games(team_id, season)
        
        if max_games:
            game_ids = game_ids[:max_games]
        
        logger.info(f"Processing {len(game_ids)} games")
        
        all_lineups = []
        
        for i, game_id in enumerate(game_ids):
            if i % 10 == 0:
                logger.info(f"Processing game {i+1}/{len(game_ids)}")
            
            game_lineups = self.get_game_lineups(game_id)
            if not game_lineups.empty:
                all_lineups.append(game_lineups)
        
        if not all_lineups:
            logger.warning(f"No lineup data collected for team {team_id}")
            return pd.DataFrame()
        
        # Combine all games
        combined_lineups = pd.concat(all_lineups, ignore_index=True)
        
        # Aggregate shared minutes across all games
        aggregated = combined_lineups.groupby([
            'team_id', 'player1_id', 'player2_id', 'player1_name', 'player2_name'
        ])['shared_minutes'].sum().reset_index()
        
        aggregated['season'] = season
        
        logger.info(f"Collected {len(aggregated)} unique player pairs for team {team_id}")
        
        return aggregated
    
    def collect_all_teams_lineup_data(self, season: str, max_games_per_team: Optional[int] = None) -> pd.DataFrame:
        """Collect lineup data for all NBA teams."""
        logger.info(f"Collecting lineup data for all teams, season {season}")
        
        all_team_data = []
        
        # Limit to just a few teams for testing to avoid rate limits
        test_teams = self.teams_info[:6]  # First 6 teams
        
        for team_info in test_teams:
            team_id = team_info['id']
            team_name = team_info['full_name']
            
            logger.info(f"Processing {team_name}")
            
            team_data = self.collect_team_lineup_data(
                team_id=team_id,
                season=season,
                max_games=max_games_per_team or 5  # Limit games for testing
            )
            
            if not team_data.empty:
                team_data['team_name'] = team_name
                all_team_data.append(team_data)
            else:
                logger.warning(f"No data collected for {team_name}")
        
        if not all_team_data:
            logger.error("No lineup data collected for any team")
            return pd.DataFrame()
        
        # Combine all team data
        combined_data = pd.concat(all_team_data, ignore_index=True)
        
        logger.info(f"Collected lineup data for {len(all_team_data)} teams")
        logger.info(f"Total player pairs: {len(combined_data)}")
        
        return combined_data


def collect_lineup_data(season: str, teams_filter: Optional[List[str]] = None, 
                       max_games: Optional[int] = None) -> pd.DataFrame:
    """
    Main function to collect lineup data.
    
    Args:
        season: NBA season (e.g., '2023-24')
        teams_filter: Optional list of team abbreviations to filter
        max_games: Maximum games per team (for testing)
        
    Returns:
        DataFrame with lineup data
    """
    logger.info(f"Starting lineup data collection for season {season}")
    
    collector = LineupDataCollector()
    
    if teams_filter:
        # Filter teams
        teams_info = [t for t in collector.teams_info 
                     if t['abbreviation'] in teams_filter]
        collector.teams_info = teams_info
        logger.info(f"Filtered to {len(teams_info)} teams: {[t['abbreviation'] for t in teams_info]}")
    
    # For testing, limit to max 5 games per team to avoid rate limits
    test_max_games = min(max_games or 5, 5)
    
    return collector.collect_all_teams_lineup_data(season, test_max_games)


if __name__ == "__main__":
    # Test with a single team and limited games
    season = "2023-24"
    
    # Test with Lakers only, 5 games
    lakers_id = 1610612747
    collector = LineupDataCollector()
    
    lineup_data = collector.collect_team_lineup_data(lakers_id, season, max_games=3)
    print(f"Collected lineup data: {len(lineup_data)} player pairs")
    if not lineup_data.empty:
        print(lineup_data.head())
