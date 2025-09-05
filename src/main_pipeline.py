"""
Main Pipeline for Roster Geometry Analysis

Integrates all components to run the complete analysis pipeline from data collection
through network analysis to playoff correlation analysis.
"""

import pandas as pd
import numpy as np
import os
import sys
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data_collection.salary_data import collect_salary_data
from src.data_collection.lineup_data import collect_lineup_data
from src.network_analysis.roster_networks import analyze_roster_networks
from src.network_analysis.robustness_simulation import run_robustness_analysis
from src.network_analysis.synthetic_roster_generation import SyntheticRosterGenerator, benchmark_against_synthetic, RosterConstraints
from src.analysis.playoff_correlation_analysis import run_playoff_correlation_analysis, print_analysis_summary
from src.visualization.interactive_plots import create_visualizations

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RosterGeometryPipeline:
    """Main pipeline for roster geometry analysis."""
    
    def __init__(self, data_dir: str = "data", results_dir: str = "results"):
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        
        # Create directories if they don't exist
        self.data_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        logger.info(f"Pipeline initialized - Data: {self.data_dir}, Results: {self.results_dir}")
    
    def collect_all_data(self, season: str, max_games_per_team: Optional[int] = None, 
                        teams_filter: Optional[list] = None) -> Dict[str, pd.DataFrame]:
        """
        Collect all required data (salary and lineup).
        
        Args:
            season: NBA season (e.g., '2023-24')
            max_games_per_team: Limit games per team for testing
            teams_filter: Optional list of team abbreviations to filter
            
        Returns:
            Dictionary with salary_data and lineup_data DataFrames
        """
        logger.info(f"Starting data collection for season {season}")
        
        # Collect salary data
        logger.info("Collecting salary data...")
        season_year = int(season.split('-')[1]) + 2000
        salary_data = collect_salary_data(season_year)
        
        if salary_data.empty:
            logger.warning("No salary data collected")
        else:
            logger.info(f"Collected salary data for {len(salary_data)} players")
        
        # Collect lineup data
        logger.info("Collecting lineup data...")
        lineup_data = collect_lineup_data(season, teams_filter, max_games_per_team)
        
        if lineup_data.empty:
            logger.warning("No lineup data collected")
        else:
            logger.info(f"Collected {len(lineup_data)} player pair interactions")
        
        # Save raw data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        salary_file = self.data_dir / f"salary_data_{season}_{timestamp}.csv"
        lineup_file = self.data_dir / f"lineup_data_{season}_{timestamp}.csv"
        
        if not salary_data.empty:
            salary_data.to_csv(salary_file, index=False)
            logger.info(f"Saved salary data to {salary_file}")
        
        if not lineup_data.empty:
            lineup_data.to_csv(lineup_file, index=False)
            logger.info(f"Saved lineup data to {lineup_file}")
        
        return {
            'salary_data': salary_data,
            'lineup_data': lineup_data
        }
    
    def run_network_analysis(self, salary_data: pd.DataFrame, 
                           lineup_data: pd.DataFrame, season: str) -> pd.DataFrame:
        """Run network analysis on collected data."""
        logger.info("Running network analysis...")
        
        # Ensure player_id present in salary_data by merging from lineup names if missing
        if 'player_id' not in salary_data.columns or salary_data['player_id'].isna().all():
            try:
                logger.info("Enriching salary_data with player_id from lineup_data")
                # Build name-id mapping from lineup data for this season
                p1 = lineup_data.loc[lineup_data['season'] == season, ['team_id', 'season', 'player1_id', 'player1_name']]
                p1 = p1.rename(columns={'player1_id': 'player_id', 'player1_name': 'player_name'})
                p2 = lineup_data.loc[lineup_data['season'] == season, ['team_id', 'season', 'player2_id', 'player2_name']]
                p2 = p2.rename(columns={'player2_id': 'player_id', 'player2_name': 'player_name'})
                name_id_map = pd.concat([p1, p2], ignore_index=True).drop_duplicates(subset=['team_id', 'player_id'])

                # Normalize names for join (case/whitespace)
                def norm_name(s):
                    return s.astype(str).str.strip().str.lower()
                if 'player_name' in salary_data.columns:
                    salary_data = salary_data.copy()
                    salary_data['__pnorm'] = norm_name(salary_data['player_name'])
                    name_id_map = name_id_map.copy()
                    name_id_map['__pnorm'] = norm_name(name_id_map['player_name'])
                    # Left join by team_id + normalized name
                    merged = salary_data.merge(
                        name_id_map[['team_id', '__pnorm', 'player_id']],
                        on=['team_id', '__pnorm'], how='left'
                    )
                    # Fill player_id
                    if 'player_id' in merged.columns and ('player_id_x' in merged.columns or 'player_id_y' in merged.columns):
                        pass  # Just in case of name conflict; handled below
                    if 'player_id_y' in merged.columns:
                        merged['player_id'] = merged['player_id_y']
                        merged = merged.drop(columns=[c for c in ['player_id_y'] if c in merged.columns])
                    # Drop helper
                    merged = merged.drop(columns=['__pnorm'])
                    salary_data = merged

                # Generate temporary IDs for unmatched players to keep nodes in graph
                if 'player_id' not in salary_data.columns or salary_data['player_id'].isna().any():
                    missing_mask = salary_data['player_id'].isna() if 'player_id' in salary_data.columns else pd.Series([True]*len(salary_data))
                    if missing_mask.any():
                        logger.warning(f"{missing_mask.sum()} salary rows missing player_id; assigning temporary IDs")
                        # Use negative IDs to avoid collision with real NBA IDs
                        temp_ids = - (pd.Series(range(1, missing_mask.sum()+1)).values)
                        if 'player_id' not in salary_data.columns:
                            salary_data['player_id'] = pd.NA
                        salary_data.loc[missing_mask, 'player_id'] = temp_ids
            except Exception as e:
                logger.warning(f"Failed to enrich salary_data with player_id: {e}")
        
        network_features = analyze_roster_networks(salary_data, lineup_data, season)
        
        if network_features.empty:
            logger.warning("No network features computed")
            return network_features
        
        logger.info(f"Computed network features for {len(network_features)} teams")
        
        # Save network features
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        features_file = self.results_dir / f"network_features_{season}_{timestamp}.csv"
        network_features.to_csv(features_file, index=False)
        logger.info(f"Saved network features to {features_file}")
        # LaTeX table
        features_tex = self.results_dir / f"network_features_table_{season}_{timestamp}.tex"
        try:
            network_features.to_latex(features_tex, index=False, longtable=True)
            logger.info(f"Saved network features LaTeX table to {features_tex}")
        except Exception as e:
            logger.warning(f"Failed to export network features LaTeX table: {e}")
        # Stable _latest copies
        try:
            import shutil
            latest_csv = self.results_dir / f"network_features_{season}_latest.csv"
            latest_tex = self.results_dir / f"network_features_table_{season}_latest.tex"
            shutil.copyfile(features_file, latest_csv)
            if features_tex.exists():
                shutil.copyfile(features_tex, latest_tex)
        except Exception as e:
            logger.warning(f"Failed to write latest copies for network features: {e}")
        
        return network_features
    
    def run_playoff_analysis(self, network_features: pd.DataFrame, season: str) -> Dict:
        """Run playoff correlation analysis."""
        logger.info("Running playoff correlation analysis...")
        
        # Load playoff outcomes data if available
        playoff_file = self.data_dir / f"playoff_outcomes_{season.replace('-', '_')}.csv"
        if playoff_file.exists():
            logger.info(f"Loading playoff outcomes from {playoff_file}")
            playoff_data = pd.read_csv(playoff_file)
            
            # Merge network features with playoff outcomes by team
            # Map team IDs to playoff data (need to match team formats)
            merged_features = self._merge_network_playoff_data(network_features, playoff_data)
            analysis_results = run_playoff_correlation_analysis(merged_features, season)
        else:
            logger.warning(f"No playoff outcomes file found at {playoff_file}")
            analysis_results = run_playoff_correlation_analysis(network_features, season)
        
        # Save analysis results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save correlations
        if not analysis_results.correlations.empty:
            corr_file = self.results_dir / f"correlations_{season}_{timestamp}.csv"
            analysis_results.correlations.to_csv(corr_file, index=False)
            logger.info(f"Saved correlations to {corr_file}")
            # LaTeX
            corr_tex = self.results_dir / f"correlations_table_{season}_{timestamp}.tex"
            try:
                analysis_results.correlations.to_latex(corr_tex, index=False, longtable=True)
                logger.info(f"Saved correlations LaTeX table to {corr_tex}")
            except Exception as e:
                logger.warning(f"Failed to export correlations LaTeX table: {e}")
            # Latest copies
            try:
                import shutil
                shutil.copyfile(corr_file, self.results_dir / f"correlations_{season}_latest.csv")
                if corr_tex.exists():
                    shutil.copyfile(corr_tex, self.results_dir / f"correlations_table_{season}_latest.tex")
            except Exception as e:
                logger.warning(f"Failed to write latest copies for correlations: {e}")
        
        # Save feature importance
        if not analysis_results.feature_importance.empty:
            importance_file = self.results_dir / f"feature_importance_{season}_{timestamp}.csv"
            analysis_results.feature_importance.to_csv(importance_file, index=False)
            logger.info(f"Saved feature importance to {importance_file}")
            # LaTeX
            importance_tex = self.results_dir / f"feature_importance_table_{season}_{timestamp}.tex"
            try:
                analysis_results.feature_importance.to_latex(importance_tex, index=False, longtable=True)
                logger.info(f"Saved feature importance LaTeX table to {importance_tex}")
            except Exception as e:
                logger.warning(f"Failed to export feature importance LaTeX table: {e}")
            # Latest copies
            try:
                import shutil
                shutil.copyfile(importance_file, self.results_dir / f"feature_importance_{season}_latest.csv")
                if importance_tex.exists():
                    shutil.copyfile(importance_tex, self.results_dir / f"feature_importance_table_{season}_latest.tex")
            except Exception as e:
                logger.warning(f"Failed to write latest copies for feature importance: {e}")
        
        return analysis_results
    
    def run_robustness_analysis(self, salary_data: pd.DataFrame, 
                               lineup_data: pd.DataFrame, 
                               network_features: pd.DataFrame, 
                               season: str) -> pd.DataFrame:
        """Run robustness simulation analysis."""
        logger.info("Running roster robustness analysis...")
        
        robustness_results = run_robustness_analysis(
            network_features, salary_data, lineup_data, season
        )
        
        if robustness_results.empty:
            logger.warning("No robustness results computed")
            return robustness_results
        
        logger.info(f"Computed robustness metrics for {len(robustness_results)} teams")
        
        # Save robustness results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        robustness_file = self.results_dir / f"robustness_analysis_{season}_{timestamp}.csv"
        robustness_results.to_csv(robustness_file, index=False)
        logger.info(f"Saved robustness analysis to {robustness_file}")
        # LaTeX table export
        robustness_tex = self.results_dir / f"robustness_analysis_table_{season}_{timestamp}.tex"
        try:
            robustness_results.to_latex(robustness_tex, index=False, longtable=True)
            logger.info(f"Saved robustness analysis LaTeX table to {robustness_tex}")
        except Exception as e:
            logger.warning(f"Failed to export robustness LaTeX table: {e}")
        # Latest copies
        try:
            import shutil
            shutil.copyfile(robustness_file, self.results_dir / f"robustness_analysis_{season}_latest.csv")
            if robustness_tex.exists():
                shutil.copyfile(robustness_tex, self.results_dir / f"robustness_analysis_table_{season}_latest.tex")
        except Exception as e:
            logger.warning(f"Failed to write latest copies for robustness: {e}")
        
        return robustness_results
    
    def create_visualizations(self, network_features: pd.DataFrame) -> Dict:
        """Create interactive visualizations."""
        logger.info("Creating visualizations...")
        
        figures = create_visualizations(network_features)
        
        # Save visualizations
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        season = network_features['season'].iloc[0] if 'season' in network_features.columns and len(network_features) > 0 else 'season'
        combined_results = {}
        for name, fig in figures.items():
            viz_file = self.results_dir / f"visualization_{name}_{season}_{timestamp}.html"
            fig.write_html(str(viz_file))
            logger.info(f"Saved {name} visualization to {viz_file}")
            # Also export PNG via kaleido
            try:
                png_file = self.results_dir / f"visualization_{name}_{season}_{timestamp}.png"
                fig.write_image(str(png_file), format='png', scale=2)
                logger.info(f"Saved {name} visualization PNG to {png_file}")
                # Latest copies
                import shutil
                shutil.copyfile(viz_file, self.results_dir / f"visualization_{name}_{season}_latest.html")
                shutil.copyfile(png_file, self.results_dir / f"visualization_{name}_{season}_latest.png")
            except Exception as e:
                logger.warning(f"Failed to export PNG or copy latest for {name}: {e}")
            combined_results[name] = fig
        
        return combined_results
    
    def _generate_synthetic_salary_data(self, lineup_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate synthetic salary data based on lineup data when real salary data is unavailable.
        """
        # Extract unique players from lineup data
        player1_data = lineup_data[['player1_id', 'player1_name']].rename(
            columns={'player1_id': 'player_id', 'player1_name': 'player_name'}
        )
        player2_data = lineup_data[['player2_id', 'player2_name']].rename(
            columns={'player2_id': 'player_id', 'player2_name': 'player_name'}
        )
        
        unique_players = pd.concat([player1_data, player2_data]).drop_duplicates(subset=['player_id'])
        
        # Generate realistic salary distribution
        np.random.seed(42)  # For reproducibility
        n_players = len(unique_players)
        
        # NBA salary distribution (2023-24 approximate)
        # Stars: $30M+ (5%), Good players: $10-30M (20%), Role players: $2-10M (50%), Min players: <$2M (25%)
        salaries = []
        for i in range(n_players):
            rand = np.random.random()
            if rand < 0.05:  # Stars
                salary = np.random.uniform(30_000_000, 50_000_000)
            elif rand < 0.25:  # Good players
                salary = np.random.uniform(10_000_000, 30_000_000)
            elif rand < 0.75:  # Role players
                salary = np.random.uniform(2_000_000, 10_000_000)
            else:  # Minimum salary players
                salary = np.random.uniform(500_000, 2_000_000)
            salaries.append(salary)
        
        # Create synthetic salary dataframe
        synthetic_salary_data = unique_players.copy()
        synthetic_salary_data['salary'] = salaries
        synthetic_salary_data['season'] = lineup_data['season'].iloc[0] if 'season' in lineup_data.columns else '2023-24'
        synthetic_salary_data['team_id'] = lineup_data['team_id'].iloc[0] if 'team_id' in lineup_data.columns else 0
        
        # Add basic performance metrics (synthetic)
        synthetic_salary_data['bpm'] = np.random.normal(0, 3, n_players)  # Box Plus/Minus
        synthetic_salary_data['ws_per_48'] = np.random.uniform(0, 0.3, n_players)  # Win Shares per 48
        
        return synthetic_salary_data
    
    def run_full_pipeline(self, season: str = '2023-24', 
                         max_games_per_team: Optional[int] = None,
                         teams_filter: Optional[list] = None,
                         create_viz: bool = True,
                         include_robustness: bool = True,
                         include_synthetic: bool = True,
                         num_synthetic_rosters: int = 50) -> Dict:
        """
        Run the complete roster geometry and resilience analysis pipeline.
        
        Args:
            season: NBA season to analyze
            max_games_per_team: Limit games for testing
            teams_filter: Optional team filter for testing
            create_viz: Whether to create visualizations
            include_robustness: Whether to run robustness simulations
            include_synthetic: Whether to run synthetic roster benchmarking
            num_synthetic_rosters: Number of synthetic rosters for benchmarking
            
        Returns:
            Dictionary with all results
        """
        logger.info("="*60)
        logger.info("STARTING ROSTER GEOMETRY AND RESILIENCE PIPELINE")
        logger.info("="*60)
        
        results = {}
        
        try:
            # Step 1: Data collection
            data = self.collect_all_data(season, max_games_per_team, teams_filter)
            results['data'] = data
            
            # Check if we have sufficient data to proceed
            if data['lineup_data'].empty:
                logger.error("No lineup data collected. Aborting pipeline.")
                return results
            
            # If salary data is missing, generate synthetic salary data based on lineup data
            if data['salary_data'].empty:
                logger.warning("No salary data available. Generating synthetic salary data...")
                data['salary_data'] = self._generate_synthetic_salary_data(data['lineup_data'])
                logger.info(f"Generated synthetic salary data for {len(data['salary_data'])} players")
            
            # Step 2: Network analysis
            network_features = self.run_network_analysis(
                data['salary_data'], data['lineup_data'], season
            )
            results['network_features'] = network_features
            
            if network_features.empty:
                logger.error("No network features computed. Aborting pipeline.")
                return results
            
            # Step 3: Robustness simulation analysis (NEW)
            if include_robustness:
                robustness_results = self.run_robustness_analysis(
                    data['salary_data'], data['lineup_data'], network_features, season
                )
                results['robustness'] = robustness_results
                
                if not robustness_results.empty:
                    # Print robustness summary
                    avg_resilience = robustness_results['avg_resilience_score'].mean()
                    logger.info(f"Average team resilience score: {avg_resilience:.3f}")
                    
                    # Step 4: Synthetic roster benchmarking (NEW)
                    if include_synthetic:
                        synthetic_benchmark = self.run_synthetic_benchmarking(
                            robustness_results, num_synthetic=num_synthetic_rosters
                        )
                        results['synthetic_benchmark'] = synthetic_benchmark
            
            # Step 5: Traditional playoff correlation analysis
            analysis_results = self.run_playoff_analysis(network_features, season)
            results['analysis'] = analysis_results
            
            # Step 6: Visualizations (optional)
            if create_viz:
                visualizations = self.create_visualizations(network_features)
                results['visualizations'] = visualizations
            
            # Print comprehensive summary
            self.print_comprehensive_summary(results)
            
            logger.info("="*60)
            logger.info("RESILIENCE PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("="*60)
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {e}")
            raise e
        
        return results
    
    def print_comprehensive_summary(self, results: Dict):
        """Print comprehensive summary of all analysis results."""
        print("="*80)
        print("ROSTER GEOMETRY AND RESILIENCE ANALYSIS SUMMARY")
        print("="*80)
        
        # Network features summary
        if 'network_features' in results and not results['network_features'].empty:
            nf = results['network_features']
            print(f"\n📊 NETWORK ANALYSIS ({len(nf)} teams):")
            print(f"   Average Salary Gini: {nf['salary_gini'].mean():.3f} ± {nf['salary_gini'].std():.3f}")
            print(f"   Average Network Density: {nf['network_density'].mean():.3f} ± {nf['network_density'].std():.3f}")
            print(f"   Average Modularity: {nf['modularity'].mean():.3f} ± {nf['modularity'].std():.3f}")
        
        # Robustness analysis summary
        if 'robustness' in results and not results['robustness'].empty:
            rob = results['robustness']
            print(f"\n🛡️ ROBUSTNESS ANALYSIS ({len(rob)} teams):")
            print(f"   Average Resilience Score: {rob['avg_resilience_score'].mean():.3f} ± {rob['avg_resilience_score'].std():.3f}")
            print(f"   Average Performance Drop: {rob['avg_performance_drop'].mean():.1%} ± {rob['avg_performance_drop'].std():.1%}")
            
            # Find most and least resilient teams
            most_resilient = rob.loc[rob['avg_resilience_score'].idxmax()]
            least_resilient = rob.loc[rob['avg_resilience_score'].idxmin()]
            print(f"   Most Resilient Team: {most_resilient['team_id']} (score: {most_resilient['avg_resilience_score']:.3f})")
            print(f"   Least Resilient Team: {least_resilient['team_id']} (score: {least_resilient['avg_resilience_score']:.3f})")
        
        # Synthetic benchmarking summary
        if 'synthetic_benchmark' in results:
            sb = results['synthetic_benchmark']
            print(f"\n🤖 SYNTHETIC ROSTER BENCHMARKING:")
            print(f"   Historical Mean Resilience: {sb['historical_mean_resilience']:.3f}")
            print(f"   Synthetic Mean Resilience: {sb['synthetic_mean_resilience']:.3f}")
            print(f"   Synthetic Rosters Outperforming Historical: {sb['synthetic_outperform_pct']:.1%}")
            
            if sb['synthetic_mean_resilience'] > sb['historical_mean_resilience']:
                improvement = ((sb['synthetic_mean_resilience'] / sb['historical_mean_resilience']) - 1) * 100
                print(f"   💡 Insight: Optimal roster construction could improve resilience by {improvement:.1f}%")
        
        # Key insights
        print(f"\n🔍 KEY RESEARCH FINDINGS:")
        
        if 'robustness' in results and not results['robustness'].empty and 'network_features' in results:
            rob = results['robustness']
            nf = results['network_features']
            
            # Merge for correlation analysis
            merged = pd.merge(nf, rob, on='team_id', how='inner')
            
            if not merged.empty:
                # Key relationships
                gini_resilience_corr = merged['salary_gini'].corr(merged['avg_resilience_score'])
                density_resilience_corr = merged['network_density'].corr(merged['avg_resilience_score'])
                modularity_resilience_corr = merged['modularity'].corr(merged['avg_resilience_score'])
                
                print(f"   • Salary inequality vs Resilience: r={gini_resilience_corr:.3f}")
                if gini_resilience_corr < -0.2:
                    print(f"     → More balanced salary distributions tend to be MORE resilient")
                elif gini_resilience_corr > 0.2:
                    print(f"     → More unequal salary distributions tend to be MORE resilient")
                
                print(f"   • Network density vs Resilience: r={density_resilience_corr:.3f}")
                if density_resilience_corr > 0.2:
                    print(f"     → Denser interaction networks tend to be MORE resilient")
                
                print(f"   • Community structure vs Resilience: r={modularity_resilience_corr:.3f}")
                if modularity_resilience_corr > 0.2:
                    print(f"     → Teams with stronger community structure tend to be MORE resilient")
                elif modularity_resilience_corr < -0.2:
                    print(f"     → Teams with weaker community structure tend to be MORE resilient")
        
        print(f"\n🎯 CONFERENCE READY: All analyses complete for Carnegie Mellon Sports Analytics Conference!")
        print("="*80)


def create_demo_data(season: str = "2023-24") -> Dict[str, pd.DataFrame]:
    """Create demo data for testing when real data is not available."""
    logger.info("Creating demo data for testing...")
    
    # Create realistic demo data
    np.random.seed(42)
    
    # Demo salary data
    teams = list(range(1, 31))  # 30 NBA teams
    players_per_team = 15
    
    salary_data = []
    player_id = 1
    
    for team_id in teams:
        for i in range(players_per_team):
            # Realistic salary distribution
            if i < 2:  # Stars
                salary = np.random.randint(25000000, 45000000)
            elif i < 5:  # Solid players
                salary = np.random.randint(10000000, 25000000)
            elif i < 10:  # Role players
                salary = np.random.randint(2000000, 10000000)
            else:  # Bench/rookies
                salary = np.random.randint(500000, 3000000)
            
            salary_data.append({
                'team_id': team_id,
                'season': season,
                'player_id': player_id,
                'player_name': f"Player_{player_id}",
                'salary': salary,
                'minutes_played': np.random.randint(200, 2500),
                'bpm': np.random.uniform(-5, 8),
                'ws_48': np.random.uniform(0, 0.3)
            })
            player_id += 1
    
    salary_df = pd.DataFrame(salary_data)
    
    # Demo lineup data
    lineup_data = []
    
    for team_id in teams:
        team_players = salary_df[salary_df['team_id'] == team_id]['player_id'].tolist()
        
        # Create realistic connections (stars play with everyone, bench players have fewer connections)
        for i, p1 in enumerate(team_players):
            connections = min(12, len(team_players) - 1 - i)  # Diminishing connections
            for j in range(i + 1, min(i + 1 + connections, len(team_players))):
                p2 = team_players[j]
                shared_minutes = np.random.uniform(50, 800)  # Realistic shared minutes
                
                lineup_data.append({
                    'team_id': team_id,
                    'season': season,
                    'player1_id': p1,
                    'player2_id': p2,
                    'player1_name': f"Player_{p1}",
                    'player2_name': f"Player_{p2}",
                    'shared_minutes': shared_minutes
                })
    
    lineup_df = pd.DataFrame(lineup_data)
    
    logger.info(f"Created demo data: {len(salary_df)} players, {len(lineup_df)} player pairs")
    
    return {
        'salary_data': salary_df,
        'lineup_data': lineup_df
    }


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Roster Geometry and Resilience Analysis Pipeline')
    parser.add_argument('--season', default='2023-24', help='NBA season (e.g., 2023-24)')
    parser.add_argument('--max-games', type=int, help='Max games per team for testing')
    parser.add_argument('--teams', nargs='+', help='Team abbreviations to filter')
    parser.add_argument('--demo', action='store_true', help='Use demo data instead of real data')
    parser.add_argument('--no-viz', action='store_true', help='Skip visualization creation')
    
    # Resilience analysis arguments
    parser.add_argument('--include-robustness', action='store_true', 
                       help='Include robustness simulation analysis')
    parser.add_argument('--include-synthetic', action='store_true', 
                       help='Include synthetic roster benchmarking')
    parser.add_argument('--num-synthetic', type=int, default=50,
                       help='Number of synthetic rosters to generate for benchmarking')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = RosterGeometryPipeline()
    
    if args.demo:
        # Use demo data
        logger.info("Running with demo data...")
        demo_data = create_demo_data(args.season)
        
        # Run analysis components with resilience options
        network_features = pipeline.run_network_analysis(
            demo_data['salary_data'], demo_data['lineup_data'], args.season
        )
        
        if not network_features.empty:
            results = {}
            results['network_features'] = network_features
            results['data'] = demo_data
            
            # Traditional playoff analysis
            analysis_results = pipeline.run_playoff_analysis(network_features, args.season)
            results['analysis'] = analysis_results
            
            # Resilience analysis if requested
            if args.include_robustness:
                logger.info("Running robustness analysis...")
                robustness_results = pipeline.run_robustness_analysis(
                    demo_data['salary_data'], demo_data['lineup_data'], network_features, args.season
                )
                results['robustness'] = robustness_results
                
                # Synthetic benchmarking if requested
                if args.include_synthetic and not robustness_results.empty:
                    logger.info("Running synthetic roster benchmarking...")
                    synthetic_benchmark = pipeline.run_synthetic_benchmarking(
                        robustness_results, num_synthetic=args.num_synthetic
                    )
                    results['synthetic_benchmark'] = synthetic_benchmark
            
            # Visualizations
            if not args.no_viz:
                visualizations = pipeline.create_visualizations(network_features)
                results['visualizations'] = visualizations
                logger.info(f"Created {len(visualizations)} visualizations")
            
            # Print comprehensive summary
            if args.include_robustness:
                pipeline.print_comprehensive_summary(results)
            else:
                print_analysis_summary(analysis_results)
    else:
        # Run full pipeline with real data
        results = pipeline.run_full_pipeline(
            season=args.season,
            max_games_per_team=args.max_games,
            teams_filter=args.teams,
            create_viz=not args.no_viz
        )


if __name__ == "__main__":
    main()
