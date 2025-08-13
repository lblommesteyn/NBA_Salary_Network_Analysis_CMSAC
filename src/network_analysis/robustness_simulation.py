"""
Robustness Simulation Module

Simulates team resilience to player disruptions (injuries, trades) by systematically
removing nodes from roster networks and measuring performance degradation.
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from itertools import combinations
import random

from .roster_networks import RosterNetworkAnalyzer, NetworkFeatures

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DisruptionScenario:
    """Container for disruption scenario parameters."""
    scenario_name: str
    players_removed: List[int]  # Player IDs
    removal_reason: str  # "injury", "trade", "suspension", etc.
    duration_games: Optional[int] = None  # How many games affected


@dataclass
class RobustnessResults:
    """Container for robustness analysis results."""
    baseline_features: NetworkFeatures
    disruption_scenarios: List[DisruptionScenario]
    post_disruption_features: List[NetworkFeatures]
    resilience_scores: List[float]
    performance_drops: List[float]
    critical_players: List[int]  # Players whose removal causes largest drops


class RobustnessSimulator:
    """Simulates and analyzes roster network robustness to disruptions."""
    
    def __init__(self):
        self.network_analyzer = RosterNetworkAnalyzer()
        self.baseline_networks = {}
        self.disruption_cache = {}
    
    def create_disruption_scenarios(self, G: nx.Graph, 
                                  scenario_types: List[str] = None) -> List[DisruptionScenario]:
        """
        Generate various disruption scenarios for testing robustness.
        
        Args:
            G: Original network graph
            scenario_types: Types of scenarios to generate
            
        Returns:
            List of disruption scenarios
        """
        if scenario_types is None:
            scenario_types = ["single_star", "single_role", "double_removal", "community_loss"]
        
        scenarios = []
        
        if G.number_of_nodes() == 0:
            return scenarios
        
        # Get node attributes for scenario generation
        salaries = {node: G.nodes[node].get('salary', 0) for node in G.nodes()}
        degrees = dict(G.degree(weight='weight'))
        betweenness = nx.betweenness_centrality(G, weight='weight')
        
        # Sort nodes by different criteria
        nodes_by_salary = sorted(salaries.keys(), key=lambda x: salaries[x], reverse=True)
        nodes_by_degree = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)
        nodes_by_betweenness = sorted(betweenness.keys(), key=lambda x: betweenness[x], reverse=True)
        
        # Single star player removal (highest salary)
        if "single_star" in scenario_types and nodes_by_salary:
            scenarios.append(DisruptionScenario(
                scenario_name="single_star_injury",
                players_removed=[nodes_by_salary[0]],
                removal_reason="injury",
                duration_games=20
            ))
        
        # Single role player removal (median salary)
        if "single_role" in scenario_types and len(nodes_by_salary) >= 3:
            mid_idx = len(nodes_by_salary) // 2
            scenarios.append(DisruptionScenario(
                scenario_name="single_role_injury",
                players_removed=[nodes_by_salary[mid_idx]],
                removal_reason="injury",
                duration_games=15
            ))
        
        # Double star removal
        if "double_removal" in scenario_types and len(nodes_by_salary) >= 2:
            scenarios.append(DisruptionScenario(
                scenario_name="double_star_injury",
                players_removed=nodes_by_salary[:2],
                removal_reason="injury",
                duration_games=10
            ))
        
        # High centrality player removal
        if "centrality_loss" in scenario_types and nodes_by_betweenness:
            scenarios.append(DisruptionScenario(
                scenario_name="key_connector_loss",
                players_removed=[nodes_by_betweenness[0]],
                removal_reason="trade",
                duration_games=None  # Permanent
            ))
        
        # Community disruption (remove multiple players from same community)
        if "community_loss" in scenario_types and G.number_of_nodes() > 5:
            try:
                import community.community_louvain as community_louvain
                partition = community_louvain.best_partition(G, weight='weight')
                
                # Find largest community
                community_sizes = {}
                for node, comm in partition.items():
                    community_sizes[comm] = community_sizes.get(comm, 0) + 1
                
                if community_sizes:
                    largest_comm = max(community_sizes.keys(), key=lambda x: community_sizes[x])
                    comm_players = [node for node, comm in partition.items() if comm == largest_comm]
                    
                    if len(comm_players) >= 2:
                        # Remove top 2 salary players from largest community
                        comm_by_salary = sorted(comm_players, 
                                              key=lambda x: salaries.get(x, 0), reverse=True)
                        scenarios.append(DisruptionScenario(
                            scenario_name="community_disruption",
                            players_removed=comm_by_salary[:2],
                            removal_reason="injury",
                            duration_games=25
                        ))
            except ImportError:
                logger.warning("Community detection not available for community_loss scenarios")
        
        # Random removal scenarios
        if "random" in scenario_types and G.number_of_nodes() >= 3:
            random_nodes = random.sample(list(G.nodes()), min(2, G.number_of_nodes()))
            scenarios.append(DisruptionScenario(
                scenario_name="random_disruption",
                players_removed=random_nodes,
                removal_reason="injury",
                duration_games=15
            ))
        
        logger.info(f"Generated {len(scenarios)} disruption scenarios")
        return scenarios
    
    def apply_disruption(self, G: nx.Graph, scenario: DisruptionScenario) -> nx.Graph:
        """
        Apply a disruption scenario to a network.
        
        Args:
            G: Original network
            scenario: Disruption to apply
            
        Returns:
            Modified network with players removed
        """
        G_disrupted = G.copy()
        
        # Remove players
        for player_id in scenario.players_removed:
            if G_disrupted.has_node(player_id):
                G_disrupted.remove_node(player_id)
        
        return G_disrupted
    
    def compute_resilience_score(self, baseline_features: NetworkFeatures,
                               disrupted_features: NetworkFeatures,
                               weights: Dict[str, float] = None) -> float:
        """
        Compute resilience score comparing baseline to disrupted network.
        
        Args:
            baseline_features: Features from original network
            disrupted_features: Features from disrupted network  
            weights: Weights for different feature categories
            
        Returns:
            Resilience score (higher = more resilient)
        """
        if weights is None:
            weights = {
                'connectivity': 0.3,  # Density, clustering
                'efficiency': 0.3,    # Path length, centralization
                'structure': 0.2,     # Modularity, communities
                'size': 0.2          # Node/edge count preservation
            }
        
        # Connectivity preservation
        density_preservation = (disrupted_features.density / max(baseline_features.density, 0.001))
        clustering_preservation = (disrupted_features.clustering_coefficient / 
                                 max(baseline_features.clustering_coefficient, 0.001))
        connectivity_score = (density_preservation + clustering_preservation) / 2
        
        # Efficiency preservation  
        # For path length, smaller disruption is better (resilient networks maintain short paths)
        if baseline_features.average_path_length and disrupted_features.average_path_length:
            path_preservation = min(1.0, baseline_features.average_path_length / 
                                  disrupted_features.average_path_length)
        else:
            path_preservation = 0.5  # Neutral if path length unavailable
        
        # Centralization should remain stable
        cent_preservation = 1.0 - abs(disrupted_features.degree_centralization - 
                                    baseline_features.degree_centralization)
        efficiency_score = (path_preservation + cent_preservation) / 2
        
        # Structural preservation
        modularity_preservation = (disrupted_features.modularity / 
                                 max(baseline_features.modularity, 0.001))
        structure_score = min(1.0, modularity_preservation)
        
        # Size preservation (how much of the network remains)
        node_preservation = disrupted_features.node_count / max(baseline_features.node_count, 1)
        edge_preservation = disrupted_features.edge_count / max(baseline_features.edge_count, 1)
        size_score = (node_preservation + edge_preservation) / 2
        
        # Weighted combination
        resilience_score = (
            weights['connectivity'] * connectivity_score +
            weights['efficiency'] * efficiency_score +
            weights['structure'] * structure_score +
            weights['size'] * size_score
        )
        
        return max(0.0, min(1.0, resilience_score))  # Clamp to [0,1]
    
    def estimate_performance_impact(self, baseline_features: NetworkFeatures,
                                  disrupted_features: NetworkFeatures,
                                  salary_data: pd.DataFrame) -> float:
        """
        Estimate performance impact based on network degradation.
        
        This is a simplified model - in practice, you'd train this on historical data.
        """
        # Simple heuristic: performance drop proportional to network degradation
        resilience = self.compute_resilience_score(baseline_features, disrupted_features)
        
        # Get salary impact (what % of salary was lost)
        total_salary = salary_data['salary'].sum() if not salary_data.empty else 1
        lost_salary = 0
        
        # This is simplified - in practice you'd look up actual removed players
        salary_drop_factor = 0.1  # Placeholder
        
        # Combine network and salary impacts
        network_impact = (1 - resilience) * 0.7  # Network structure impact
        salary_impact = salary_drop_factor * 0.3  # Direct talent impact
        
        total_performance_drop = network_impact + salary_impact
        
        return min(1.0, total_performance_drop)
    
    def run_robustness_analysis(self, G: nx.Graph, 
                               salary_data: pd.DataFrame,
                               team_id: int,
                               scenario_types: List[str] = None) -> RobustnessResults:
        """
        Run complete robustness analysis for a team.
        
        Args:
            G: Team network graph
            salary_data: Salary information for players
            team_id: Team identifier
            scenario_types: Types of disruption scenarios to test
            
        Returns:
            RobustnessResults with complete analysis
        """
        logger.info(f"Running robustness analysis for team {team_id}")
        
        # Compute baseline features
        baseline_features = self.network_analyzer.analyze_network(G)
        
        # Generate disruption scenarios
        scenarios = self.create_disruption_scenarios(G, scenario_types)
        
        post_disruption_features = []
        resilience_scores = []
        performance_drops = []
        
        # Test each scenario
        for scenario in scenarios:
            logger.info(f"Testing scenario: {scenario.scenario_name}")
            
            # Apply disruption
            G_disrupted = self.apply_disruption(G, scenario)
            
            # Analyze disrupted network
            disrupted_features = self.network_analyzer.analyze_network(G_disrupted)
            post_disruption_features.append(disrupted_features)
            
            # Compute resilience score
            resilience = self.compute_resilience_score(baseline_features, disrupted_features)
            resilience_scores.append(resilience)
            
            # Estimate performance impact
            perf_drop = self.estimate_performance_impact(
                baseline_features, disrupted_features, salary_data
            )
            performance_drops.append(perf_drop)
        
        # Identify critical players (those whose removal causes largest drops)
        critical_players = []
        min_resilience_idx = np.argmin(resilience_scores) if resilience_scores else 0
        if scenarios and min_resilience_idx < len(scenarios):
            critical_players = scenarios[min_resilience_idx].players_removed
        
        results = RobustnessResults(
            baseline_features=baseline_features,
            disruption_scenarios=scenarios,
            post_disruption_features=post_disruption_features,
            resilience_scores=resilience_scores,
            performance_drops=performance_drops,
            critical_players=critical_players
        )
        
        logger.info(f"Robustness analysis complete. Average resilience: {np.mean(resilience_scores):.3f}")
        
        return results
    
    def compare_robustness_across_teams(self, 
                                       network_features_df: pd.DataFrame,
                                       salary_data: pd.DataFrame,
                                       lineup_data: pd.DataFrame,
                                       season: str) -> pd.DataFrame:
        """
        Compare robustness across all teams in the dataset.
        
        Returns:
            DataFrame with robustness metrics for each team
        """
        logger.info("Running robustness comparison across all teams")
        
        robustness_results = []
        
        for _, row in network_features_df.iterrows():
            team_id = row['team_id']
            
            # Build network for this team
            team_salary = salary_data[salary_data['team_id'] == team_id]
            team_lineup = lineup_data[lineup_data['team_id'] == team_id]
            
            G = self.network_analyzer.build_roster_network(
                team_salary, team_lineup, team_id, season
            )
            
            if G.number_of_nodes() == 0:
                continue
            
            # Run robustness analysis
            robustness = self.run_robustness_analysis(G, team_salary, team_id)
            
            # Aggregate results
            result = {
                'team_id': team_id,
                'season': season,
                'avg_resilience_score': np.mean(robustness.resilience_scores),
                'min_resilience_score': np.min(robustness.resilience_scores),
                'max_performance_drop': np.max(robustness.performance_drops),
                'avg_performance_drop': np.mean(robustness.performance_drops),
                'num_critical_players': len(robustness.critical_players),
                'star_injury_resilience': None,
                'community_disruption_resilience': None
            }
            
            # Extract specific scenario results
            for i, scenario in enumerate(robustness.disruption_scenarios):
                if scenario.scenario_name == "single_star_injury" and i < len(robustness.resilience_scores):
                    result['star_injury_resilience'] = robustness.resilience_scores[i]
                elif scenario.scenario_name == "community_disruption" and i < len(robustness.resilience_scores):
                    result['community_disruption_resilience'] = robustness.resilience_scores[i]
            
            robustness_results.append(result)
        
        return pd.DataFrame(robustness_results)


def run_robustness_analysis(network_features_df: pd.DataFrame,
                           salary_data: pd.DataFrame,
                           lineup_data: pd.DataFrame,
                           season: str) -> pd.DataFrame:
    """
    Main function to run robustness analysis across all teams.
    
    Args:
        network_features_df: Network features for all teams
        salary_data: Salary data
        lineup_data: Lineup interaction data  
        season: Season to analyze
        
    Returns:
        DataFrame with robustness results
    """
    simulator = RobustnessSimulator()
    return simulator.compare_robustness_across_teams(
        network_features_df, salary_data, lineup_data, season
    )


if __name__ == "__main__":
    # Test robustness simulation with dummy data
    import sys
    sys.path.append('../../')
    from src.main_pipeline import create_demo_data
    
    # Create test data
    demo_data = create_demo_data("2023-24")
    
    # Test single team robustness
    simulator = RobustnessSimulator()
    analyzer = RosterNetworkAnalyzer()
    
    team_id = 1
    team_salary = demo_data['salary_data'][demo_data['salary_data']['team_id'] == team_id]
    team_lineup = demo_data['lineup_data'][demo_data['lineup_data']['team_id'] == team_id]
    
    G = analyzer.build_roster_network(team_salary, team_lineup, team_id, "2023-24")
    
    if G.number_of_nodes() > 0:
        results = simulator.run_robustness_analysis(G, team_salary, team_id)
        
        print(f"Robustness Analysis Results for Team {team_id}:")
        print(f"Baseline nodes: {results.baseline_features.node_count}")
        print(f"Baseline edges: {results.baseline_features.edge_count}")
        print(f"Average resilience: {np.mean(results.resilience_scores):.3f}")
        print(f"Critical players: {results.critical_players}")
    else:
        print("No network data available for testing")
