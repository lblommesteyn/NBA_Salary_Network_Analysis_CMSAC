"""
Roster Network Analysis Module

Builds and analyzes network graphs of NBA team rosters where:
- Nodes = Players (sized by salary, colored by impact metrics)
- Edges = Shared on-court minutes

Computes network topology features for correlating with team performance.
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
import community.community_louvain as community_louvain

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class NetworkFeatures:
    """Container for computed network features."""
    # Salary distribution features
    salary_gini: float
    salary_centralization: float
    salary_assortativity: float
    
    # Network topology features
    density: float
    clustering_coefficient: float
    average_path_length: Optional[float]
    
    # Community structure
    modularity: float
    num_communities: int
    community_sizes: List[int]
    
    # Centrality measures
    degree_centralization: float
    betweenness_centralization: float
    closeness_centralization: float
    
    # Additional metrics
    node_count: int
    edge_count: int
    total_shared_minutes: float


class RosterNetworkAnalyzer:
    """Analyzes roster networks and computes geometric features."""
    
    def __init__(self):
        self.network_cache = {}
    
    def build_roster_network(self, 
                           salary_data: pd.DataFrame,
                           lineup_data: pd.DataFrame,
                           team_id: int,
                           season: str,
                           min_shared_minutes: float = 10.0) -> nx.Graph:
        """
        Build a network graph for a team's roster.
        
        Args:
            salary_data: DataFrame with player salary information
            lineup_data: DataFrame with shared on-court minutes
            team_id: NBA team ID
            season: Season string
            min_shared_minutes: Minimum shared minutes to create an edge
            
        Returns:
            NetworkX graph representing the roster
        """
        # Filter data for this team and season
        team_salary = salary_data[
            (salary_data['team_id'] == team_id) & 
            (salary_data['season'] == season)
        ].copy()
        
        team_lineup = lineup_data[
            (lineup_data['team_id'] == team_id) & 
            (lineup_data['season'] == season) &
            (lineup_data['shared_minutes'] >= min_shared_minutes)
        ].copy()
        
        if team_salary.empty:
            logger.warning(f"No salary data for team {team_id}, season {season}")
            return nx.Graph()
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes (players) with attributes
        for _, player in team_salary.iterrows():
            G.add_node(
                player['player_id'],
                name=player['player_name'],
                salary=player['salary'],
                # Add impact metrics if available
                bpm=player.get('bpm', 0),
                ws_48=player.get('ws_48', 0),
                minutes=player.get('minutes_played', 0)
            )
        
        # Add edges (shared minutes)
        for _, lineup in team_lineup.iterrows():
            p1, p2 = lineup['player1_id'], lineup['player2_id']
            
            if G.has_node(p1) and G.has_node(p2):
                G.add_edge(p1, p2, weight=lineup['shared_minutes'])
        
        logger.info(f"Built network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        return G
    
    def compute_salary_gini(self, G: nx.Graph) -> float:
        """Compute Gini coefficient for salary distribution."""
        salaries = [G.nodes[node]['salary'] for node in G.nodes()]
        
        if not salaries or len(salaries) < 2:
            return 0.0
        
        # Sort salaries
        salaries.sort()
        n = len(salaries)
        
        # Compute Gini coefficient
        cumsum = np.cumsum(salaries)
        gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
        
        return gini
    
    def compute_salary_assortativity(self, G: nx.Graph) -> float:
        """Compute salary assortativity (tendency for similar salaries to connect)."""
        try:
            return nx.numeric_assortativity_coefficient(G, 'salary')
        except:
            return 0.0
    
    def compute_salary_centralization(self, G: nx.Graph) -> float:
        """Compute salary centralization index."""
        salaries = [G.nodes[node]['salary'] for node in G.nodes()]
        
        if not salaries:
            return 0.0
        
        max_salary = max(salaries)
        total_salary = sum(salaries)
        
        if total_salary == 0:
            return 0.0
        
        # Salary centralization: max salary / total salary
        return max_salary / total_salary
    
    def compute_community_structure(self, G: nx.Graph) -> Tuple[float, int, List[int]]:
        """Compute community structure metrics."""
        if G.number_of_edges() == 0:
            return 0.0, 1, [G.number_of_nodes()]
        
        try:
            # Use Louvain community detection
            partition = community_louvain.best_partition(G, weight='weight')
            modularity = community_louvain.modularity(partition, G, weight='weight')
            
            # Count communities and their sizes
            communities = {}
            for node, comm in partition.items():
                communities[comm] = communities.get(comm, 0) + 1
            
            num_communities = len(communities)
            community_sizes = sorted(communities.values(), reverse=True)
            
            return modularity, num_communities, community_sizes
            
        except Exception as e:
            logger.warning(f"Community detection failed: {e}")
            return 0.0, 1, [G.number_of_nodes()]
    
    def compute_centralization_measures(self, G: nx.Graph) -> Tuple[float, float, float]:
        """Compute network centralization measures."""
        if G.number_of_nodes() <= 1:
            return 0.0, 0.0, 0.0
        
        # Degree centralization
        degrees = dict(G.degree(weight='weight'))
        degree_values = list(degrees.values())
        max_degree = max(degree_values) if degree_values else 0
        
        n = G.number_of_nodes()
        max_possible = (n - 1) * (n - 2)  # Max possible centralization
        
        if max_possible == 0:
            degree_centralization = 0.0
        else:
            degree_centralization = sum(max_degree - d for d in degree_values) / max_possible
        
        # Betweenness centralization
        try:
            betweenness = nx.betweenness_centrality(G, weight='weight')
            bet_values = list(betweenness.values())
            max_bet = max(bet_values) if bet_values else 0
            betweenness_centralization = sum(max_bet - b for b in bet_values) / ((n-1)*(n-2)/2)
        except:
            betweenness_centralization = 0.0
        
        # Closeness centralization
        try:
            if nx.is_connected(G):
                closeness = nx.closeness_centrality(G, distance='weight')
                close_values = list(closeness.values())
                max_close = max(close_values) if close_values else 0
                closeness_centralization = sum(max_close - c for c in close_values) / ((n-1)*(n-2)/(2*n-3))
            else:
                closeness_centralization = 0.0
        except:
            closeness_centralization = 0.0
        
        return degree_centralization, betweenness_centralization, closeness_centralization
    
    def analyze_network(self, G: nx.Graph) -> NetworkFeatures:
        """
        Compute all network features for a roster graph.
        
        Args:
            G: NetworkX graph representing team roster
            
        Returns:
            NetworkFeatures object with all computed metrics
        """
        if G.number_of_nodes() == 0:
            return NetworkFeatures(
                salary_gini=0, salary_centralization=0, salary_assortativity=0,
                density=0, clustering_coefficient=0, average_path_length=None,
                modularity=0, num_communities=0, community_sizes=[],
                degree_centralization=0, betweenness_centralization=0, closeness_centralization=0,
                node_count=0, edge_count=0, total_shared_minutes=0
            )
        
        # Salary-based features
        salary_gini = self.compute_salary_gini(G)
        salary_centralization = self.compute_salary_centralization(G)
        salary_assortativity = self.compute_salary_assortativity(G)
        
        # Basic network topology
        density = nx.density(G)
        clustering_coefficient = nx.average_clustering(G, weight='weight')
        
        # Average path length (only for connected graphs)
        try:
            if nx.is_connected(G):
                average_path_length = nx.average_shortest_path_length(G, weight='weight')
            else:
                average_path_length = None
        except:
            average_path_length = None
        
        # Community structure
        modularity, num_communities, community_sizes = self.compute_community_structure(G)
        
        # Centralization measures
        deg_cent, bet_cent, close_cent = self.compute_centralization_measures(G)
        
        # Additional metrics
        node_count = G.number_of_nodes()
        edge_count = G.number_of_edges()
        total_shared_minutes = sum(data['weight'] for _, _, data in G.edges(data=True))
        
        return NetworkFeatures(
            salary_gini=salary_gini,
            salary_centralization=salary_centralization,
            salary_assortativity=salary_assortativity,
            density=density,
            clustering_coefficient=clustering_coefficient,
            average_path_length=average_path_length,
            modularity=modularity,
            num_communities=num_communities,
            community_sizes=community_sizes,
            degree_centralization=deg_cent,
            betweenness_centralization=bet_cent,
            closeness_centralization=close_cent,
            node_count=node_count,
            edge_count=edge_count,
            total_shared_minutes=total_shared_minutes
        )
    
    def analyze_all_teams(self,
                         salary_data: pd.DataFrame,
                         lineup_data: pd.DataFrame,
                         season: str) -> pd.DataFrame:
        """Analyze networks for all teams and return features DataFrame."""
        
        results = []
        
        # Get unique teams from salary data
        teams = salary_data['team_id'].unique()
        
        for team_id in teams:
            logger.info(f"Analyzing network for team {team_id}")
            
            # Build network
            G = self.build_roster_network(salary_data, lineup_data, team_id, season)
            
            # Analyze network
            features = self.analyze_network(G)
            
            # Convert to dictionary and add metadata
            result = {
                'team_id': team_id,
                'season': season,
                'salary_gini': features.salary_gini,
                'salary_centralization': features.salary_centralization,
                'salary_assortativity': features.salary_assortativity,
                'network_density': features.density,
                'clustering_coefficient': features.clustering_coefficient,
                'average_path_length': features.average_path_length,
                'modularity': features.modularity,
                'num_communities': features.num_communities,
                'largest_community_size': max(features.community_sizes) if features.community_sizes else 0,
                'degree_centralization': features.degree_centralization,
                'betweenness_centralization': features.betweenness_centralization,
                'closeness_centralization': features.closeness_centralization,
                'node_count': features.node_count,
                'edge_count': features.edge_count,
                'total_shared_minutes': features.total_shared_minutes
            }
            
            results.append(result)
        
        return pd.DataFrame(results)


def analyze_roster_networks(salary_data: pd.DataFrame,
                           lineup_data: pd.DataFrame,
                           season: str) -> pd.DataFrame:
    """
    Main function to analyze all team networks.
    
    Args:
        salary_data: Player salary data
        lineup_data: Lineup/shared minutes data
        season: Season to analyze
        
    Returns:
        DataFrame with network features for all teams
    """
    analyzer = RosterNetworkAnalyzer()
    return analyzer.analyze_all_teams(salary_data, lineup_data, season)


if __name__ == "__main__":
    # Test network analysis with dummy data
    import random
    
    # Create dummy data for testing
    players = [f"Player_{i}" for i in range(15)]
    
    salary_data = pd.DataFrame({
        'team_id': [1] * 15,
        'season': ['2023-24'] * 15,
        'player_id': list(range(15)),
        'player_name': players,
        'salary': [random.randint(1000000, 40000000) for _ in range(15)]
    })
    
    # Create dummy lineup data
    lineup_pairs = []
    for i in range(15):
        for j in range(i+1, min(i+6, 15)):  # Each player connects to next 5 players
            lineup_pairs.append({
                'team_id': 1,
                'season': '2023-24',
                'player1_id': i,
                'player2_id': j,
                'shared_minutes': random.uniform(50, 500)
            })
    
    lineup_data = pd.DataFrame(lineup_pairs)
    
    # Analyze network
    results = analyze_roster_networks(salary_data, lineup_data, '2023-24')
    print("Network Analysis Results:")
    print(results.round(3))
