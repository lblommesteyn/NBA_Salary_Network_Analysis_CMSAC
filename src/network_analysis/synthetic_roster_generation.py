"""
Synthetic Roster Generation Module

Generates synthetic NBA rosters under salary cap and positional constraints
for benchmarking against historical rosters in terms of network resilience.
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from itertools import combinations
import random
from scipy.optimize import minimize
from scipy.stats import norm

from .roster_networks import RosterNetworkAnalyzer
from .robustness_simulation import RobustnessSimulator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RosterConstraints:
    """Container for roster construction constraints."""
    salary_cap: float
    min_players: int = 13
    max_players: int = 17
    max_salary_share: float = 0.35  # Max % of cap for one player
    min_salary: float = 500_000     # Rookie minimum
    positions_required: Dict[str, int] = None  # e.g., {'PG': 2, 'SG': 2, 'SF': 2, 'PF': 2, 'C': 2}


@dataclass
class SyntheticPlayer:
    """Container for synthetic player attributes."""
    player_id: int
    salary: float
    position: str
    skill_level: float  # 0-1 normalized skill
    chemistry_type: str  # "star", "glue", "role", "bench"
    minutes_expectation: float  # Expected minutes per game


class SyntheticRosterGenerator:
    """Generates synthetic rosters for benchmarking roster geometry."""
    
    def __init__(self, season: str = "2023-24"):
        self.season = season
        self.historical_distributions = {}
        self.position_templates = {
            'PG': {'min_salary': 1_000_000, 'max_salary': 50_000_000, 'minutes': 25},
            'SG': {'min_salary': 1_000_000, 'max_salary': 45_000_000, 'minutes': 24},
            'SF': {'min_salary': 1_000_000, 'max_salary': 48_000_000, 'minutes': 28},
            'PF': {'min_salary': 1_000_000, 'max_salary': 40_000_000, 'minutes': 26},
            'C': {'min_salary': 1_000_000, 'max_salary': 35_000_000, 'minutes': 22}
        }
    
    def learn_historical_distributions(self, salary_data: pd.DataFrame, 
                                     lineup_data: pd.DataFrame) -> Dict:
        """
        Learn salary and interaction patterns from historical data.
        
        Args:
            salary_data: Historical salary information
            lineup_data: Historical lineup interaction data
            
        Returns:
            Dictionary with learned distributions
        """
        logger.info("Learning historical salary and interaction patterns")
        
        distributions = {
            'salary_by_percentile': {},
            'interaction_patterns': {},
            'roster_sizes': [],
            'salary_structures': []
        }
        
        # Salary distributions by position and skill level
        if 'position' in salary_data.columns:
            for pos in salary_data['position'].unique():
                pos_salaries = salary_data[salary_data['position'] == pos]['salary']
                distributions['salary_by_percentile'][pos] = {
                    'p10': pos_salaries.quantile(0.1),
                    'p25': pos_salaries.quantile(0.25),
                    'p50': pos_salaries.quantile(0.5),
                    'p75': pos_salaries.quantile(0.75),
                    'p90': pos_salaries.quantile(0.9),
                    'mean': pos_salaries.mean(),
                    'std': pos_salaries.std()
                }
        
        # Team-level patterns
        for team_id in salary_data['team_id'].unique():
            team_salaries = salary_data[salary_data['team_id'] == team_id]['salary']
            if len(team_salaries) > 0:
                distributions['roster_sizes'].append(len(team_salaries))
                
                # Salary structure (top player %, top 3 players %, Gini, etc.)
                sorted_salaries = sorted(team_salaries, reverse=True)
                total_salary = sum(sorted_salaries)
                
                if total_salary > 0:
                    structure = {
                        'total_payroll': total_salary,
                        'top_player_share': sorted_salaries[0] / total_salary,
                        'top_3_share': sum(sorted_salaries[:3]) / total_salary if len(sorted_salaries) >= 3 else 1.0,
                        'salary_gini': self._compute_gini(sorted_salaries)
                    }
                    distributions['salary_structures'].append(structure)
        
        # Interaction patterns (simplified)
        if not lineup_data.empty:
            avg_shared_minutes = lineup_data.groupby('team_id')['shared_minutes'].mean()
            distributions['interaction_patterns'] = {
                'mean_interaction': avg_shared_minutes.mean(),
                'std_interaction': avg_shared_minutes.std(),
                'typical_connections': lineup_data.groupby('team_id').size().mean()
            }
        
        self.historical_distributions = distributions
        logger.info("Historical distribution learning complete")
        
        return distributions
    
    def _compute_gini(self, values: List[float]) -> float:
        """Compute Gini coefficient for salary distribution."""
        if not values or len(values) < 2:
            return 0.0
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        cumsum = np.cumsum(sorted_values)
        
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
    
    def generate_salary_distribution(self, constraints: RosterConstraints,
                                   distribution_type: str = "balanced") -> List[float]:
        """
        Generate salary distribution for a roster.
        
        Args:
            constraints: Roster construction constraints
            distribution_type: Type of distribution ("balanced", "top_heavy", "star_duo", etc.)
            
        Returns:
            List of salaries for roster
        """
        target_players = random.randint(constraints.min_players, constraints.max_players)
        
        if distribution_type == "balanced":
            # More even salary distribution
            salaries = self._generate_balanced_salaries(constraints, target_players)
        elif distribution_type == "top_heavy":
            # One superstar, rest distributed normally
            salaries = self._generate_top_heavy_salaries(constraints, target_players)
        elif distribution_type == "star_duo":
            # Two stars, rest role players
            salaries = self._generate_star_duo_salaries(constraints, target_players)
        elif distribution_type == "deep":
            # No superstars, many solid players
            salaries = self._generate_deep_salaries(constraints, target_players)
        else:
            # Default to balanced
            salaries = self._generate_balanced_salaries(constraints, target_players)
        
        # Ensure constraints are met
        salaries = self._adjust_for_constraints(salaries, constraints)
        
        return salaries
    
    def _generate_balanced_salaries(self, constraints: RosterConstraints, num_players: int) -> List[float]:
        """Generate balanced salary distribution."""
        # Target Gini around 0.3-0.4 (relatively balanced)
        
        # Start with base salaries
        min_sal = constraints.min_salary
        available_cap = constraints.salary_cap - (num_players * min_sal)
        
        if available_cap <= 0:
            return [min_sal] * num_players
        
        # Use log-normal distribution for more realistic salary spread
        mu = np.log(available_cap / num_players / 3)  # Mean of log-normal
        sigma = 0.8  # Standard deviation (controls inequality)
        
        extra_salaries = np.random.lognormal(mu, sigma, num_players)
        
        # Scale to fit available cap
        scale_factor = available_cap / sum(extra_salaries)
        extra_salaries *= scale_factor
        
        final_salaries = [min_sal + extra for extra in extra_salaries]
        
        return final_salaries
    
    def _generate_top_heavy_salaries(self, constraints: RosterConstraints, num_players: int) -> List[float]:
        """Generate top-heavy salary distribution (one superstar)."""
        min_sal = constraints.min_salary
        max_star_salary = constraints.salary_cap * constraints.max_salary_share
        
        # Superstar salary
        star_salary = random.uniform(max_star_salary * 0.8, max_star_salary)
        
        # Remaining budget for other players
        remaining_budget = constraints.salary_cap - star_salary - ((num_players - 1) * min_sal)
        
        if remaining_budget <= 0:
            # Adjust star salary if needed
            star_salary = constraints.salary_cap - ((num_players - 1) * min_sal * 1.5)
            remaining_budget = constraints.salary_cap - star_salary - ((num_players - 1) * min_sal)
        
        # Distribute remaining budget among other players (more uneven)
        if remaining_budget > 0:
            weights = np.random.exponential(1, num_players - 1)  # Exponential for high inequality
            weights = weights / sum(weights)
            extra_salaries = weights * remaining_budget
            
            other_salaries = [min_sal + extra for extra in extra_salaries]
        else:
            other_salaries = [min_sal] * (num_players - 1)
        
        return [star_salary] + other_salaries
    
    def _generate_star_duo_salaries(self, constraints: RosterConstraints, num_players: int) -> List[float]:
        """Generate star duo salary distribution."""
        min_sal = constraints.min_salary
        max_combined_stars = constraints.salary_cap * 0.6  # Two stars take 60% of cap
        
        # Two star salaries
        star1_salary = random.uniform(max_combined_stars * 0.45, max_combined_stars * 0.6)
        star2_salary = random.uniform(max_combined_stars * 0.3, max_combined_stars * 0.55)
        
        # Ensure they don't exceed combined limit
        if star1_salary + star2_salary > max_combined_stars:
            scale = max_combined_stars / (star1_salary + star2_salary)
            star1_salary *= scale
            star2_salary *= scale
        
        # Remaining budget
        remaining_budget = constraints.salary_cap - star1_salary - star2_salary - ((num_players - 2) * min_sal)
        
        if remaining_budget > 0:
            # More even distribution among role players
            other_salaries = [min_sal + (remaining_budget / (num_players - 2))] * (num_players - 2)
        else:
            other_salaries = [min_sal] * (num_players - 2)
        
        return [star1_salary, star2_salary] + other_salaries
    
    def _generate_deep_salaries(self, constraints: RosterConstraints, num_players: int) -> List[float]:
        """Generate deep roster salary distribution (no superstars)."""
        min_sal = constraints.min_salary
        max_player_salary = constraints.salary_cap * 0.2  # No player gets more than 20%
        
        available_cap = constraints.salary_cap - (num_players * min_sal)
        
        # Use more uniform distribution
        extra_salaries = []
        for _ in range(num_players):
            max_extra = min(max_player_salary - min_sal, available_cap * 0.3)
            extra = random.uniform(0, max_extra)
            extra_salaries.append(extra)
        
        # Scale to fit budget
        if sum(extra_salaries) > available_cap:
            scale = available_cap / sum(extra_salaries)
            extra_salaries = [extra * scale for extra in extra_salaries]
        
        final_salaries = [min_sal + extra for extra in extra_salaries]
        
        return final_salaries
    
    def _adjust_for_constraints(self, salaries: List[float], constraints: RosterConstraints) -> List[float]:
        """Adjust salaries to meet constraints."""
        # Ensure total doesn't exceed cap
        total = sum(salaries)
        if total > constraints.salary_cap:
            scale = constraints.salary_cap / total
            salaries = [sal * scale for sal in salaries]
        
        # Ensure no player exceeds max share
        max_allowed = constraints.salary_cap * constraints.max_salary_share
        for i in range(len(salaries)):
            if salaries[i] > max_allowed:
                excess = salaries[i] - max_allowed
                salaries[i] = max_allowed
                # Redistribute excess to other players
                if len(salaries) > 1:
                    per_player_bonus = excess / (len(salaries) - 1)
                    for j in range(len(salaries)):
                        if j != i:
                            salaries[j] += per_player_bonus
        
        # Ensure minimums
        for i in range(len(salaries)):
            salaries[i] = max(salaries[i], constraints.min_salary)
        
        return salaries
    
    def generate_interaction_network(self, players: List[SyntheticPlayer],
                                   interaction_style: str = "realistic") -> nx.Graph:
        """
        Generate interaction network for synthetic roster.
        
        Args:
            players: List of synthetic players
            interaction_style: Style of interactions ("realistic", "dense", "sparse", "hierarchical")
            
        Returns:
            NetworkX graph with synthetic interactions
        """
        G = nx.Graph()
        
        # Add nodes
        for player in players:
            G.add_node(
                player.player_id,
                name=f"Player_{player.player_id}",
                salary=player.salary,
                position=player.position,
                skill_level=player.skill_level,
                chemistry_type=player.chemistry_type,
                minutes=player.minutes_expectation
            )
        
        # Generate edges based on interaction style
        if interaction_style == "realistic":
            self._add_realistic_interactions(G, players)
        elif interaction_style == "dense":
            self._add_dense_interactions(G, players)
        elif interaction_style == "sparse":
            self._add_sparse_interactions(G, players)
        elif interaction_style == "hierarchical":
            self._add_hierarchical_interactions(G, players)
        else:
            self._add_realistic_interactions(G, players)
        
        return G
    
    def _add_realistic_interactions(self, G: nx.Graph, players: List[SyntheticPlayer]):
        """Add realistic interaction patterns based on positions and roles."""
        for i, p1 in enumerate(players):
            for j, p2 in enumerate(players[i+1:], i+1):
                # Base interaction probability
                base_prob = 0.4
                
                # Adjust based on roles
                if p1.chemistry_type == "star" or p2.chemistry_type == "star":
                    base_prob += 0.3  # Stars play with many players
                
                if p1.chemistry_type == "glue" or p2.chemistry_type == "glue":
                    base_prob += 0.2  # Glue players connect others
                
                # Adjust based on positions (guards more likely to interact with forwards, etc.)
                pos_compatibility = self._get_position_compatibility(p1.position, p2.position)
                base_prob *= pos_compatibility
                
                # Skill level similarity (better players play together more)
                skill_diff = abs(p1.skill_level - p2.skill_level)
                skill_bonus = max(0, 1 - skill_diff) * 0.3
                base_prob += skill_bonus
                
                if random.random() < base_prob:
                    # Calculate shared minutes based on individual minutes and interaction strength
                    avg_minutes = (p1.minutes_expectation + p2.minutes_expectation) / 2
                    interaction_factor = random.uniform(0.3, 0.8)  # How much they overlap
                    shared_minutes = avg_minutes * interaction_factor * 30  # ~30 games worth
                    
                    G.add_edge(p1.player_id, p2.player_id, weight=shared_minutes)
    
    def _get_position_compatibility(self, pos1: str, pos2: str) -> float:
        """Get position compatibility factor for interactions."""
        # Position compatibility matrix (simplified)
        compatibility = {
            ('PG', 'SG'): 1.2, ('PG', 'SF'): 1.0, ('PG', 'PF'): 0.8, ('PG', 'C'): 0.7,
            ('SG', 'SF'): 1.1, ('SG', 'PF'): 0.9, ('SG', 'C'): 0.8,
            ('SF', 'PF'): 1.2, ('SF', 'C'): 1.0,
            ('PF', 'C'): 1.1
        }
        
        key = tuple(sorted([pos1, pos2]))
        return compatibility.get(key, 1.0)
    
    def _add_dense_interactions(self, G: nx.Graph, players: List[SyntheticPlayer]):
        """Add dense interaction pattern (most players connect)."""
        for i, p1 in enumerate(players):
            for j, p2 in enumerate(players[i+1:], i+1):
                if random.random() < 0.8:  # High connection probability
                    shared_minutes = random.uniform(100, 600)
                    G.add_edge(p1.player_id, p2.player_id, weight=shared_minutes)
    
    def _add_sparse_interactions(self, G: nx.Graph, players: List[SyntheticPlayer]):
        """Add sparse interaction pattern (fewer connections)."""
        for i, p1 in enumerate(players):
            for j, p2 in enumerate(players[i+1:], i+1):
                if random.random() < 0.3:  # Low connection probability
                    shared_minutes = random.uniform(50, 300)
                    G.add_edge(p1.player_id, p2.player_id, weight=shared_minutes)
    
    def _add_hierarchical_interactions(self, G: nx.Graph, players: List[SyntheticPlayer]):
        """Add hierarchical interaction pattern (stars connect to many, role players to few)."""
        # Sort by skill level
        sorted_players = sorted(players, key=lambda x: x.skill_level, reverse=True)
        
        for i, p1 in enumerate(sorted_players):
            # Number of connections depends on skill level
            max_connections = int(p1.skill_level * len(players))
            actual_connections = 0
            
            for j, p2 in enumerate(sorted_players[i+1:], i+1):
                if actual_connections >= max_connections:
                    break
                
                # Higher probability to connect to players of similar skill
                skill_similarity = 1 - abs(p1.skill_level - p2.skill_level)
                if random.random() < skill_similarity * 0.7:
                    shared_minutes = random.uniform(100, 500)
                    G.add_edge(p1.player_id, p2.player_id, weight=shared_minutes)
                    actual_connections += 1
    
    def create_synthetic_roster(self, constraints: RosterConstraints,
                              distribution_type: str = "balanced",
                              interaction_style: str = "realistic") -> Tuple[List[SyntheticPlayer], nx.Graph]:
        """
        Create complete synthetic roster with network.
        
        Args:
            constraints: Roster construction constraints
            distribution_type: Type of salary distribution
            interaction_style: Style of player interactions
            
        Returns:
            Tuple of (players list, interaction network)
        """
        # Generate salary distribution
        salaries = self.generate_salary_distribution(constraints, distribution_type)
        
        # Create synthetic players
        players = []
        positions = ['PG', 'SG', 'SF', 'PF', 'C']
        
        for i, salary in enumerate(salaries):
            # Assign position (cycling through positions)
            position = positions[i % len(positions)]
            
            # Skill level roughly correlates with salary (with some noise)
            max_salary = max(salaries)
            base_skill = salary / max_salary
            skill_level = max(0.1, min(1.0, base_skill + random.uniform(-0.2, 0.2)))
            
            # Determine chemistry type based on salary and skill
            if salary >= sorted(salaries, reverse=True)[0] * 0.8:
                chemistry_type = "star"
                minutes_expectation = random.uniform(28, 36)
            elif salary >= sorted(salaries, reverse=True)[2] * 0.8 if len(salaries) >= 3 else salary * 0.8:
                chemistry_type = "glue" if random.random() < 0.3 else "role"
                minutes_expectation = random.uniform(20, 30)
            else:
                chemistry_type = "bench"
                minutes_expectation = random.uniform(8, 20)
            
            player = SyntheticPlayer(
                player_id=i + 1,
                salary=salary,
                position=position,
                skill_level=skill_level,
                chemistry_type=chemistry_type,
                minutes_expectation=minutes_expectation
            )
            players.append(player)
        
        # Generate interaction network
        network = self.generate_interaction_network(players, interaction_style)
        
        return players, network
    
    def generate_roster_variants(self, base_constraints: RosterConstraints,
                               num_variants: int = 50) -> List[Tuple[List[SyntheticPlayer], nx.Graph]]:
        """
        Generate multiple roster variants for comprehensive analysis.
        
        Args:
            base_constraints: Base constraints for all rosters
            num_variants: Number of roster variants to generate
            
        Returns:
            List of (players, network) tuples
        """
        logger.info(f"Generating {num_variants} synthetic roster variants")
        
        variants = []
        distribution_types = ["balanced", "top_heavy", "star_duo", "deep"]
        interaction_styles = ["realistic", "dense", "sparse", "hierarchical"]
        
        for i in range(num_variants):
            # Vary constraints slightly
            variant_constraints = RosterConstraints(
                salary_cap=base_constraints.salary_cap * random.uniform(0.95, 1.05),
                min_players=base_constraints.min_players,
                max_players=base_constraints.max_players,
                max_salary_share=base_constraints.max_salary_share * random.uniform(0.9, 1.1),
                min_salary=base_constraints.min_salary
            )
            
            # Select random distribution and interaction types
            dist_type = random.choice(distribution_types)
            interact_style = random.choice(interaction_styles)
            
            players, network = self.create_synthetic_roster(
                variant_constraints, dist_type, interact_style
            )
            
            variants.append((players, network))
        
        logger.info(f"Generated {len(variants)} roster variants")
        return variants


def benchmark_against_synthetic(historical_robustness: pd.DataFrame,
                               constraints: RosterConstraints,
                               num_synthetic: int = 100) -> Dict:
    """
    Benchmark historical rosters against synthetic alternatives.
    
    Args:
        historical_robustness: Robustness results from historical rosters
        constraints: Constraints for synthetic roster generation
        num_synthetic: Number of synthetic rosters to generate
        
    Returns:
        Comparison results
    """
    generator = SyntheticRosterGenerator()
    simulator = RobustnessSimulator()
    analyzer = RosterNetworkAnalyzer()
    
    # Generate synthetic rosters
    synthetic_variants = generator.generate_roster_variants(constraints, num_synthetic)
    
    # Analyze synthetic rosters
    synthetic_results = []
    
    for players, network in synthetic_variants:
        # Convert to salary data format for robustness analysis
        salary_data = pd.DataFrame([
            {
                'team_id': 9999,  # Synthetic team ID
                'season': "2023-24",
                'player_id': p.player_id,
                'player_name': f"Player_{p.player_id}",
                'salary': p.salary
            }
            for p in players
        ])
        
        # Run robustness analysis
        robustness = simulator.run_robustness_analysis(network, salary_data, 9999)
        
        synthetic_results.append({
            'avg_resilience_score': np.mean(robustness.resilience_scores),
            'min_resilience_score': np.min(robustness.resilience_scores),
            'max_performance_drop': np.max(robustness.performance_drops),
            'salary_gini': generator._compute_gini([p.salary for p in players]),
            'network_density': robustness.baseline_features.density,
            'total_salary': sum(p.salary for p in players)
        })
    
    synthetic_df = pd.DataFrame(synthetic_results)
    
    # Compare with historical
    comparison = {
        'historical_mean_resilience': historical_robustness['avg_resilience_score'].mean(),
        'synthetic_mean_resilience': synthetic_df['avg_resilience_score'].mean(),
        'historical_std_resilience': historical_robustness['avg_resilience_score'].std(),
        'synthetic_std_resilience': synthetic_df['avg_resilience_score'].std(),
        'synthetic_outperform_pct': (synthetic_df['avg_resilience_score'] > 
                                   historical_robustness['avg_resilience_score'].mean()).mean()
    }
    
    return comparison


if __name__ == "__main__":
    # Test synthetic roster generation
    constraints = RosterConstraints(
        salary_cap=120_000_000,
        min_players=13,
        max_players=15,
        max_salary_share=0.35,
        min_salary=500_000
    )
    
    generator = SyntheticRosterGenerator()
    
    # Generate a test roster
    players, network = generator.create_synthetic_roster(constraints, "balanced", "realistic")
    
    print(f"Generated roster with {len(players)} players:")
    total_salary = sum(p.salary for p in players)
    print(f"Total salary: ${total_salary:,.0f}")
    print(f"Network nodes: {network.number_of_nodes()}, edges: {network.number_of_edges()}")
    
    # Show top players
    sorted_players = sorted(players, key=lambda x: x.salary, reverse=True)
    print("\nTop 5 players by salary:")
    for i, p in enumerate(sorted_players[:5]):
        print(f"{i+1}. {p.chemistry_type} {p.position}: ${p.salary:,.0f} (skill: {p.skill_level:.2f})")
