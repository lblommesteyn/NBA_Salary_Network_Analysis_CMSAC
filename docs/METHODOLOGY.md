# Methodology: Roster Geometry Analysis

## Abstract

This research introduces a novel approach to analyzing NBA team construction through network graph theory. We model each team as a network where players are nodes (sized by salary, colored by performance metrics) and edges represent shared on-court minutes. By extracting geometric and topological features from these networks, we investigate whether certain "roster shapes" correlate with playoff success.

## Research Questions

1. **Primary Hypothesis**: Do balanced salary "meshes" outperform top-heavy rosters when accounting for actual on-court interactions?

2. **Secondary Questions**:
   - Is there an optimal level of network modularity for team success?
   - How does lineup connectivity relate to playoff performance?
   - Do highly centralized salary structures correlate with reduced team performance?

## Data Sources

### Salary Data
- **Primary**: Basketball Reference contract data
- **Secondary**: Spotrac salary database
- **Variables**: Player salary, contract details, team affiliation

### Lineup Data
- **Primary**: NBA API play-by-play data
- **Secondary**: pbpstats.com lineup statistics
- **Variables**: Shared on-court minutes between player pairs

### Performance Data
- **Source**: Basketball Reference advanced statistics
- **Variables**: Box Plus/Minus (BPM), Win Shares per 48 minutes (WS/48)

### Outcome Variables
- Regular season wins/losses
- Win percentage
- Playoff qualification
- Playoff rounds advanced
- Championship success

## Network Construction

### Node Attributes
- **Size**: Proportional to player salary
- **Color**: Mapped to impact metrics (BPM or WS/48)
- **Position**: Determined by network layout algorithms

### Edge Weights
- **Weight**: Shared on-court minutes between players
- **Threshold**: Minimum 10 minutes shared to create edge
- **Aggregation**: Summed across all games in season

### Network Types
- **Undirected**: Shared minutes are symmetric
- **Weighted**: Edge weights represent interaction strength
- **Simple**: No self-loops or multiple edges

## Feature Extraction

### 1. Salary Distribution Features

#### Gini Coefficient
Measures salary inequality within team roster:
```
Gini = (n + 1 - 2 * Σ(cumsum_i)) / (n * total_salary)
```
- Range: [0, 1]
- 0 = Perfect equality
- 1 = Maximum inequality

#### Salary Centralization
Proportion of total salary held by highest-paid player:
```
Centralization = max_salary / total_salary
```

#### Salary Assortativity
Tendency for players with similar salaries to share court time:
```
r = Σ(salary_i * salary_j * weight_ij) / Σ(weight_ij)
```

### 2. Network Topology Features

#### Density
Proportion of possible edges that exist:
```
Density = 2 * |E| / (|V| * (|V| - 1))
```

#### Clustering Coefficient
Local connectivity measure:
```
C = (1/n) * Σ(2 * triangles_i / (degree_i * (degree_i - 1)))
```

#### Average Path Length
Mean shortest path between all node pairs (connected components only)

### 3. Community Structure

#### Modularity
Strength of community division using Louvain algorithm:
```
Q = (1/2m) * Σ(A_ij - k_i*k_j/2m) * δ(c_i, c_j)
```

#### Number of Communities
Count of distinct communities detected

#### Community Size Distribution
Size of largest community relative to network size

### 4. Centralization Measures

#### Degree Centralization
Variation in node degrees:
```
CD = Σ(max_degree - degree_i) / max_possible_variation
```

#### Betweenness Centralization
Variation in betweenness centrality values

#### Closeness Centralization
Variation in closeness centrality values (connected graphs only)

## Statistical Analysis

### Correlation Analysis
- **Pearson Correlation**: Linear relationships
- **Spearman Correlation**: Monotonic relationships
- **Significance Testing**: p < 0.05 threshold

### Predictive Modeling
- **Algorithm**: Random Forest (handles non-linear relationships)
- **Cross-Validation**: 5-fold CV for robust estimates
- **Metrics**: R² for regression, accuracy for classification

### Hypothesis Testing
- **T-tests**: Compare means between groups
- **Mann-Whitney U**: Non-parametric alternative
- **Multiple Comparisons**: Bonferroni correction where applicable

## Key Innovations

### 1. Integration of Salary and Performance
Traditional analyses examine salary distribution or on-court performance separately. Our approach combines both through network structure.

### 2. Actual Lineup Data
Rather than assuming uniform player interactions, we use real shared minutes to weight network connections.

### 3. Geometric Interpretation
Network topology features provide geometric interpretation of team structure (e.g., "star-heavy" vs "balanced mesh").

### 4. Reproducible Pipeline
End-to-end automated pipeline from data collection through analysis and visualization.

## Limitations

### Data Limitations
- Play-by-play data may have recording inconsistencies
- Salary data excludes performance incentives and bonuses
- Limited historical depth for some statistics

### Methodological Limitations
- Network construction assumes minutes played reflects interaction quality
- Community detection algorithms may not capture basketball-specific groupings
- Causality cannot be established through correlation analysis

### External Validity
- Analysis focused on NBA; may not generalize to other leagues
- Salary cap rules specific to NBA context
- Cultural and coaching factors not captured in network structure

## Future Directions

### Enhanced Data Sources
- Integration of player tracking data for more precise interaction measures
- Incorporation of coaching decisions and strategic contexts
- Addition of injury and availability data

### Advanced Network Analysis
- Temporal network analysis across seasons
- Multi-layer networks (offense/defense specific)
- Dynamic community detection

### Broader Applications
- Extension to other professional sports
- Application to team construction in other domains
- Real-time roster optimization tools

## Reproducibility Statement

All code, data processing steps, and analysis procedures are documented and version-controlled. The analysis can be reproduced using the provided Docker environment and dependency specifications. Random seeds are set for all stochastic procedures to ensure exact replication of results.
