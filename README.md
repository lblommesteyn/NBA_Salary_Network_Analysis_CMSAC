# Roster Geometry and Resilience

**A Network-Based Approach to Payroll Structure and Performance Stability in the NBA**

Presented at Carnegie Mellon Sports Analytics Conference - Reproducible Research Track

## Abstract

**Problem**: Prior salary-performance studies ignore network structure and resilience to player absences.

**Method**: We combine salary data, shared-minutes networks, and robustness simulations to quantify how different roster geometries handle disruptions.

**Findings**: Certain network shapes buffer better against player absences, with balanced salary meshes showing superior resilience compared to top-heavy structures.

**Contribution**: Open-data, reproducible framework for cap-efficient, resilient roster building with synthetic roster benchmarking.

## Research Questions

1. **Primary**: How do different roster network geometries affect team resilience to player disruptions (injuries/trades)?
2. **Secondary**: Do balanced salary distributions create more robust lineup networks than star-heavy allocations?
3. **Applied**: Can we identify optimal roster construction patterns under salary cap constraints?

## Key Innovations

- **Roster Geometry Metrics**: Novel integration of salary distribution with actual on-court interaction networks
- **Robustness Simulations**: Systematic testing of team performance under player removal scenarios
- **Synthetic Roster Generation**: Benchmarking against optimally constructed rosters under identical constraints
- **Resilience Quantification**: Mathematical framework for measuring roster stability

## Data Sources

- **Salary Data**: Basketball Reference, Spotrac
- **Lineup/Play-by-Play Data**: NBA API, pbpstats.com
- **Performance Metrics**: Basketball Reference advanced stats

## Project Structure

```
├── data/                   # Raw and processed data
├── src/                    # Source code
│   ├── data_collection/    # Data pipeline scripts
│   ├── network_analysis/   # Network feature computation
│   ├── visualization/      # Interactive visualization tools
│   └── analysis/          # Statistical analysis and modeling
├── notebooks/             # Jupyter notebooks for exploration
├── results/               # Generated results and figures
└── docs/                  # Documentation
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# Collect data for 2023-24 season
python src/data_collection/collect_all_data.py --season 2024

# Build network graphs
python src/network_analysis/build_networks.py

# Generate visualizations
python src/visualization/create_interactive_plots.py

# Run analysis
python src/analysis/playoff_correlation_analysis.py
```

## Reproducibility

This project follows reproducible research practices:
- All code is version controlled
- Data collection is fully automated
- Analysis parameters are configurable
- Results are generated with seeds for reproducibility

## Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{roster_geometry_2025,
  title={Roster Geometry: A Network Analysis of NBA Team Construction},
  author={Luke Blommesteyn, Lucian Lavric, Yuvraj Sharma},
  booktitle={Carnegie Mellon Sports Analytics Conference},
  year={2025}
}
```

## License

MIT License - See LICENSE file for details
