"""
Test Script for Roster Geometry Analysis Pipeline

Validates all components of the analysis pipeline with demo data.
Run this script to ensure everything works correctly before the conference presentation.
"""

import sys
import traceback
from pathlib import Path
import pandas as pd
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

try:
    from src.main_pipeline import RosterGeometryPipeline, create_demo_data
    from src.network_analysis.roster_networks import RosterNetworkAnalyzer
    from src.visualization.interactive_plots import RosterNetworkVisualizer
    from src.analysis.playoff_correlation_analysis import PlayoffCorrelationAnalyzer
except ImportError as e:
    logger.error(f"Failed to import project modules: {e}")
    logger.error("Make sure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)


class PipelineTester:
    """Test suite for the roster geometry analysis pipeline."""
    
    def __init__(self):
        self.results = {
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'failures': []
        }
    
    def run_test(self, test_name: str, test_func):
        """Run a single test and record results."""
        self.results['tests_run'] += 1
        logger.info(f"Running test: {test_name}")
        
        try:
            test_func()
            self.results['tests_passed'] += 1
            logger.info(f"✓ PASSED: {test_name}")
            return True
        except Exception as e:
            self.results['tests_failed'] += 1
            self.results['failures'].append({
                'test': test_name,
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            logger.error(f"✗ FAILED: {test_name} - {e}")
            return False
    
    def test_demo_data_creation(self):
        """Test demo data generation."""
        data = create_demo_data("2023-24")
        
        assert 'salary_data' in data, "Missing salary_data in demo data"
        assert 'lineup_data' in data, "Missing lineup_data in demo data"
        
        salary_df = data['salary_data']
        lineup_df = data['lineup_data']
        
        assert len(salary_df) > 0, "Empty salary data"
        assert len(lineup_df) > 0, "Empty lineup data"
        
        # Check required columns
        salary_required_cols = ['team_id', 'season', 'player_id', 'player_name', 'salary']
        for col in salary_required_cols:
            assert col in salary_df.columns, f"Missing column {col} in salary data"
        
        lineup_required_cols = ['team_id', 'season', 'player1_id', 'player2_id', 'shared_minutes']
        for col in lineup_required_cols:
            assert col in lineup_df.columns, f"Missing column {col} in lineup data"
        
        # Check data quality
        assert salary_df['salary'].min() > 0, "Invalid salary values"
        assert lineup_df['shared_minutes'].min() > 0, "Invalid shared minutes values"
        
        logger.info(f"Demo data: {len(salary_df)} players, {len(lineup_df)} player pairs")
    
    def test_network_analysis(self):
        """Test network analysis components."""
        data = create_demo_data("2023-24")
        analyzer = RosterNetworkAnalyzer()
        
        # Test single team network
        team_id = 1
        G = analyzer.build_roster_network(
            data['salary_data'], data['lineup_data'], team_id, "2023-24"
        )
        
        assert G.number_of_nodes() > 0, "Network has no nodes"
        assert G.number_of_edges() > 0, "Network has no edges"
        
        # Test network features computation
        features = analyzer.analyze_network(G)
        
        assert 0 <= features.salary_gini <= 1, f"Invalid Gini coefficient: {features.salary_gini}"
        assert 0 <= features.density <= 1, f"Invalid density: {features.density}"
        assert features.modularity >= -0.5, f"Invalid modularity: {features.modularity}"  # Modularity can be negative
        
        # Test full analysis
        network_features_df = analyzer.analyze_all_teams(
            data['salary_data'], data['lineup_data'], "2023-24"
        )
        
        assert len(network_features_df) > 0, "No network features computed"
        assert 'team_id' in network_features_df.columns, "Missing team_id column"
        assert 'salary_gini' in network_features_df.columns, "Missing salary_gini column"
        
        logger.info(f"Network analysis: {len(network_features_df)} teams analyzed")
    
    def test_playoff_analysis(self):
        """Test playoff correlation analysis."""
        data = create_demo_data("2023-24")
        analyzer = RosterNetworkAnalyzer()
        
        # Generate network features
        network_features_df = analyzer.analyze_all_teams(
            data['salary_data'], data['lineup_data'], "2023-24"
        )
        
        # Test playoff analysis
        playoff_analyzer = PlayoffCorrelationAnalyzer()
        
        # Test playoff data generation
        playoff_data = playoff_analyzer.collect_playoff_data("2023-24")
        assert len(playoff_data) > 0, "No playoff data generated"
        assert 'wins' in playoff_data.columns, "Missing wins column in playoff data"
        assert 'made_playoffs' in playoff_data.columns, "Missing made_playoffs column"
        
        # Test data merging
        merged_data = playoff_analyzer.merge_network_and_playoff_data(
            network_features_df, playoff_data
        )
        assert len(merged_data) > 0, "No merged data created"
        
        # Test correlation analysis
        correlations = playoff_analyzer.compute_correlations(merged_data)
        assert len(correlations) > 0, "No correlations computed"
        
        # Test predictive modeling
        regression_results = playoff_analyzer.predict_wins(merged_data)
        assert 'test_r2' in regression_results, "Missing R² score in regression results"
        
        classification_results = playoff_analyzer.predict_playoff_success(merged_data)
        assert 'cv_accuracy_mean' in classification_results, "Missing accuracy in classification results"
        
        logger.info(f"Playoff analysis: {len(correlations)} correlations computed")
    
    def test_visualization(self):
        """Test visualization components."""
        data = create_demo_data("2023-24")
        analyzer = RosterNetworkAnalyzer()
        visualizer = RosterNetworkVisualizer()
        
        # Generate network features
        network_features_df = analyzer.analyze_all_teams(
            data['salary_data'], data['lineup_data'], "2023-24"
        )
        
        # Test correlation visualization
        corr_fig = visualizer.plot_feature_correlations(network_features_df)
        assert corr_fig is not None, "Failed to create correlation plot"
        
        # Test team comparison
        comparison_fig = visualizer.plot_team_comparison(network_features_df)
        assert comparison_fig is not None, "Failed to create team comparison plot"
        
        # Test scatter plot
        if 'salary_gini' in network_features_df.columns and 'modularity' in network_features_df.columns:
            scatter_fig = visualizer.plot_salary_vs_performance(
                network_features_df, 'salary_gini', 'modularity'
            )
            assert scatter_fig is not None, "Failed to create scatter plot"
        
        # Test network visualization
        team_id = 1
        G = analyzer.build_roster_network(
            data['salary_data'], data['lineup_data'], team_id, "2023-24"
        )
        
        if G.number_of_nodes() > 0:
            network_fig = visualizer.plot_network_graph(G, "Test Team")
            assert network_fig is not None, "Failed to create network plot"
        
        logger.info("Visualization tests completed successfully")
    
    def test_full_pipeline(self):
        """Test complete pipeline execution."""
        pipeline = RosterGeometryPipeline(data_dir="test_data", results_dir="test_results")
        
        # Create demo data
        data = create_demo_data("2023-24")
        
        # Test network analysis
        network_features = pipeline.run_network_analysis(
            data['salary_data'], data['lineup_data'], "2023-24"
        )
        assert len(network_features) > 0, "Pipeline network analysis failed"
        
        # Test playoff analysis
        analysis_results = pipeline.run_playoff_analysis(network_features, "2023-24")
        assert analysis_results is not None, "Pipeline playoff analysis failed"
        assert not analysis_results.correlations.empty, "No correlations in pipeline results"
        
        # Test visualization creation
        visualizations = pipeline.create_visualizations(network_features)
        assert len(visualizations) > 0, "No visualizations created by pipeline"
        
        logger.info("Full pipeline test completed successfully")
    
    def test_data_validation(self):
        """Test data validation and error handling."""
        analyzer = RosterNetworkAnalyzer()
        
        # Test empty data handling
        empty_salary = pd.DataFrame()
        empty_lineup = pd.DataFrame()
        
        G = analyzer.build_roster_network(empty_salary, empty_lineup, 1, "2023-24")
        assert G.number_of_nodes() == 0, "Should handle empty data gracefully"
        
        features = analyzer.analyze_network(G)
        assert features.node_count == 0, "Should handle empty network gracefully"
        
        # Test invalid data handling
        invalid_salary = pd.DataFrame({
            'team_id': [1, 1],
            'season': ['2023-24', '2023-24'],
            'player_id': [1, 2],
            'player_name': ['Player1', 'Player2'],
            'salary': [-1000, 0]  # Invalid salaries
        })
        
        # Should not crash with invalid data
        G_invalid = analyzer.build_roster_network(invalid_salary, empty_lineup, 1, "2023-24")
        assert G_invalid.number_of_nodes() >= 0, "Should handle invalid data without crashing"
        
        logger.info("Data validation tests completed")
    
    def run_all_tests(self):
        """Run all tests and report results."""
        logger.info("="*60)
        logger.info("STARTING ROSTER GEOMETRY PIPELINE TESTS")
        logger.info("="*60)
        
        # Define tests
        tests = [
            ("Demo Data Creation", self.test_demo_data_creation),
            ("Network Analysis", self.test_network_analysis),
            ("Playoff Analysis", self.test_playoff_analysis),
            ("Visualization", self.test_visualization),
            ("Data Validation", self.test_data_validation),
            ("Full Pipeline", self.test_full_pipeline),
        ]
        
        # Run tests
        for test_name, test_func in tests:
            self.run_test(test_name, test_func)
        
        # Report results
        self.print_test_summary()
        
        # Return success status
        return self.results['tests_failed'] == 0
    
    def print_test_summary(self):
        """Print test results summary."""
        logger.info("="*60)
        logger.info("TEST RESULTS SUMMARY")
        logger.info("="*60)
        logger.info(f"Tests Run: {self.results['tests_run']}")
        logger.info(f"Tests Passed: {self.results['tests_passed']}")
        logger.info(f"Tests Failed: {self.results['tests_failed']}")
        
        if self.results['tests_failed'] > 0:
            logger.error("\nFAILURES:")
            for failure in self.results['failures']:
                logger.error(f"- {failure['test']}: {failure['error']}")
        else:
            logger.info("\n🎉 ALL TESTS PASSED! 🎉")
            logger.info("The pipeline is ready for the Carnegie Mellon Sports Analytics Conference!")
        
        logger.info("="*60)


def main():
    """Main test execution function."""
    tester = PipelineTester()
    success = tester.run_all_tests()
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
