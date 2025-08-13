"""
Playoff Correlation Analysis Module

Analyzes the correlation between roster network features and playoff success.
Tests the hypothesis that certain "roster shapes" correlate with team performance.
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AnalysisResults:
    """Container for analysis results."""
    correlations: pd.DataFrame
    regression_results: Dict
    classification_results: Dict
    feature_importance: pd.DataFrame
    statistical_tests: Dict


class PlayoffCorrelationAnalyzer:
    """Analyzes correlation between network features and playoff success."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.network_features = [
            'salary_gini', 'salary_centralization', 'salary_assortativity',
            'network_density', 'clustering_coefficient', 'modularity',
            'degree_centralization', 'betweenness_centralization', 'closeness_centralization'
        ]
    
    def collect_playoff_data(self, season: str) -> pd.DataFrame:
        """
        Collect playoff success data for NBA teams.
        This is a simplified version - in practice, you'd get this from NBA API or other sources.
        
        Args:
            season: Season to analyze
            
        Returns:
            DataFrame with team playoff outcomes
        """
        # Simplified playoff data - in practice, collect from NBA API
        # For demonstration, creating realistic looking data based on historical patterns
        
        teams = list(range(1, 31))  # 30 NBA teams
        
        # Simulate realistic playoff outcomes
        np.random.seed(42)  # For reproducibility
        
        playoff_data = []
        for team_id in teams:
            # Simulate wins (30-70 range typical for NBA)
            wins = np.random.randint(15, 70)
            
            # Playoff probability based on wins (teams with 45+ wins more likely)
            playoff_prob = max(0.1, min(0.95, (wins - 30) / 40))
            made_playoffs = np.random.random() < playoff_prob
            
            # Playoff rounds (0=didn't make, 1=first round, 2=second round, etc.)
            if made_playoffs:
                # Better teams advance further
                advance_prob = max(0.1, min(0.8, (wins - 45) / 25))
                rounds_won = 0
                for round_num in range(4):  # Max 4 playoff rounds
                    if np.random.random() < advance_prob:
                        rounds_won += 1
                        advance_prob *= 0.6  # Decreasing probability each round
                    else:
                        break
            else:
                rounds_won = 0
            
            playoff_data.append({
                'team_id': team_id,
                'season': season,
                'wins': wins,
                'losses': 82 - wins,
                'win_percentage': wins / 82,
                'made_playoffs': made_playoffs,
                'playoff_rounds_won': rounds_won,
                'championship': rounds_won == 4
            })
        
        return pd.DataFrame(playoff_data)
    
    def merge_network_and_playoff_data(self,
                                     network_features_df: pd.DataFrame,
                                     playoff_data_df: pd.DataFrame) -> pd.DataFrame:
        """Merge network features with playoff outcome data."""
        
        # Map team_id to team abbreviations for consistent merging
        team_id_mapping = {
            1610612737: 'ATL',  # Atlanta Hawks
            1610612738: 'BOS',  # Boston Celtics
            1610612747: 'LAL',  # Los Angeles Lakers
            1610612744: 'GSW'   # Golden State Warriors
        }
        
        # Ensure both dataframes have a 'team' column for merging
        network_df = network_features_df.copy()
        playoff_df = playoff_data_df.copy()
        
        # Convert team_id to team abbreviations in network features if needed
        if 'team_id' in network_df.columns and 'team' not in network_df.columns:
            network_df['team'] = network_df['team_id'].map(team_id_mapping)
        elif 'team_id' in network_df.columns and 'team' in network_df.columns:
            # Update team column using team_id mapping if both exist
            network_df['team'] = network_df['team_id'].map(team_id_mapping).fillna(network_df['team'])
            
        # Ensure playoff data has team column
        if 'team' not in playoff_df.columns:
            logger.warning("No 'team' column found in playoff data")
            return network_features_df
            
        # Merge on team abbreviation
        merged = network_df.merge(
            playoff_df,
            on='team',
            how='left'  # Use left join to keep all network features
        )
        
        logger.info(f"Merged data contains {len(merged)} team-seasons")
        
        return merged
    
    def compute_correlations(self, merged_data: pd.DataFrame) -> pd.DataFrame:
        """Compute correlations between network features and playoff outcomes."""
        
        outcome_vars = ['wins', 'win_percentage', 'made_playoffs', 'playoff_rounds_won']
        available_features = [f for f in self.network_features if f in merged_data.columns]
        
        if not available_features:
            logger.warning("No network features available for correlation analysis")
            return pd.DataFrame()
        
        correlations = []
        
        for outcome in outcome_vars:
            if outcome not in merged_data.columns:
                continue
                
            for feature in available_features:
                # Filter out rows where either feature or outcome is missing
                valid_data = merged_data[[feature, outcome]].dropna()
                
                if len(valid_data) < 2:
                    logger.warning(f"Insufficient data for correlation between {feature} and {outcome}")
                    continue
                    
                # Compute Pearson correlation with aligned data
                corr_coef, p_value = stats.pearsonr(
                    valid_data[feature],
                    valid_data[outcome]
                )
                
                # Compute Spearman correlation (rank-based) with same aligned data
                spearman_coef, spearman_p = stats.spearmanr(
                    valid_data[feature],
                    valid_data[outcome]
                )
                
                correlations.append({
                    'network_feature': feature,
                    'outcome_variable': outcome,
                    'pearson_correlation': corr_coef,
                    'pearson_p_value': p_value,
                    'spearman_correlation': spearman_coef,
                    'spearman_p_value': spearman_p,
                    'significant_at_05': p_value < 0.05
                })
        
        correlations_df = pd.DataFrame(correlations)
        
        # Handle empty correlations (small dataset or no valid data)
        if correlations_df.empty:
            logger.warning("No correlations calculated - creating empty dataframe with expected columns")
            correlations_df = pd.DataFrame(columns=[
                'network_feature', 'outcome_variable', 'pearson_correlation', 'pearson_p_value',
                'spearman_correlation', 'spearman_p_value', 'significant_at_05', 'abs_correlation'
            ])
            return correlations_df
        
        # Sort by absolute correlation strength
        correlations_df['abs_correlation'] = correlations_df['pearson_correlation'].abs()
        correlations_df = correlations_df.sort_values('abs_correlation', ascending=False)
        
        return correlations_df
    
    def predict_wins(self, merged_data: pd.DataFrame) -> Dict:
        """Build predictive model for team wins using network features."""
        
        available_features = [f for f in self.network_features if f in merged_data.columns]
        
        if not available_features or 'wins' not in merged_data.columns:
            logger.warning("Insufficient data for wins prediction")
            return {}
        
        # Prepare data
        X = merged_data[available_features].fillna(0)
        y = merged_data['wins']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = rf_model.predict(X_test_scaled)
        
        # Evaluate model
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5, scoring='r2')
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': available_features,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return {
            'model': rf_model,
            'test_mse': mse,
            'test_r2': r2,
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            'feature_importance': feature_importance,
            'predictions': y_pred,
            'actual': y_test
        }
    
    def predict_playoff_success(self, merged_data: pd.DataFrame) -> Dict:
        """Build classification model for playoff success."""
        # Prepare features and target
        available_features = [col for col in merged_data.columns if col in [
            'salary_gini', 'network_density', 'modularity', 'centralization_index',
            'assortativity', 'clustering_coefficient', 'avg_shortest_path_length',
            'total_salary', 'avg_salary'
        ]]
        
        if not available_features:
            logger.warning("No valid features found for classification")
            return {}
            
        # Handle missing playoff data
        if 'made_playoffs' not in merged_data.columns:
            logger.warning("No 'made_playoffs' column found - skipping classification")
            return {}
            
        # Drop rows with missing playoff outcomes
        valid_data = merged_data.dropna(subset=['made_playoffs'])
        if len(valid_data) < 2:
            logger.warning(f"Insufficient valid playoff data for classification (n={len(valid_data)})")
            return {}
            
        X = valid_data[available_features].fillna(0)
        y = valid_data['made_playoffs'].astype(int)
        
        # Split data - disable stratification for small datasets
        if len(X) < 10 or len(np.unique(y)) < 2 or np.min(np.bincount(y)) < 2:
            # For small datasets or insufficient class samples, don't use stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest classifier
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_classifier.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = rf_classifier.predict(X_test_scaled)
        y_pred_proba = rf_classifier.predict_proba(X_test_scaled)[:, 1]
        
        # Cross-validation - skip for very small datasets
        if len(X_train) < 6 or len(np.unique(y_train)) < 2:
            # For very small datasets or single class, skip cross-validation
            cv_scores = np.array([0.5])  # Dummy score
            logger.warning(f"Skipping cross-validation for small dataset (n={len(X_train)})")
        else:
            cv_scores = cross_val_score(rf_classifier, X_train_scaled, y_train, cv=5, scoring='accuracy')
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': available_features,
            'importance': rf_classifier.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return {
            'model': rf_classifier,
            'cv_accuracy_mean': cv_scores.mean(),
            'cv_accuracy_std': cv_scores.std(),
            'feature_importance': feature_importance,
            'predictions': y_pred,
            'prediction_probabilities': y_pred_proba,
            'actual': y_test
        }
    
    def statistical_hypothesis_tests(self, merged_data: pd.DataFrame) -> Dict:
        """Perform statistical hypothesis tests on key relationships."""
        
        tests_results = {}
        
        # Test 1: Do teams with lower salary Gini (more balanced) perform better?
        if 'salary_gini' in merged_data.columns and 'wins' in merged_data.columns:
            # Split teams into high vs low salary inequality
            median_gini = merged_data['salary_gini'].median()
            low_gini = merged_data[merged_data['salary_gini'] <= median_gini]['wins']
            high_gini = merged_data[merged_data['salary_gini'] > median_gini]['wins']
            
            t_stat, p_value = stats.ttest_ind(low_gini, high_gini)
            
            tests_results['salary_balance_wins_test'] = {
                'hypothesis': 'Teams with more balanced salary distribution win more games',
                'test': 't-test',
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'low_gini_mean_wins': low_gini.mean(),
                'high_gini_mean_wins': high_gini.mean()
            }
        
        # Test 2: Do teams with higher modularity have better playoff success?
        if 'modularity' in merged_data.columns and 'made_playoffs' in merged_data.columns:
            playoff_teams = merged_data[merged_data['made_playoffs'] == True]['modularity']
            non_playoff_teams = merged_data[merged_data['made_playoffs'] == False]['modularity']
            
            t_stat, p_value = stats.ttest_ind(playoff_teams, non_playoff_teams)
            
            tests_results['modularity_playoff_test'] = {
                'hypothesis': 'Playoff teams have higher network modularity',
                'test': 't-test',
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'playoff_mean_modularity': playoff_teams.mean(),
                'non_playoff_mean_modularity': non_playoff_teams.mean()
            }
        
        # Test 3: Correlation between network density and team success
        if 'network_density' in merged_data.columns and 'win_percentage' in merged_data.columns:
            corr, p_value = stats.pearsonr(
                merged_data['network_density'], 
                merged_data['win_percentage']
            )
            
            tests_results['density_success_correlation'] = {
                'hypothesis': 'Network density correlates with team success',
                'test': 'Pearson correlation',
                'correlation': corr,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        
        return tests_results
    
    def analyze_roster_geometry_hypothesis(self, merged_data: pd.DataFrame) -> AnalysisResults:
        """
        Complete analysis of the roster geometry hypothesis.
        
        Args:
            merged_data: DataFrame with network features and playoff outcomes
            
        Returns:
            AnalysisResults object with all findings
        """
        logger.info("Starting comprehensive roster geometry analysis")
        
        # Compute correlations
        correlations = self.compute_correlations(merged_data)
        
        # Build predictive models
        regression_results = self.predict_wins(merged_data)
        classification_results = self.predict_playoff_success(merged_data)
        
        # Combine feature importance
        feature_importance = pd.DataFrame()
        if regression_results and 'feature_importance' in regression_results:
            reg_importance = regression_results['feature_importance'].copy()
            reg_importance['model'] = 'wins_prediction'
            feature_importance = pd.concat([feature_importance, reg_importance], ignore_index=True)
        
        if classification_results and 'feature_importance' in classification_results:
            class_importance = classification_results['feature_importance'].copy()
            class_importance['model'] = 'playoff_prediction'
            feature_importance = pd.concat([feature_importance, class_importance], ignore_index=True)
        
        # Statistical hypothesis tests
        statistical_tests = self.statistical_hypothesis_tests(merged_data)
        
        logger.info("Analysis complete")
        
        return AnalysisResults(
            correlations=correlations,
            regression_results=regression_results,
            classification_results=classification_results,
            feature_importance=feature_importance,
            statistical_tests=statistical_tests
        )


def run_playoff_correlation_analysis(network_features_df: pd.DataFrame, 
                                   season: str = '2023-24') -> AnalysisResults:
    """
    Main function to run playoff correlation analysis.
    
    Args:
        network_features_df: DataFrame with network analysis results
        season: Season to analyze
        
    Returns:
        AnalysisResults object
    """
    analyzer = PlayoffCorrelationAnalyzer()
    
    # Collect playoff data
    playoff_data = analyzer.collect_playoff_data(season)
    
    # Merge with network features
    merged_data = analyzer.merge_network_and_playoff_data(network_features_df, playoff_data)
    
    # Run comprehensive analysis
    results = analyzer.analyze_roster_geometry_hypothesis(merged_data)
    
    return results


def print_analysis_summary(results: AnalysisResults):
    """Print a summary of key findings."""
    print("="*60)
    print("ROSTER GEOMETRY ANALYSIS SUMMARY")
    print("="*60)
    
    # Top correlations
    if not results.correlations.empty:
        print("\nTOP 5 STRONGEST CORRELATIONS:")
        print("-"*40)
        top_corr = results.correlations.head()
        for _, row in top_corr.iterrows():
            significance = "***" if row['significant_at_05'] else ""
            print(f"{row['network_feature']} -> {row['outcome_variable']}: "
                  f"r={row['pearson_correlation']:.3f} {significance}")
    
    # Statistical tests
    if results.statistical_tests:
        print(f"\nHYPOTHESIS TESTS:")
        print("-"*40)
        for test_name, test_result in results.statistical_tests.items():
            significance = "SIGNIFICANT" if test_result['significant'] else "NOT SIGNIFICANT"
            print(f"{test_result['hypothesis']}: {significance} (p={test_result['p_value']:.3f})")
    
    # Model performance
    if results.regression_results:
        print(f"\nWINS PREDICTION MODEL:")
        print("-"*40)
        print(f"R² Score: {results.regression_results['test_r2']:.3f}")
        print(f"Cross-validation R² (mean±std): {results.regression_results['cv_r2_mean']:.3f}±{results.regression_results['cv_r2_std']:.3f}")
    
    if results.classification_results:
        print(f"\nPLAYOFF PREDICTION MODEL:")
        print("-"*40)
        print(f"Cross-validation Accuracy (mean±std): {results.classification_results['cv_accuracy_mean']:.3f}±{results.classification_results['cv_accuracy_std']:.3f}")
    
    print("="*60)


if __name__ == "__main__":
    # Test analysis with dummy data
    import sys
    sys.path.append('../../')
    
    # Create dummy network features data
    teams = list(range(1, 31))
    np.random.seed(42)
    
    dummy_network_features = pd.DataFrame({
        'team_id': teams,
        'season': ['2023-24'] * len(teams),
        'salary_gini': np.random.uniform(0.3, 0.7, len(teams)),
        'salary_centralization': np.random.uniform(0.2, 0.5, len(teams)),
        'network_density': np.random.uniform(0.3, 0.8, len(teams)),
        'modularity': np.random.uniform(0.1, 0.6, len(teams)),
        'clustering_coefficient': np.random.uniform(0.4, 0.9, len(teams)),
        'degree_centralization': np.random.uniform(0.2, 0.7, len(teams))
    })
    
    # Run analysis
    results = run_playoff_correlation_analysis(dummy_network_features)
    
    # Print summary
    print_analysis_summary(results)
