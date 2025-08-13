"""
Streamlit Dashboard for Roster Geometry Analysis

Interactive web application for exploring NBA roster network analysis results.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.main_pipeline import RosterGeometryPipeline, create_demo_data
from src.network_analysis.roster_networks import RosterNetworkAnalyzer
from src.visualization.interactive_plots import RosterNetworkVisualizer
from src.analysis.playoff_correlation_analysis import PlayoffCorrelationAnalyzer

# Page config
st.set_page_config(
    page_title="Roster Geometry Analysis",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_demo_data():
    """Load demo data with caching."""
    return create_demo_data("2023-24")

@st.cache_data
def load_real_data(season="2023-24"):
    """Load real NBA data from CSV files."""
    project_root = Path(__file__).parent.parent
    
    # Load real salary data
    salary_file = project_root / "data" / f"real_nba_salaries_{season.replace('-', '_')}.csv"
    if salary_file.exists():
        salary_data = pd.read_csv(salary_file)
        
        # Map columns to expected format
        if 'team' in salary_data.columns and 'team_id' not in salary_data.columns:
            salary_data['team_id'] = salary_data['team']  # Map team -> team_id
        if 'player_name' in salary_data.columns and 'player' not in salary_data.columns:
            salary_data['player'] = salary_data['player_name']  # Map player_name -> player
        if 'player_name' in salary_data.columns and 'player_id' not in salary_data.columns:
            # Create simple player_id from player names for matching
            salary_data['player_id'] = salary_data['player_name']
            
    else:
        st.error(f"Real salary data not found: {salary_file}")
        return None
    
    # Try to load real lineup data (from previous nba_api collection)
    lineup_files = list((project_root / "data").glob(f"lineup_data_{season}*.csv"))
    if lineup_files:
        # Use the most recent file (sort by name, last one should be most recent)
        lineup_files.sort()
        lineup_data = pd.read_csv(lineup_files[-1])
        st.info(f"Using real lineup data from: {lineup_files[-1].name}")
    else:
        # If no real lineup data, create minimal demo lineup data for the real teams
        st.warning("No real lineup data found, generating minimal demo data for real teams")
        lineup_data = create_demo_data(season)['lineup_data']
    
    return {
        'salary_data': salary_data,
        'lineup_data': lineup_data
    }


@st.cache_data
def run_analysis_cached(salary_data_dict, lineup_data_dict, season):
    """Run analysis with caching."""
    salary_data = pd.DataFrame(salary_data_dict)
    lineup_data = pd.DataFrame(lineup_data_dict)
    
    pipeline = RosterGeometryPipeline()
    network_features = pipeline.run_network_analysis(salary_data, lineup_data, season)
    analysis_results = pipeline.run_playoff_analysis(network_features, season)
    
    return network_features, analysis_results


def main():
    """Main dashboard function."""
    
    # Header
    st.markdown('<div class="main-header">🏀 Roster Geometry Analysis</div>', unsafe_allow_html=True)
    st.markdown("**A Network Analysis Approach to NBA Team Construction**")
    st.markdown("*Carnegie Mellon Sports Analytics Conference - Reproducible Research Track*")
    
    # Sidebar
    st.sidebar.title("Analysis Settings")
    
    # Data source selection
    data_source = st.sidebar.selectbox(
        "Data Source",
        ["Real NBA Data", "Demo Data", "Upload Data", "Live NBA API"],
        help="Choose your data source. Real NBA Data uses collected salary/lineup data."
    )
    
    season = st.sidebar.selectbox(
        "Season",
        ["2023-24", "2022-23", "2021-22"],
        index=0
    )
    
    # Load data based on selection
    if data_source == "Real NBA Data":
        with st.spinner("Loading real NBA data..."):
            data = load_real_data(season)
            if data is None:
                st.error("Failed to load real NBA data. Please check data files.")
                return
            salary_data = data['salary_data']
            lineup_data = data['lineup_data']
        
        # Show data info
        teams = salary_data['team'].unique() if 'team' in salary_data.columns else salary_data.columns[:4]
        st.sidebar.success(f"Real NBA data loaded: {len(salary_data)} players from teams: {', '.join(teams[:4])}")
        
    elif data_source == "Demo Data":
        with st.spinner("Loading demo data..."):
            data = load_demo_data()
            salary_data = data['salary_data']
            lineup_data = data['lineup_data']
        
        st.sidebar.success(f"Demo data loaded: {len(salary_data)} players, {len(lineup_data)} player pairs")
        
    elif data_source == "Upload Data":
        st.sidebar.markdown("### Upload CSV Files")
        
        salary_file = st.sidebar.file_uploader("Salary Data CSV", type=['csv'])
        lineup_file = st.sidebar.file_uploader("Lineup Data CSV", type=['csv'])
        
        if salary_file and lineup_file:
            salary_data = pd.read_csv(salary_file)
            lineup_data = pd.read_csv(lineup_file)
            st.sidebar.success("Files uploaded successfully!")
        else:
            st.warning("Please upload both salary and lineup data files.")
            return
            
    else:  # Live NBA API
        st.sidebar.warning("Live NBA API integration requires API setup.")
        st.info("For this demo, please use Demo Data or upload your own files.")
        return
    
    # Run analysis
    if st.sidebar.button("Run Analysis", type="primary"):
        with st.spinner("Running roster geometry analysis..."):
            # Convert to dict for caching
            salary_dict = salary_data.to_dict('records')
            lineup_dict = lineup_data.to_dict('records')
            
            network_features, analysis_results = run_analysis_cached(
                salary_dict, lineup_dict, season
            )
        
        st.session_state['network_features'] = network_features
        st.session_state['analysis_results'] = analysis_results
        st.session_state['salary_data'] = salary_data
        st.session_state['lineup_data'] = lineup_data
        
        st.sidebar.success("Analysis complete!")
    
    # Display results if analysis has been run
    if 'network_features' in st.session_state:
        display_analysis_results()
    else:
        display_methodology()


def display_methodology():
    """Display methodology and concept explanation."""
    st.markdown('<div class="section-header">Research Concept</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### The Roster Geometry Hypothesis
        
        This analysis models NBA teams as **network graphs** where:
        
        - **Nodes** = Players (sized by salary, colored by impact metrics like BPM or WS/48)
        - **Edges** = Shared on-court minutes between players
        
        We extract geometric and topological features from these networks to test whether certain 
        "roster shapes" correlate with playoff success.
        
        ### Key Network Features
        
        1. **Salary Distribution**
           - Gini coefficient (salary inequality)
           - Salary centralization 
           - Salary assortativity (do similar salaries connect?)
        
        2. **Network Topology**
           - Density (how connected is the team?)
           - Clustering coefficient (local connectivity)
           - Community structure (modularity)
        
        3. **Centralization Measures**
           - Degree, betweenness, closeness centralization
           - How hierarchical is the team structure?
        
        ### Research Questions
        
        - Do balanced salary "meshes" outperform top-heavy rosters?
        - Is there an optimal level of network modularity for team success?
        - How does lineup connectivity relate to playoff performance?
        """)
    
    with col2:
        # Create a sample network visualization
        st.markdown("### Sample Roster Network")
        
        # Create a small demo network
        G = nx.Graph()
        nodes = ["Star1", "Star2", "Starter1", "Starter2", "Starter3", "Bench1", "Bench2"]
        salaries = [40, 35, 15, 12, 8, 3, 2]  # In millions
        
        for i, (node, salary) in enumerate(zip(nodes, salaries)):
            G.add_node(node, salary=salary)
        
        # Add edges (connections)
        edges = [("Star1", "Star2"), ("Star1", "Starter1"), ("Star2", "Starter2"), 
                ("Starter1", "Starter2"), ("Starter1", "Starter3"), 
                ("Starter2", "Bench1"), ("Starter3", "Bench2")]
        
        for edge in edges:
            G.add_edge(edge[0], edge[1], weight=np.random.randint(100, 500))
        
        # Create visualization
        pos = nx.spring_layout(G, k=2)
        
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        node_sizes = [G.nodes[node]['salary'] for node in G.nodes()]
        
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='rgba(125,125,125,0.5)'),
            hoverinfo='none',
            mode='lines'
        ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(
                size=[s*2 for s in node_sizes],
                color=node_sizes,
                colorscale='Viridis',
                showscale=False,
                line=dict(width=2, color='white')
            ),
            text=list(G.nodes()),
            textposition="middle center",
            hovertemplate='<b>%{text}</b><br>Salary: $%{marker.color}M<extra></extra>',
            showlegend=False
        ))
        
        fig.update_layout(
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(l=0, r=0, t=0, b=0),
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Node size = Salary | Edges = Shared minutes")


def display_analysis_results():
    """Display analysis results with interactive visualizations."""
    
    network_features = st.session_state['network_features']
    analysis_results = st.session_state['analysis_results']
    salary_data = st.session_state['salary_data']
    lineup_data = st.session_state['lineup_data']
    
    # Overview metrics
    st.markdown('<div class="section-header">Analysis Overview</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Teams Analyzed",
            len(network_features),
            help="Number of teams in the analysis"
        )
    
    with col2:
        avg_density = network_features['network_density'].mean()
        st.metric(
            "Avg Network Density",
            f"{avg_density:.3f}",
            help="Average network density across all teams"
        )
    
    with col3:
        avg_gini = network_features['salary_gini'].mean()
        st.metric(
            "Avg Salary Gini",
            f"{avg_gini:.3f}",
            help="Average salary inequality (0=equal, 1=very unequal)"
        )
    
    with col4:
        if not analysis_results.correlations.empty:
            significant_corr = len(analysis_results.correlations[analysis_results.correlations['significant_at_05']])
            st.metric(
                "Significant Correlations",
                significant_corr,
                help="Number of statistically significant correlations found"
            )
    
    # Network Features Analysis
    st.markdown('<div class="section-header">Network Features Analysis</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["Feature Correlations", "Team Comparison", "Playoff Analysis", "Individual Networks"])
    
    with tab1:
        st.subheader("Feature Correlation Matrix")
        
        # Create correlation matrix
        numeric_cols = network_features.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['team_id']]
        
        if len(numeric_cols) >= 2:
            corr_matrix = network_features[numeric_cols].corr()
            
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale='RdBu',
                color_continuous_midpoint=0
            )
            fig.update_layout(
                title="Network Features Correlation Matrix",
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Top correlations
            st.subheader("Strongest Correlations")
            if not analysis_results.correlations.empty:
                top_corr = analysis_results.correlations.head(10)
                st.dataframe(
                    top_corr[['network_feature', 'outcome_variable', 'pearson_correlation', 'pearson_p_value', 'significant_at_05']],
                    use_container_width=True
                )
    
    with tab2:
        st.subheader("Team Comparison")
        
        # Feature selection for comparison
        feature_options = [col for col in network_features.columns 
                          if col not in ['team_id', 'season']]
        
        col1, col2 = st.columns(2)
        with col1:
            x_feature = st.selectbox("X-axis Feature", feature_options, 
                                   index=feature_options.index('salary_gini') if 'salary_gini' in feature_options else 0)
        with col2:
            y_feature = st.selectbox("Y-axis Feature", feature_options,
                                   index=feature_options.index('modularity') if 'modularity' in feature_options else 1)
        
        # Scatter plot
        fig = px.scatter(
            network_features,
            x=x_feature,
            y=y_feature,
            hover_data=['team_id', 'node_count', 'edge_count'],
            title=f"{y_feature.replace('_', ' ').title()} vs {x_feature.replace('_', ' ').title()}"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"{x_feature.replace('_', ' ').title()} Statistics")
            st.write(network_features[x_feature].describe())
        with col2:
            st.subheader(f"{y_feature.replace('_', ' ').title()} Statistics")
            st.write(network_features[y_feature].describe())
    
    with tab3:
        st.subheader("Playoff Success Analysis")
        
        if analysis_results.statistical_tests:
            st.subheader("Hypothesis Test Results")
            for test_name, result in analysis_results.statistical_tests.items():
                with st.expander(f"Test: {result['hypothesis']}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("P-value", f"{result['p_value']:.4f}")
                        st.metric("Significant?", "Yes" if result['significant'] else "No")
                    with col2:
                        if 'mean_wins' in str(result):
                            st.write("Mean values:")
                            for key, value in result.items():
                                if 'mean' in key:
                                    st.write(f"- {key}: {value:.1f}")
        
        # Model performance
        if analysis_results.regression_results:
            st.subheader("Wins Prediction Model Performance")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("R² Score", f"{analysis_results.regression_results['test_r2']:.3f}")
            with col2:
                st.metric("Cross-val R² (mean)", f"{analysis_results.regression_results['cv_r2_mean']:.3f}")
            with col3:
                st.metric("Cross-val R² (std)", f"{analysis_results.regression_results['cv_r2_std']:.3f}")
        
        if analysis_results.classification_results:
            st.subheader("Playoff Prediction Model Performance")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Cross-val Accuracy (mean)", f"{analysis_results.classification_results['cv_accuracy_mean']:.3f}")
            with col2:
                st.metric("Cross-val Accuracy (std)", f"{analysis_results.classification_results['cv_accuracy_std']:.3f}")
    
    with tab4:
        st.subheader("Individual Team Networks")
        
        # Team selection
        team_options = network_features['team_id'].unique()
        selected_team = st.selectbox("Select Team", team_options)
        
        if st.button("Generate Network Visualization"):
            # Build network for selected team
            analyzer = RosterNetworkAnalyzer()
            visualizer = RosterNetworkVisualizer()
            
            team_salary = salary_data[salary_data['team_id'] == selected_team]
            team_lineup = lineup_data[lineup_data['team_id'] == selected_team]
            
            G = analyzer.build_roster_network(
                team_salary, team_lineup, selected_team, "2023-24"
            )
            
            if G.number_of_nodes() > 0:
                fig = visualizer.plot_network_graph(G, f"Team {selected_team}")
                st.plotly_chart(fig, use_container_width=True)
                
                # Network statistics
                team_features = network_features[network_features['team_id'] == selected_team].iloc[0]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Players", G.number_of_nodes())
                    st.metric("Connections", G.number_of_edges())
                with col2:
                    st.metric("Salary Gini", f"{team_features['salary_gini']:.3f}")
                    st.metric("Network Density", f"{team_features['network_density']:.3f}")
                with col3:
                    st.metric("Modularity", f"{team_features['modularity']:.3f}")
                    st.metric("Clustering", f"{team_features['clustering_coefficient']:.3f}")
            else:
                st.error("No network data available for this team.")
    
    # Feature importance
    if not analysis_results.feature_importance.empty:
        st.markdown('<div class="section-header">Feature Importance</div>', unsafe_allow_html=True)
        
        fig = px.bar(
            analysis_results.feature_importance.head(10),
            x='importance',
            y='feature',
            color='model',
            orientation='h',
            title="Top 10 Most Important Network Features"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
