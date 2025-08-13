"""
Interactive Visualization Module

Creates interactive visualizations for roster network analysis including:
- Network graphs with salary-sized nodes
- Feature correlation plots
- Team comparison dashboards
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
import streamlit as st
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RosterNetworkVisualizer:
    """Creates interactive visualizations for roster network analysis."""
    
    def __init__(self):
        self.color_schemes = {
            'salary': px.colors.sequential.Viridis,
            'impact': px.colors.sequential.RdBu,
            'communities': px.colors.qualitative.Set3
        }
    
    def plot_network_graph(self, 
                          G: nx.Graph,
                          team_name: str = "Team",
                          layout_type: str = "spring",
                          node_size_attr: str = "salary",
                          node_color_attr: str = "bpm") -> go.Figure:
        """
        Create interactive network visualization.
        
        Args:
            G: NetworkX graph
            team_name: Team name for title
            layout_type: Layout algorithm ('spring', 'circular', etc.)
            node_size_attr: Node attribute for sizing
            node_color_attr: Node attribute for coloring
            
        Returns:
            Plotly figure object
        """
        if G.number_of_nodes() == 0:
            fig = go.Figure()
            fig.add_annotation(text="No network data available", 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Get layout positions
        if layout_type == "spring":
            pos = nx.spring_layout(G, weight='weight', k=2, iterations=50)
        elif layout_type == "circular":
            pos = nx.circular_layout(G)
        else:
            pos = nx.spring_layout(G)
        
        # Extract node information
        node_trace = go.Scatter(
            x=[pos[node][0] for node in G.nodes()],
            y=[pos[node][1] for node in G.nodes()],
            mode='markers+text',
            hoverinfo='text',
            text=[G.nodes[node].get('name', str(node)) for node in G.nodes()],
            textposition="middle center",
            textfont=dict(size=10),
            marker=dict(
                size=[max(10, G.nodes[node].get(node_size_attr, 0) / 1000000 * 20) 
                      for node in G.nodes()],  # Scale salary to reasonable sizes
                color=[G.nodes[node].get(node_color_attr, 0) for node in G.nodes()],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=node_color_attr.upper()),
                line=dict(width=2, color='white')
            )
        )
        
        # Create hover text
        hover_text = []
        for node in G.nodes():
            node_data = G.nodes[node]
            hover_info = f"<b>{node_data.get('name', str(node))}</b><br>"
            hover_info += f"Salary: ${node_data.get('salary', 0):,.0f}<br>"
            hover_info += f"BPM: {node_data.get('bpm', 0):.1f}<br>"
            hover_info += f"Minutes: {node_data.get('minutes', 0):.0f}"
            hover_text.append(hover_info)
        
        node_trace.hovertext = hover_text
        
        # Extract edge information
        edge_x = []
        edge_y = []
        edge_weights = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            weight = G[edge[0]][edge[1]]['weight']
            
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_weights.append(weight)
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='rgba(125,125,125,0.5)'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                        title=f'{team_name} Roster Network',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[ dict(
                            text="Node size = Salary | Node color = " + node_color_attr.upper() + " | Edge width = Shared minutes",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002,
                            xanchor='left', yanchor='bottom',
                            font=dict(size=12)
                        )],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                        ))
        
        return fig
    
    def plot_feature_correlations(self, network_features_df: pd.DataFrame, 
                                 outcome_data: Optional[pd.DataFrame] = None) -> go.Figure:
        """
        Create correlation matrix of network features.
        
        Args:
            network_features_df: DataFrame with network features
            outcome_data: Optional DataFrame with team outcomes (wins, playoff success)
            
        Returns:
            Plotly figure with correlation heatmap
        """
        # Select numeric columns for correlation
        numeric_cols = network_features_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['team_id']]
        
        if len(numeric_cols) < 2:
            fig = go.Figure()
            fig.add_annotation(text="Not enough numeric features for correlation",
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Compute correlation matrix
        corr_matrix = network_features_df[numeric_cols].corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size":10},
            hovertemplate='<b>%{x}</b><br><b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Network Features Correlation Matrix',
            xaxis_title='Features',
            yaxis_title='Features',
            width=800,
            height=700
        )
        
        return fig
    
    def plot_team_comparison(self, network_features_df: pd.DataFrame,
                           team_names_map: Optional[Dict[int, str]] = None) -> go.Figure:
        """
        Create interactive comparison of teams across multiple dimensions.
        
        Args:
            network_features_df: DataFrame with network features
            team_names_map: Optional mapping of team_id to team names
            
        Returns:
            Plotly figure with parallel coordinates plot
        """
        df = network_features_df.copy()
        
        # Add team names if available
        if team_names_map:
            df['team_name'] = df['team_id'].map(team_names_map)
        else:
            df['team_name'] = df['team_id'].astype(str)
        
        # Select key features for comparison
        key_features = [
            'salary_gini', 'salary_centralization', 'network_density',
            'clustering_coefficient', 'modularity', 'degree_centralization'
        ]
        
        # Filter to available features
        available_features = [f for f in key_features if f in df.columns]
        
        if not available_features:
            fig = go.Figure()
            fig.add_annotation(text="No key features available for comparison",
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Create parallel coordinates plot
        fig = go.Figure(data=
            go.Parcoords(
                line=dict(color=df.index, colorscale='Viridis'),
                dimensions=[
                    dict(
                        range=[df[feat].min(), df[feat].max()],
                        label=feat.replace('_', ' ').title(),
                        values=df[feat]
                    ) for feat in available_features
                ],
                labelfont=dict(size=12),
                tickfont=dict(size=10)
            )
        )
        
        fig.update_layout(
            title='Team Network Features Comparison',
            font=dict(size=14),
            margin=dict(l=100, r=100, t=50, b=50)
        )
        
        return fig
    
    def plot_salary_vs_performance(self, 
                                 network_features_df: pd.DataFrame,
                                 x_feature: str = "salary_gini",
                                 y_feature: str = "modularity") -> go.Figure:
        """
        Create scatter plot comparing two network features.
        
        Args:
            network_features_df: DataFrame with network features
            x_feature: Feature for x-axis
            y_feature: Feature for y-axis
            
        Returns:
            Plotly scatter plot
        """
        if x_feature not in network_features_df.columns or y_feature not in network_features_df.columns:
            fig = go.Figure()
            fig.add_annotation(text=f"Features {x_feature} or {y_feature} not available",
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        
        fig = px.scatter(
            network_features_df,
            x=x_feature,
            y=y_feature,
            title=f'{y_feature.replace("_", " ").title()} vs {x_feature.replace("_", " ").title()}',
            hover_data=['team_id', 'node_count', 'edge_count'],
            labels={
                x_feature: x_feature.replace('_', ' ').title(),
                y_feature: y_feature.replace('_', ' ').title()
            }
        )
        
        fig.update_traces(marker=dict(size=10, opacity=0.7))
        fig.update_layout(height=500)
        
        return fig
    
    def create_dashboard_layout(self,
                              network_features_df: pd.DataFrame,
                              sample_network: Optional[nx.Graph] = None,
                              team_name: str = "Sample Team") -> Dict[str, go.Figure]:
        """
        Create a collection of figures for a dashboard.
        
        Args:
            network_features_df: DataFrame with network features
            sample_network: Optional sample network to display
            team_name: Name for sample network
            
        Returns:
            Dictionary of figure objects
        """
        figures = {}
        
        # Feature correlations
        figures['correlations'] = self.plot_feature_correlations(network_features_df)
        
        # Team comparison
        figures['team_comparison'] = self.plot_team_comparison(network_features_df)
        
        # Salary vs performance plots
        if 'salary_gini' in network_features_df.columns and 'modularity' in network_features_df.columns:
            figures['salary_modularity'] = self.plot_salary_vs_performance(
                network_features_df, 'salary_gini', 'modularity'
            )
        
        if 'salary_centralization' in network_features_df.columns and 'network_density' in network_features_df.columns:
            figures['centralization_density'] = self.plot_salary_vs_performance(
                network_features_df, 'salary_centralization', 'network_density'
            )
        
        # Sample network visualization
        if sample_network:
            figures['sample_network'] = self.plot_network_graph(sample_network, team_name)
        
        return figures


def create_visualizations(network_features_df: pd.DataFrame,
                         sample_networks: Optional[Dict[str, nx.Graph]] = None) -> Dict[str, go.Figure]:
    """
    Main function to create all visualizations.
    
    Args:
        network_features_df: DataFrame with network analysis results
        sample_networks: Optional dictionary of sample networks to visualize
        
    Returns:
        Dictionary of Plotly figures
    """
    visualizer = RosterNetworkVisualizer()
    
    figures = visualizer.create_dashboard_layout(network_features_df)
    
    # Add sample network visualizations
    if sample_networks:
        for team_name, network in sample_networks.items():
            figures[f'network_{team_name.lower().replace(" ", "_")}'] = \
                visualizer.plot_network_graph(network, team_name)
    
    return figures


if __name__ == "__main__":
    # Test visualization with dummy data
    import random
    
    # Create dummy network features
    teams = list(range(1, 31))  # 30 NBA teams
    
    dummy_features = pd.DataFrame({
        'team_id': teams,
        'season': ['2023-24'] * len(teams),
        'salary_gini': [random.uniform(0.3, 0.7) for _ in teams],
        'salary_centralization': [random.uniform(0.2, 0.5) for _ in teams],
        'network_density': [random.uniform(0.3, 0.8) for _ in teams],
        'modularity': [random.uniform(0.1, 0.6) for _ in teams],
        'clustering_coefficient': [random.uniform(0.4, 0.9) for _ in teams],
        'degree_centralization': [random.uniform(0.2, 0.7) for _ in teams],
        'node_count': [random.randint(12, 18) for _ in teams],
        'edge_count': [random.randint(40, 120) for _ in teams]
    })
    
    # Create visualizations
    figures = create_visualizations(dummy_features)
    
    print(f"Created {len(figures)} visualization figures:")
    for name, fig in figures.items():
        print(f"- {name}")
        
    # Save one figure as HTML for testing
    if 'correlations' in figures:
        figures['correlations'].write_html('test_correlation_plot.html')
        print("Saved correlation plot as test_correlation_plot.html")
