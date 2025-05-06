#!/usr/bin/env python3
# Target Entropy Analysis Script
# Analyzes target entropy across clusters and creates alluvial plots
# to compare target consolidation between PMA and noPMA conditions

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import hdbscan
from scipy.stats import entropy
from tqdm import tqdm
import os
import plotly.graph_objects as go
import plotly.io as pio
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

# Create results directory if it doesn't exist
RESULTS_DIR = "Data/pathway_enrichment"
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_data(condition, conc):
    """Load UMAP projections and extract targets."""
    print(f"Loading data for {condition} {conc}µM...")
    file = f"Data/UMAP_loadings_above1e-4_filtered/{condition}_{conc}uM_UMAP_loadings.csv"
    df = pd.read_csv(file)
    
    # Extract target from the first column (name;target format)
    name_col = df.columns[0]  # Get the first column name
    df['Target'] = df[name_col].apply(lambda x: str(x).split(';')[1].strip() if pd.notna(x) and ';' in str(x) else 'Unknown')
    
    # Get UMAP coordinates
    umap_cols = [col for col in df.columns if col.startswith('UMAP_')]
    
    # Create data with UMAP dimensions
    data = df[umap_cols + ['Target']].copy()
    # Standardize column names
    data.columns = [col.replace('UMAP_', 'UMAP') if 'UMAP_' in col else col for col in data.columns]
    
    print(f"  Loaded {len(data)} compounds with {data['Target'].nunique()} unique targets")
    print(f"  Using {len(umap_cols)} UMAP dimensions")
    
    return data

def cluster_data(data, n_clusters=50, min_cluster_size=10):
    """Perform clustering on UMAP projections."""
    print("Performing clustering...")
    # Get UMAP columns
    umap_cols = [col for col in data.columns if col.startswith('UMAP')]
    
    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    data['KMeans_Cluster'] = kmeans.fit_predict(data[umap_cols])
    
    # HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=5,
        metric='euclidean',
        cluster_selection_method='eom'
    )
    data['HDBSCAN_Cluster'] = clusterer.fit_predict(data[umap_cols[:2]])  # Using first 2 UMAP dimensions
    
    # Print cluster stats
    print(f"  KMeans: {n_clusters} clusters")
    print(f"  HDBSCAN: {len(set(data['HDBSCAN_Cluster'])) - (1 if -1 in data['HDBSCAN_Cluster'].values else 0)} clusters")
    print(f"  HDBSCAN noise points: {(data['HDBSCAN_Cluster'] == -1).sum()}")
    
    return data

def calculate_target_entropy(data, cluster_col):
    """Calculate Shannon entropy for each target's distribution across clusters."""
    print(f"Calculating target entropy using {cluster_col}...")
    
    # Create contingency table
    contingency = pd.crosstab(data['Target'], data[cluster_col])
    
    # Filter to keep only targets with at least 5 compounds
    target_counts = contingency.sum(axis=1)
    valid_targets = target_counts[target_counts >= 5].index
    contingency = contingency.loc[valid_targets]
    
    # Calculate entropy for each target
    target_entropy = {}
    for target in contingency.index:
        # Get the distribution of the target across clusters
        dist = contingency.loc[target].values
        # Normalize to get probabilities
        probs = dist / dist.sum()
        # Calculate Shannon entropy
        target_entropy[target] = entropy(probs)
    
    # Create DataFrame
    entropy_df = pd.DataFrame({
        'Target': list(target_entropy.keys()),
        'Entropy': list(target_entropy.values()),
        'Compound_Count': target_counts[valid_targets].values
    })
    
    # Sort by entropy
    entropy_df = entropy_df.sort_values('Entropy')
    
    print(f"  Calculated entropy for {len(entropy_df)} targets")
    return entropy_df

def calculate_enrichment_scores(data, cluster_col):
    """Calculate enrichment scores for targets in each cluster."""
    # Create contingency table
    contingency = pd.crosstab(data[cluster_col], data['Target'])
    
    # Calculate enrichment scores
    enrichment_scores = pd.DataFrame(index=contingency.index, columns=contingency.columns)
    
    for cluster in contingency.index:
        cluster_size = contingency.loc[cluster].sum()
        total_size = len(data)
        
        for target in contingency.columns:
            target_in_cluster = contingency.loc[cluster, target]
            target_total = contingency[target].sum()
            
            # Calculate enrichment score (observed/expected)
            expected = (target_total * cluster_size) / total_size
            enrichment = target_in_cluster / expected if expected > 0 else 0
            
            enrichment_scores.loc[cluster, target] = enrichment
    
    return enrichment_scores

def create_entropy_comparison_plot(pma_entropy, nopma_entropy, conc):
    """Create comparison plot of target entropy between PMA and noPMA."""
    print("Creating entropy comparison plot...")
    
    # Merge entropy data
    merged = pd.merge(pma_entropy, nopma_entropy, 
                     on='Target', suffixes=('_PMA', '_noPMA'))
    
    # Calculate entropy difference
    merged['Entropy_Diff'] = merged['Entropy_PMA'] - merged['Entropy_noPMA']
    
    # Calculate average compound count
    merged['Avg_Compound_Count'] = (merged['Compound_Count_PMA'] + merged['Compound_Count_noPMA']) / 2
    
    # Create scatter plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(merged['Entropy_PMA'], merged['Entropy_noPMA'], 
                         alpha=0.7, c=merged['Entropy_Diff'], cmap='coolwarm',
                         s=merged['Avg_Compound_Count'] * 2)
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Entropy Difference (PMA - noPMA)')
    
    # Add diagonal line
    max_val = max(merged['Entropy_PMA'].max(), merged['Entropy_noPMA'].max())
    plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
    
    # Label top targets with largest differences
    top_diff = merged.sort_values('Entropy_Diff', ascending=False).head(10)
    bottom_diff = merged.sort_values('Entropy_Diff').head(10)
    
    for _, row in pd.concat([top_diff, bottom_diff]).iterrows():
        plt.annotate(row['Target'], 
                    (row['Entropy_PMA'], row['Entropy_noPMA']),
                    fontsize=8, alpha=0.8)
    
    plt.title(f'Target Entropy Comparison: PMA vs noPMA ({conc}µM)')
    plt.xlabel('Entropy in PMA')
    plt.ylabel('Entropy in noPMA')
    plt.tight_layout()
    
    # Save figure
    plt.savefig(f"{RESULTS_DIR}/entropy_comparison_{conc}uM.png", dpi=300)
    plt.close()
    
    # Return the merged DataFrame for further analysis
    return merged

def create_alluvial_plot(pma_data, nopma_data, merged_entropy, top_n=20, conc='10'):
    """Create alluvial plot showing how clusters change between PMA and noPMA."""
    print("Creating alluvial plot for target consolidation...")
    
    # Get targets with largest absolute entropy difference
    top_targets = merged_entropy.loc[merged_entropy['Avg_Compound_Count'] >= 10] \
                             .sort_values(by='Entropy_Diff', key=abs, ascending=False) \
                             .head(top_n)['Target'].tolist()
    
    # Extract cluster assignments for top targets
    pma_subset = pma_data[pma_data['Target'].isin(top_targets)]
    nopma_subset = nopma_data[nopma_data['Target'].isin(top_targets)]
    
    # Create frequency tables for each condition
    pma_freq = pd.crosstab(pma_subset['Target'], pma_subset['KMeans_Cluster'])
    nopma_freq = pd.crosstab(nopma_subset['Target'], nopma_subset['KMeans_Cluster'])
    
    # Get top 3 clusters for each target in each condition
    pma_top_clusters = {}
    nopma_top_clusters = {}
    
    for target in top_targets:
        if target in pma_freq.index:
            pma_top_clusters[target] = pma_freq.loc[target].nlargest(3).index.tolist()
        else:
            pma_top_clusters[target] = []
            
        if target in nopma_freq.index:
            nopma_top_clusters[target] = nopma_freq.loc[target].nlargest(3).index.tolist()
        else:
            nopma_top_clusters[target] = []
    
    # Create alluvial plot data
    sources = []
    targets = []
    values = []
    labels = []
    colors = []
    
    # Assign colors to targets
    color_palette = sns.color_palette('husl', n_colors=len(top_targets))
    target_colors = {target: f'rgba({int(r*255)},{int(g*255)},{int(b*255)},0.8)' 
                    for target, (r,g,b) in zip(top_targets, color_palette)}
    
    # Create nodes for PMA clusters
    pma_node_start = 0
    pma_nodes = {}
    for target in top_targets:
        for i, cluster in enumerate(pma_top_clusters[target]):
            node_name = f"{target}_PMA_{cluster}"
            node_id = len(labels)
            labels.append(node_name)
            colors.append(target_colors[target])
            pma_nodes[(target, cluster)] = node_id
    
    # Create nodes for noPMA clusters
    nopma_node_start = len(labels)
    nopma_nodes = {}
    for target in top_targets:
        for i, cluster in enumerate(nopma_top_clusters[target]):
            node_name = f"{target}_noPMA_{cluster}"
            node_id = len(labels)
            labels.append(node_name)
            colors.append(target_colors[target])
            nopma_nodes[(target, cluster)] = node_id
    
    # Create links
    for target in top_targets:
        for pma_cluster in pma_top_clusters[target]:
            for nopma_cluster in nopma_top_clusters[target]:
                if target in pma_freq.index and target in nopma_freq.index:
                    pma_count = pma_freq.loc[target, pma_cluster]
                    nopma_count = nopma_freq.loc[target, nopma_cluster]
                    value = min(pma_count, nopma_count)  # Flow volume
                    
                    if value > 0:
                        sources.append(pma_nodes[(target, pma_cluster)])
                        targets.append(nopma_nodes[(target, nopma_cluster)])
                        values.append(value)
    
    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color=colors
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=[f'rgba({int(r*255)},{int(g*255)},{int(b*255)},0.3)' 
                  for r, g, b in color_palette[:len(sources)]]
        )
    )])
    
    fig.update_layout(
        title_text=f"Target Consolidation Across Clusters: PMA vs noPMA ({conc}µM)",
        font_size=10,
        width=1200, 
        height=800
    )
    
    # Save figure
    fig.write_html(f"{RESULTS_DIR}/alluvial_target_consolidation_{conc}uM.html")
    # Also save as PNG for easy viewing
    fig.write_image(f"{RESULTS_DIR}/alluvial_target_consolidation_{conc}uM.png", scale=2)
    
    print(f"  Saved alluvial plot to {RESULTS_DIR}/alluvial_target_consolidation_{conc}uM.html")

def create_heatmap_clustering(pma_data, nopma_data, merged_entropy, conc, cluster_col='KMeans_Cluster'):
    """Create hierarchical clustering heatmap of target enrichment."""
    print("Creating hierarchical clustering heatmap...")
    
    # Get top targets by entropy difference
    top_targets = merged_entropy.sort_values(by='Entropy_Diff', key=abs, ascending=False).head(30)['Target'].tolist()
    
    # Calculate enrichment scores
    pma_enrichment = calculate_enrichment_scores(pma_data, cluster_col)
    nopma_enrichment = calculate_enrichment_scores(nopma_data, cluster_col)
    
    # Filter to top targets
    pma_enrichment_top = pma_enrichment[top_targets].fillna(0)
    nopma_enrichment_top = nopma_enrichment[top_targets].fillna(0)
    
    # Create hierarchical clustering
    # Cluster targets
    pma_target_linkage = sch.linkage(pdist(pma_enrichment_top.T), method='ward')
    nopma_target_linkage = sch.linkage(pdist(nopma_enrichment_top.T), method='ward')
    
    # Get order of targets
    pma_target_order = sch.dendrogram(pma_target_linkage, no_plot=True)['leaves']
    nopma_target_order = sch.dendrogram(nopma_target_linkage, no_plot=True)['leaves']
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot PMA heatmap
    sns.heatmap(pma_enrichment_top.iloc[:, pma_target_order], 
                cmap='YlOrRd', ax=axes[0], xticklabels=True, yticklabels=True)
    axes[0].set_title('PMA Target Enrichment')
    axes[0].set_xticklabels([top_targets[i] for i in pma_target_order], rotation=45, ha='right')
    
    # Plot noPMA heatmap
    sns.heatmap(nopma_enrichment_top.iloc[:, nopma_target_order], 
                cmap='YlOrRd', ax=axes[1], xticklabels=True, yticklabels=True)
    axes[1].set_title('noPMA Target Enrichment')
    axes[1].set_xticklabels([top_targets[i] for i in nopma_target_order], rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/target_enrichment_heatmap_{conc}uM.png", dpi=300)
    plt.close()

def analyze_entropy_patterns(conditions=['PMA', 'noPMA'], concentrations=['1', '10']):
    """Analyze entropy patterns across conditions and concentrations."""
    for conc in concentrations:
        print(f"\n=== Analyzing {conc}µM concentration ===")
        
        # Load data
        pma_data = load_data(conditions[0], conc)
        nopma_data = load_data(conditions[1], conc)
        
        # Cluster data
        pma_clustered = cluster_data(pma_data)
        nopma_clustered = cluster_data(nopma_data)
        
        # Calculate entropy using KMeans clusters
        pma_entropy = calculate_target_entropy(pma_clustered, 'KMeans_Cluster')
        nopma_entropy = calculate_target_entropy(nopma_clustered, 'KMeans_Cluster')
        
        # Save entropy data
        pma_entropy.to_csv(f"{RESULTS_DIR}/{conditions[0]}_{conc}uM_entropy.csv", index=False)
        nopma_entropy.to_csv(f"{RESULTS_DIR}/{conditions[1]}_{conc}uM_entropy.csv", index=False)
        
        # Create comparison plot
        merged_entropy = create_entropy_comparison_plot(pma_entropy, nopma_entropy, conc)
        merged_entropy.to_csv(f"{RESULTS_DIR}/entropy_comparison_{conc}uM.csv", index=False)
        
        # Create alluvial plot
        create_alluvial_plot(pma_clustered, nopma_clustered, merged_entropy, conc=conc)
        
        # Create heatmap clustering
        create_heatmap_clustering(pma_clustered, nopma_clustered, merged_entropy, conc)
        
        print(f"Completed analysis for {conc}µM concentration")

if __name__ == "__main__":
    print("=== Target Entropy Analysis ===")
    analyze_entropy_patterns()
    print("\nAnalysis complete. Results saved to:", RESULTS_DIR) 