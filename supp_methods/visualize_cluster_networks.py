#!/usr/bin/env python3
# Visualize Cluster Networks
# Creates network visualizations from pre-existing HDBSCAN clustering results
# Clusters are arranged in circular layout and connected by edges based on euclidean distances

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics.pairwise import euclidean_distances
import seaborn as sns
from scipy.spatial import distance
import os
from collections import defaultdict
import matplotlib.cm as cm
import matplotlib.patheffects as pe

# Directory for results
RESULTS_DIR = "Data/pathway_enrichment"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Directory with pre-existing cluster data
CLUSTER_DATA_DIR = "Data/network_analysis"

def load_cluster_data(condition, conc):
    """Load pre-existing clustering results."""
    print(f"Loading existing cluster data for {condition} {conc}µM...")
    cluster_file = f"{CLUSTER_DATA_DIR}/cluster_analysis_{condition}_{conc}.csv"
    
    try:
        data = pd.read_csv(cluster_file)
        print(f"  Loaded {len(data)} compounds with clustering data")
        
        # Extract target information - assuming format "name;target"
        if "Target" not in data.columns:
            name_col = data.columns[0]
            data['Target'] = data[name_col].apply(
                lambda x: str(x).split(';')[1].strip() if pd.notna(x) and ';' in str(x) else 'Unknown'
            )
            
            # Drop rows with missing targets
            data = data[data['Target'] != 'Unknown']
        
        # Ensure HDBSCAN_Cluster column exists (third-to-last column)
        if "HDBSCAN_Cluster" not in data.columns:
            print("  HDBSCAN_Cluster column not found directly, using third-to-last column")
            cluster_col = data.columns[-3]
            data['HDBSCAN_Cluster'] = data[cluster_col]
        
        # Get UMAP columns for centroids
        umap_cols = [col for col in data.columns if col.startswith('UMAP')]
        if not umap_cols:
            print("  Warning: No UMAP columns found, searching for alternative names")
            umap_cols = [col for col in data.columns if 'UMAP' in col]
        
        print(f"  Found {len(umap_cols)} UMAP dimensions")
        print(f"  Found {data['HDBSCAN_Cluster'].nunique()} unique clusters")
        print(f"  Found {data['Target'].nunique()} unique targets")
        
        return data
    except FileNotFoundError:
        print(f"  Error: Cluster data file not found at {cluster_file}")
        return None

def load_entropy_data(conc):
    """Load entropy comparison data for the given concentration."""
    file_path = f"{RESULTS_DIR}/entropy_comparison_{conc}uM.csv"
    try:
        data = pd.read_csv(file_path)
        print(f"  Loaded entropy data with {len(data)} targets")
        return data
    except FileNotFoundError:
        print(f"  Warning: Entropy data file not found at {file_path}")
        return None

def get_top_consolidated_targets(entropy_data, top_n=15):
    """Get the top N most consolidated targets for each condition."""
    if entropy_data is None:
        return {}, {}
    
    # Get targets more consolidated in PMA (negative entropy diff)
    pma_targets = entropy_data[entropy_data['Entropy_Diff'] < 0].sort_values('Entropy_Diff').head(top_n)
    
    # Get targets more consolidated in noPMA (positive entropy diff)
    nopma_targets = entropy_data[entropy_data['Entropy_Diff'] > 0].sort_values('Entropy_Diff', ascending=False).head(top_n)
    
    pma_dict = {row['Target']: abs(row['Entropy_Diff']) for _, row in pma_targets.iterrows()}
    nopma_dict = {row['Target']: abs(row['Entropy_Diff']) for _, row in nopma_targets.iterrows()}
    
    # Print top targets
    print("\nTop consolidated targets in PMA:")
    for target, value in sorted(pma_dict.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {target}: {value:.3f}")
    
    print("\nTop consolidated targets in noPMA:")
    for target, value in sorted(nopma_dict.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {target}: {value:.3f}")
    
    return pma_dict, nopma_dict

def calculate_cluster_centroids(data, cluster_col='HDBSCAN_Cluster', exclude_noise=True):
    """Calculate the centroids for each cluster."""
    # Get UMAP columns
    umap_cols = [col for col in data.columns if col.startswith('UMAP')]
    if not umap_cols:
        umap_cols = [col for col in data.columns if 'UMAP' in col]
    
    # Calculate centroids
    centroids = {}
    
    # Filter clusters
    clusters = data[cluster_col].unique()
    if exclude_noise and -1 in clusters:
        clusters = [c for c in clusters if c != -1]
    
    for cluster in clusters:
        cluster_data = data[data[cluster_col] == cluster][umap_cols]
        if len(cluster_data) > 0:  # Ensure cluster has data points
            centroids[cluster] = cluster_data.mean().values
    
    return centroids

def calculate_cluster_distances(centroids):
    """Calculate Euclidean distances between cluster centroids."""
    clusters = list(centroids.keys())
    n_clusters = len(clusters)
    
    # Initialize distance matrix
    dist_matrix = np.zeros((n_clusters, n_clusters))
    
    # Calculate distances
    for i, c1 in enumerate(clusters):
        for j, c2 in enumerate(clusters):
            if i != j:
                dist_matrix[i, j] = distance.euclidean(centroids[c1], centroids[c2])
    
    return dist_matrix, clusters

def calculate_target_enrichment(data, cluster_col='HDBSCAN_Cluster'):
    """Calculate enrichment of targets in each cluster."""
    # Get target distribution across clusters
    target_cluster_counts = defaultdict(lambda: defaultdict(int))
    
    # Skip noise points (-1)
    for _, row in data[data[cluster_col] != -1].iterrows():
        target = row['Target']
        cluster = row[cluster_col]
        target_cluster_counts[target][cluster] += 1
    
    # Calculate total counts
    target_totals = {target: sum(clusters.values()) for target, clusters in target_cluster_counts.items()}
    cluster_totals = defaultdict(int)
    for target, clusters in target_cluster_counts.items():
        for cluster, count in clusters.items():
            cluster_totals[cluster] += count
    
    total_compounds = sum(cluster_totals.values())
    
    # Calculate enrichment for each target in each cluster
    enrichment = defaultdict(lambda: defaultdict(float))
    for target, clusters in target_cluster_counts.items():
        for cluster, count in clusters.items():
            expected = (target_totals[target] * cluster_totals[cluster]) / total_compounds
            if expected > 0:
                enrichment[target][cluster] = count / expected
    
    return dict(enrichment)

def create_network_graph(centroids, dist_matrix, clusters, enrichment, consolidated_targets, condition, threshold=0.65):
    """Create a network graph from cluster centroids and distances."""
    G = nx.Graph()
    
    # Add nodes (clusters)
    for i, cluster in enumerate(clusters):
        G.add_node(cluster)
    
    # Connect each cluster only to its nearest neighbor
    for i, c1 in enumerate(clusters):
        # Find the nearest neighbor (excluding self)
        if len(clusters) > 1:  # Only if we have more than one cluster
            distances = []
            for j, c2 in enumerate(clusters):
                if i != j:  # Skip self
                    distances.append((j, dist_matrix[i, j]))
            
            # Sort by distance and get the nearest neighbor
            nearest_idx, nearest_dist = min(distances, key=lambda x: x[1])
            nearest_cluster = clusters[nearest_idx]
            
            # Add edge to the nearest neighbor
            # Only add if we haven't already (avoid duplicates)
            if not G.has_edge(c1, nearest_cluster):
                # Set weight inversely proportional to distance
                max_dist = np.max(dist_matrix) if np.max(dist_matrix) > 0 else 1
                weight = 1 - (nearest_dist / max_dist)
                G.add_edge(c1, nearest_cluster, weight=weight)
    
    # Find the most enriched consolidated target for each cluster
    cluster_targets = {}
    for cluster in clusters:
        max_enrichment = 0
        max_target = None
        
        for target, cluster_dict in enrichment.items():
            if cluster in cluster_dict and cluster_dict[cluster] > max_enrichment:
                # Only consider targets that are in the consolidated list
                if target in consolidated_targets:
                    max_enrichment = cluster_dict[cluster]
                    max_target = target
        
        cluster_targets[cluster] = (max_target, max_enrichment if max_target else 0)
    
    return G, cluster_targets

def draw_network(G, cluster_targets, consolidated_targets, condition, conc, figsize=(16, 16)):
    """Draw the network graph with an organic layout."""
    plt.figure(figsize=figsize)
    
    # Create an organic layout instead of circular
    # Spring layout tends to spread nodes more organically
    pos = nx.spring_layout(G, k=0.4, iterations=100)
    
    # Get edge weights for edge thickness
    edge_weights = [G[u][v]['weight'] * 5 for u, v in G.edges()]
    
    # Draw the graph
    # First draw edges with transparency based on weight
    for (u, v, d) in G.edges(data=True):
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=d['weight'] * 5, 
                              alpha=d['weight'], edge_color='gray')
    
    # Prepare node colors based on enrichment
    node_colors = []
    node_sizes = []
    
    for node in G.nodes():
        target, enrichment = cluster_targets[node]
        if target:
            # Use color intensity based on consolidation score
            consolidation = consolidated_targets.get(target, 0)
            node_colors.append(consolidation)
            # Size based on enrichment
            node_sizes.append(1000 * min(enrichment, 5))  # Cap size for very high enrichment
        else:
            node_colors.append(0)
            node_sizes.append(300)  # Default size
    
    # Create a colormap that goes from light to dark
    cmap = plt.cm.Reds if condition == 'PMA' else plt.cm.Blues
    
    # Draw nodes
    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                                   node_color=node_colors, cmap=cmap,
                                   alpha=0.8)
    
    # Add a colorbar
    if any(node_colors) and max(node_colors) > 0:
        plt.colorbar(nodes, label='Target Consolidation Score')
    
    # Draw labels for clusters and their top targets
    labels = {}
    for node in G.nodes():
        target, enrichment = cluster_targets[node]
        if target and enrichment > 2:  # Only show enriched targets
            labels[node] = f"Cluster {node}\n{target}\n(E={enrichment:.1f})"
        else:
            labels[node] = f"Cluster {node}"
    
    # Draw labels with white outline for better visibility
    for node, label in labels.items():
        x, y = pos[node]
        plt.text(x, y, label, horizontalalignment='center', verticalalignment='center',
                fontsize=9, fontweight='bold', 
                path_effects=[pe.withStroke(linewidth=3, foreground='white')])
    
    # Set title and remove axis
    plt.title(f"{condition} {conc}µM Cluster Network\nNodes colored by target consolidation, sized by target enrichment", fontsize=16)
    plt.axis('off')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/{condition}_{conc}uM_cluster_network.png", dpi=300)
    plt.close()

def visualize_cluster_networks(pma_data, nopma_data, entropy_data, conc, cluster_col='HDBSCAN_Cluster'):
    """Create network visualizations for PMA and noPMA conditions."""
    # Get top consolidated targets
    pma_targets, nopma_targets = get_top_consolidated_targets(entropy_data)
    
    # Process each condition
    for condition, data, targets in [('PMA', pma_data, pma_targets), ('noPMA', nopma_data, nopma_targets)]:
        if data is None or targets is None:
            print(f"  Cannot create network for {condition} (missing data)")
            continue
        
        print(f"Creating network for {condition} {conc}µM...")
        
        # Calculate centroids
        centroids = calculate_cluster_centroids(data, cluster_col)
        
        if not centroids:
            print(f"  Warning: No valid clusters found for {condition}")
            continue
            
        # Calculate distances
        dist_matrix, clusters = calculate_cluster_distances(centroids)
        
        # Calculate target enrichment
        enrichment = calculate_target_enrichment(data, cluster_col)
        
        # Create network graph
        G, cluster_targets = create_network_graph(centroids, dist_matrix, clusters, enrichment, targets, condition)
        
        # Draw the network
        draw_network(G, cluster_targets, targets, condition, conc)
        
        print(f"  Network saved to {RESULTS_DIR}/{condition}_{conc}uM_cluster_network.png")

def create_combined_network(pma_data, nopma_data, entropy_data, conc, cluster_col='HDBSCAN_Cluster'):
    """Create a combined network showing both PMA and noPMA clusters."""
    if pma_data is None or nopma_data is None or entropy_data is None:
        print("Cannot create combined network due to missing data")
        return
    
    print(f"Creating combined network for PMA and noPMA {conc}µM...")
    
    # Get top consolidated targets
    pma_targets, nopma_targets = get_top_consolidated_targets(entropy_data)
    
    # Calculate centroids for both conditions
    pma_centroids = calculate_cluster_centroids(pma_data, cluster_col)
    nopma_centroids = calculate_cluster_centroids(nopma_data, cluster_col)
    
    if not pma_centroids or not nopma_centroids:
        print("  Warning: No valid clusters found for one or both conditions")
        return
        
    # Combine centroids with prefixes to distinguish conditions
    combined_centroids = {}
    for cluster, centroid in pma_centroids.items():
        combined_centroids[f"PMA_{cluster}"] = centroid
    for cluster, centroid in nopma_centroids.items():
        combined_centroids[f"noPMA_{cluster}"] = centroid
    
    # Calculate distances for combined centroids
    combined_dist_matrix, combined_clusters = calculate_cluster_distances(combined_centroids)
    
    # Calculate target enrichment for both conditions
    pma_enrichment = calculate_target_enrichment(pma_data, cluster_col)
    nopma_enrichment = calculate_target_enrichment(nopma_data, cluster_col)
    
    # Combine enrichment with prefixes
    combined_enrichment = {}
    for target, clusters in pma_enrichment.items():
        combined_enrichment[target] = {f"PMA_{cluster}": value for cluster, value in clusters.items()}
        
    for target, clusters in nopma_enrichment.items():
        if target not in combined_enrichment:
            combined_enrichment[target] = {}
        combined_enrichment[target].update({f"noPMA_{cluster}": value for cluster, value in clusters.items()})
    
    # Combine consolidated targets
    combined_targets = {}
    for target, value in pma_targets.items():
        combined_targets[target] = value
    for target, value in nopma_targets.items():
        combined_targets[target] = value
    
    # Create network graph
    G, cluster_targets = create_network_graph(combined_centroids, combined_dist_matrix, combined_clusters, 
                                             combined_enrichment, combined_targets, "Combined", threshold=0.5)
    
    # Prepare for drawing with different colors for PMA and noPMA
    plt.figure(figsize=(18, 18))
    
    # Create an organic layout
    pos = nx.spring_layout(G, k=0.4, iterations=100)
    
    # Draw edges
    for (u, v, d) in G.edges(data=True):
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=d['weight'] * 3, 
                              alpha=0.8, edge_color='gray')
    
    # Prepare node colors based on condition and enrichment
    pma_nodes = [node for node in G.nodes() if node.startswith('PMA_')]
    nopma_nodes = [node for node in G.nodes() if node.startswith('noPMA_')]
    
    # Get sizes based on enrichment
    pma_sizes = []
    nopma_sizes = []
    
    for node in pma_nodes:
        target, enrichment = cluster_targets[node]
        pma_sizes.append(800 * min(enrichment, 5) if enrichment > 0 else 300)
        
    for node in nopma_nodes:
        target, enrichment = cluster_targets[node]
        nopma_sizes.append(800 * min(enrichment, 5) if enrichment > 0 else 300)
    
    # Draw nodes with different colors for PMA and noPMA
    nx.draw_networkx_nodes(G, pos, nodelist=pma_nodes, node_size=pma_sizes, 
                          node_color='red', alpha=0.7)
    nx.draw_networkx_nodes(G, pos, nodelist=nopma_nodes, node_size=nopma_sizes, 
                          node_color='blue', alpha=0.7)
    
    # Draw labels
    for node in G.nodes():
        target, enrichment = cluster_targets[node]
        cluster_num = node.split('_')[1]
        label_text = f"{node.split('_')[0]} {cluster_num}"
        
        if target and enrichment > 2:  # Only show highly enriched targets
            label_text += f"\n{target}\n(E={enrichment:.1f})"
            
        x, y = pos[node]
        plt.text(x, y, label_text, horizontalalignment='center', verticalalignment='center',
                fontsize=9, fontweight='bold', 
                path_effects=[pe.withStroke(linewidth=3, foreground='white')])
    
    # Add legend
    plt.plot([], [], 'ro', markersize=10, label='PMA Clusters')
    plt.plot([], [], 'bo', markersize=10, label='noPMA Clusters')
    plt.legend(fontsize=12)
    
    # Set title and remove axis
    plt.title(f"Combined PMA and noPMA {conc}µM Cluster Network\nNodes sized by target enrichment", fontsize=16)
    plt.axis('off')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/combined_{conc}uM_cluster_network.png", dpi=300)
    plt.close()
    
    print(f"  Combined network saved to {RESULTS_DIR}/combined_{conc}uM_cluster_network.png")

def analyze_target_distribution(data, conc, condition, cluster_col='HDBSCAN_Cluster'):
    """Analyze and visualize target distribution across clusters."""
    print(f"Analyzing target distribution for {condition} {conc}µM...")
    
    # Skip if no data
    if data is None:
        print("  No data available for analysis")
        return
        
    # Get unique targets
    targets = data['Target'].unique()
    print(f"  Found {len(targets)} unique targets")
    
    # Get cluster distribution for each target
    target_clusters = {}
    for target in targets:
        target_data = data[data['Target'] == target]
        if len(target_data) >= 3:  # Only consider targets with at least 3 compounds
            cluster_counts = target_data[cluster_col].value_counts()
            # Filter out noise cluster (-1)
            if -1 in cluster_counts:
                cluster_counts = cluster_counts.drop(-1)
            
            if not cluster_counts.empty:
                target_clusters[target] = cluster_counts
    
    # Prepare data for visualization
    if target_clusters:
        # Get all clusters
        all_clusters = set()
        for counts in target_clusters.values():
            all_clusters.update(counts.index)
        all_clusters = sorted(list(all_clusters))
        
        # Create heatmap data
        heatmap_data = np.zeros((len(target_clusters), len(all_clusters)))
        
        for i, (target, counts) in enumerate(target_clusters.items()):
            for cluster in counts.index:
                if cluster in all_clusters:
                    j = all_clusters.index(cluster)
                    heatmap_data[i, j] = counts[cluster]
        
        # Create DataFrame for heatmap
        heatmap_df = pd.DataFrame(heatmap_data, 
                                  index=list(target_clusters.keys()),
                                  columns=[f"Cluster {c}" for c in all_clusters])
        
        # Normalize by row (targets)
        row_sums = heatmap_df.sum(axis=1)
        heatmap_df_norm = heatmap_df.div(row_sums, axis=0)
        
        # Sort targets by their spread (entropy)
        target_entropy = {}
        for target, row in heatmap_df_norm.iterrows():
            # Calculate Shannon entropy
            probs = row[row > 0].values
            if len(probs) > 0:
                entropy_val = -np.sum(probs * np.log2(probs))
                target_entropy[target] = entropy_val
        
        # Sort the targets by entropy
        sorted_targets = sorted(target_entropy.items(), key=lambda x: x[1])
        sorted_target_names = [t[0] for t in sorted_targets]
        
        # Limit to top 30 targets for readability
        if len(sorted_target_names) > 30:
            # Get top 15 most concentrated (low entropy) and top 15 most distributed (high entropy)
            sorted_target_names = sorted_target_names[:15] + sorted_target_names[-15:]
        
        # Plot heatmap
        plt.figure(figsize=(12, max(8, len(sorted_target_names) * 0.3)))
        
        # Reorder heatmap data by entropy
        heatmap_df_sorted = heatmap_df_norm.loc[sorted_target_names]
        
        # Create heatmap
        sns.heatmap(heatmap_df_sorted, cmap="YlGnBu", annot=False, 
                   linewidths=0.5, cbar_kws={'label': 'Proportion of compounds'})
        
        plt.title(f"Target Distribution Across Clusters - {condition} {conc}µM")
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/{condition}_{conc}uM_target_distribution.png", dpi=300)
        plt.close()
        
        print(f"  Target distribution heatmap saved to {RESULTS_DIR}/{condition}_{conc}uM_target_distribution.png")
    else:
        print("  No suitable targets found for heatmap visualization")

def create_spoke_layout(G, center=None):
    """Create a spoke/radial layout where nodes are arranged around a central node."""
    pos = {}
    nodes = list(G.nodes())
    
    # If center is not specified, use the first node
    if center is None and nodes:
        center = nodes[0]
    
    # Place the center node
    pos[center] = np.array([0, 0])
    
    # Place other nodes in a circle around the center
    other_nodes = [n for n in nodes if n != center]
    if other_nodes:
        # Number of nodes
        n = len(other_nodes)
        # Radius
        radius = 1.0
        # Angle between nodes
        angle = 2 * np.pi / n
        
        for i, node in enumerate(other_nodes):
            theta = i * angle
            pos[node] = np.array([radius * np.cos(theta), radius * np.sin(theta)])
    
    return pos

def create_integrated_visualization(pma_data, nopma_data, entropy_data, conc, 
                                   cluster_col='HDBSCAN_Cluster', highlight_clusters=None):
    """Create an integrated visualization showing UMAP scatter, zoomed regions, and cluster networks."""
    if pma_data is None or nopma_data is None:
        print("Cannot create visualization due to missing data")
        return
    
    print(f"Creating integrated visualization for {conc}µM concentration...")
    
    # Get top consolidated targets
    pma_targets, nopma_targets = get_top_consolidated_targets(entropy_data) if entropy_data is not None else ({}, {})
    
    # Set up the figure with a complex grid layout
    fig = plt.figure(figsize=(20, 16))
    
    # Create GridSpec for layout control - allow for up to 3 highlight clusters
    gs = fig.add_gridspec(3, 3, width_ratios=[2, 1, 1], height_ratios=[1, 1, 1])
    
    # Main scatter plot (UMAP visualization) in the left panel - spans all rows
    ax_main = fig.add_subplot(gs[:, 0])
    
    # Get UMAP columns for visualization
    umap_cols = [col for col in pma_data.columns if col.startswith('UMAP')]
    if not umap_cols or len(umap_cols) < 2:
        umap_cols = [col for col in pma_data.columns if 'UMAP' in col]
        if len(umap_cols) < 2:
            print("  Error: Need at least 2 UMAP dimensions for visualization")
            return
    
    # Use the first two UMAP dimensions for the scatter plot
    umap1, umap2 = umap_cols[0], umap_cols[1]
    
    # Plot PMA data points with colors based on cluster
    scatter_pma = ax_main.scatter(pma_data[umap1], pma_data[umap2], 
                                 c=pma_data[cluster_col], cmap='tab20', 
                                 alpha=0.6, s=15, marker='o')
    
    # Set axis labels
    ax_main.set_xlabel(umap1, fontsize=12)
    ax_main.set_ylabel(umap2, fontsize=12)
    ax_main.set_title(f'UMAP Visualization - PMA {conc}µM', fontsize=14)
    
    # Identify clusters to highlight based on target enrichment
    if highlight_clusters is None:
        # Find clusters with high target enrichment
        enrichment = calculate_target_enrichment(pma_data, cluster_col)
        
        # Identify top enriched clusters for consolidated targets
        top_clusters = []
        for target, score in pma_targets.items():
            if target in enrichment:
                # Get clusters with this target and sort by enrichment
                target_clusters = sorted(enrichment[target].items(), key=lambda x: x[1], reverse=True)
                if target_clusters:
                    top_clusters.append((target_clusters[0][0], target, target_clusters[0][1], score))
        
        # Sort by target consolidation * cluster enrichment
        top_clusters.sort(key=lambda x: x[2] * x[3], reverse=True)
        
        # Take top 3 clusters (or fewer if not enough)
        highlight_clusters = [c[0] for c in top_clusters[:min(3, len(top_clusters))]]
    
    if not highlight_clusters:
        # If still no clusters to highlight, take the top 3 largest
        cluster_sizes = pma_data[cluster_col].value_counts()
        # Filter out noise cluster (-1)
        if -1 in cluster_sizes:
            cluster_sizes = cluster_sizes.drop(-1)
        highlight_clusters = cluster_sizes.head(3).index.tolist()
    
    # Color the highlight clusters differently in the main plot
    for cluster_id in highlight_clusters:
        cluster_mask = pma_data[cluster_col] == cluster_id
        ax_main.scatter(pma_data.loc[cluster_mask, umap1], pma_data.loc[cluster_mask, umap2],
                      c='purple', alpha=0.8, s=20, marker='o', edgecolors='black')
    
    # Zoom boxes and network visualizations for highlighted clusters
    for i, cluster_id in enumerate(highlight_clusters[:3]):  # Limit to 3 clusters for clarity
        # Get cluster data
        cluster_data = pma_data[pma_data[cluster_col] == cluster_id]
        if len(cluster_data) == 0:
            continue
        
        # Calculate cluster bounding box with margin
        x_min, x_max = cluster_data[umap1].min(), cluster_data[umap1].max()
        y_min, y_max = cluster_data[umap2].min(), cluster_data[umap2].max()
        
        # Add margin
        margin_x = (x_max - x_min) * 0.2
        margin_y = (y_max - y_min) * 0.2
        x_min -= margin_x
        x_max += margin_x
        y_min -= margin_y
        y_max += margin_y
        
        # Draw zoom box on main plot
        rect = plt.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, 
                            fill=False, edgecolor='purple', linestyle='--', linewidth=2)
        ax_main.add_patch(rect)
        
        # Add zoom inset in the middle column
        ax_zoom = fig.add_subplot(gs[i, 1])
        ax_zoom.scatter(cluster_data[umap1], cluster_data[umap2], 
                       c='purple', alpha=0.8, s=30)
        ax_zoom.set_xlim(x_min, x_max)
        ax_zoom.set_ylim(y_min, y_max)
        ax_zoom.set_title(f'Cluster {cluster_id}', fontsize=12)
        ax_zoom.set_xticks([])  # Remove x ticks for cleaner look
        ax_zoom.set_yticks([])  # Remove y ticks for cleaner look
        
        # Find enriched targets for this cluster
        target_enrichment = {}
        if enrichment:
            for target in pma_targets.keys():
                if target in enrichment and cluster_id in enrichment[target]:
                    target_enrichment[target] = enrichment[target][cluster_id]
        
        # Sort targets by enrichment
        sorted_targets = sorted(target_enrichment.items(), key=lambda x: x[1], reverse=True)
        
        # Get top target for this cluster
        top_target, top_enrichment_score = sorted_targets[0] if sorted_targets else (None, 0)
        
        # Plot network visualization for this cluster
        ax_network = fig.add_subplot(gs[i, 2])
        
        # Create a network just for this cluster
        compounds = cluster_data.index.tolist()
        target_compounds = []
        
        if top_target:
            target_compounds = cluster_data[cluster_data['Target'] == top_target].index.tolist()
        
        G = nx.Graph()
        
        # Add center node representing the cluster/target
        G.add_node('center', size=2000, color='purple')
        
        # Add compound nodes - limit to a reasonable number for clarity
        max_display = min(40, len(compounds))
        for idx, compound in enumerate(compounds):
            if idx >= max_display:
                break
            
            # Add compounds with specific layout considerations
            is_target_compound = compound in target_compounds
            
            G.add_node(compound, 
                      size=300 if is_target_compound else 200, 
                      color='yellow' if is_target_compound else 'pink')
            G.add_edge('center', compound)
        
        # Create a spoke layout
        pos = create_spoke_layout(G, 'center')
        
        # Draw the network
        node_colors = [G.nodes[n].get('color', 'pink') for n in G.nodes()]
        node_sizes = [G.nodes[n].get('size', 300) for n in G.nodes()]
        
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, 
                              alpha=0.8, ax=ax_network)
        nx.draw_networkx_edges(G, pos, width=0.8, alpha=0.5, ax=ax_network)
        
        # Add annotation for the target
        if top_target:
            ax_network.text(0, -1.4, f"{top_target}\n(E={top_enrichment_score:.1f})", 
                          ha='center', va='center', fontsize=12, fontweight='bold',
                          bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
        
        # Set title
        if top_target:
            ax_network.set_title(f'{top_target} Modulators Cluster', fontsize=12)
        else:
            ax_network.set_title(f'Cluster {cluster_id} Network', fontsize=12)
        
        ax_network.axis('off')
        
        # Draw an arrow connecting zoom to network
        # Use annotation arrow instead of Arrow patch for better reliability
        arrow_props = dict(arrowstyle='->', color='purple', linewidth=2)
        ax_zoom.annotate('', xy=(1.1, 0.5), xytext=(1, 0.5), 
                       xycoords='axes fraction', textcoords='axes fraction',
                       arrowprops=arrow_props)
    
    # Add a legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', 
                  markersize=10, label='Compound in highlighted cluster'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', 
                  markersize=10, label=f'Compound with specific target')
    ]
    ax_main.legend(handles=legend_elements, loc='lower right')
    
    # Add a common title for the entire figure
    fig.suptitle(f'Cluster Network Analysis - PMA {conc}µM', fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
    
    # Save figure
    plt.savefig(f"{RESULTS_DIR}/integrated_visualization_{conc}uM.png", dpi=300)
    plt.close()
    
    print(f"  Integrated visualization saved to {RESULTS_DIR}/integrated_visualization_{conc}uM.png")

def visualize_individual_clusters(data, conc, condition, cluster_col='HDBSCAN_Cluster', top_n=3):
    """Create detailed visualizations of individual clusters with compounds in a circular layout."""
    print(f"Creating detailed cluster visualizations for {condition} {conc}µM...")
    
    if data is None:
        print("  Cannot create cluster visualizations due to missing data")
        return
        
    # Calculate target enrichment
    enrichment = calculate_target_enrichment(data, cluster_col)
    
    # Get the top N clusters by size (excluding noise)
    cluster_sizes = data[data[cluster_col] != -1][cluster_col].value_counts()
    top_clusters = cluster_sizes.head(top_n).index.tolist()
    
    # For each top cluster, create a detailed visualization
    for cluster_id in top_clusters:
        # Get cluster data
        cluster_data = data[data[cluster_col] == cluster_id]
        
        # Skip if no data
        if len(cluster_data) == 0:
            continue
            
        print(f"  Creating visualization for Cluster {cluster_id} ({len(cluster_data)} compounds)")
        
        # Find the top 3 enriched targets for this cluster
        top_targets = []
        for target in data['Target'].unique():
            if target in enrichment and cluster_id in enrichment[target]:
                top_targets.append((target, enrichment[target][cluster_id]))
                
        # Sort by enrichment
        top_targets = sorted(top_targets, key=lambda x: x[1], reverse=True)[:3]
        
        # Create a figure
        plt.figure(figsize=(14, 14))
        
        # Create a graph for this cluster
        G = nx.Graph()
        
        # Add center node (cluster)
        G.add_node('cluster', size=3000, color='purple')
        
        # Add compound nodes
        # Limit to a reasonable number for clarity
        max_compounds = min(100, len(cluster_data))
        compounds_to_show = cluster_data.head(max_compounds).index.tolist()
        
        # Group compounds by target
        compounds_by_target = {}
        for target, _ in top_targets:
            target_compounds = cluster_data[cluster_data['Target'] == target].index.tolist()
            compounds_by_target[target] = [c for c in target_compounds if c in compounds_to_show]
            
        # Add nodes for each compound
        for compound in compounds_to_show:
            # Check if this compound has one of the top targets
            compound_target = None
            for target, target_compounds in compounds_by_target.items():
                if compound in target_compounds:
                    compound_target = target
                    break
                    
            # Set node attributes based on target
            if compound_target:
                # Use different colors for different targets
                target_idx = [t[0] for t in top_targets].index(compound_target)
                colors = ['yellow', 'orange', 'lime']
                G.add_node(compound, size=300, color=colors[target_idx], target=compound_target)
            else:
                G.add_node(compound, size=200, color='lightblue', target=None)
                
            # Add edge to cluster
            G.add_edge('cluster', compound)
        
        # Create a circular layout for compounds
        # First position the cluster in the center
        pos = {'cluster': np.array([0, 0])}
        
        # Position compounds in a circle
        other_nodes = [n for n in G.nodes() if n != 'cluster']
        n_compounds = len(other_nodes)
        
        # Group compounds by target for better visualization
        arranged_nodes = []
        
        # First add compounds with top targets (grouped by target)
        for target, _ in top_targets:
            target_compounds = [n for n in other_nodes if G.nodes[n].get('target') == target]
            arranged_nodes.extend(target_compounds)
            
        # Then add remaining compounds
        remaining = [n for n in other_nodes if n not in arranged_nodes]
        arranged_nodes.extend(remaining)
        
        # Position all compounds in a circle
        radius = 1.0
        for i, node in enumerate(arranged_nodes):
            angle = 2 * np.pi * i / n_compounds
            pos[node] = np.array([radius * np.cos(angle), radius * np.sin(angle)])
        
        # Draw edges first
        nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5)
        
        # Draw nodes with colors based on target
        node_colors = [G.nodes[n].get('color', 'lightblue') for n in G.nodes()]
        node_sizes = [G.nodes[n].get('size', 100) for n in G.nodes()]
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
        
        # Add target labels
        # Position them in a circle slightly larger than the compounds
        legend_radius = 1.2
        for i, (target, enrichment_score) in enumerate(top_targets):
            # Position around the circle, evenly spaced
            angle = 2 * np.pi * i / len(top_targets)
            x = legend_radius * np.cos(angle)
            y = legend_radius * np.sin(angle)
            
            # Colors matching the nodes
            colors = ['yellow', 'orange', 'lime']
            
            plt.text(x, y, f"{target}\n(E={enrichment_score:.1f})", 
                    ha='center', va='center', fontsize=12, fontweight='bold',
                    bbox=dict(facecolor=colors[i], alpha=0.3, boxstyle='round,pad=0.5'))
        
        # Add title
        plt.title(f"{condition} {conc}µM - Cluster {cluster_id} ({len(cluster_data)} compounds)", fontsize=16)
        plt.axis('off')
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/{condition}_{conc}uM_cluster_{cluster_id}_detail.png", dpi=300)
        plt.close()
        
        print(f"    Visualization saved to {RESULTS_DIR}/{condition}_{conc}uM_cluster_{cluster_id}_detail.png")
    
    print(f"  Completed detailed cluster visualizations for {condition} {conc}µM")

def main():
    """Main function to create cluster network visualizations."""
    concentrations = ['1', '10']
    
    for conc in concentrations:
        print(f"\n=== Creating visualizations for {conc}µM concentration ===")
        
        # Load existing cluster data for both conditions
        pma_data = load_cluster_data('PMA', conc)
        nopma_data = load_cluster_data('noPMA', conc)
        
        # Load entropy data
        entropy_data = load_entropy_data(conc)
        
        if pma_data is None or nopma_data is None:
            print("Warning: Cluster data missing, visualizations cannot be created.")
            continue
        
        # Create the integrated visualization (similar to the reference figure)
        create_integrated_visualization(pma_data, nopma_data, entropy_data, conc)
        
        # Create individual and combined cluster networks
        visualize_cluster_networks(pma_data, nopma_data, entropy_data, conc)
        create_combined_network(pma_data, nopma_data, entropy_data, conc)
        
        # Create detailed visualizations for top clusters
        visualize_individual_clusters(pma_data, conc, 'PMA')
        visualize_individual_clusters(nopma_data, conc, 'noPMA')
        
        # Analyze target distribution
        analyze_target_distribution(pma_data, conc, 'PMA')
        analyze_target_distribution(nopma_data, conc, 'noPMA')
        
        print(f"Completed visualization for {conc}µM concentration")

if __name__ == "__main__":
    main() 