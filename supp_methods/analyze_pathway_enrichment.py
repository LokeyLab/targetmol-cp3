#!/usr/bin/env python3
# Pathway Enrichment Analysis for Entropy Differences
# Analyzes pathway representation among targets with high entropy differences

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import os

# Directory for results
RESULTS_DIR = "Data/pathway_enrichment"
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_entropy_data(concentration):
    """Load entropy comparison data for the given concentration."""
    file_path = f"{RESULTS_DIR}/entropy_comparison_{concentration}uM.csv"
    data = pd.read_csv(file_path)
    # Add a column for absolute entropy difference
    data['Abs_Entropy_Diff'] = data['Entropy_Diff'].abs()
    return data

def categorize_targets(targets):
    """Categorize targets into specific pathway groups based on their names."""
    # Define pathway categories and related keywords with more specific subcategories
    pathway_categories = {
        # Neurotransmitter systems
        'Serotonergic': ['5-HT', 'Serotonin', '5-hydroxytryptamine'],
        'Dopaminergic': ['Dopamine', 'D1', 'D2', 'D3', 'D4', 'D5'],
        'Adrenergic': ['Adrenergic', 'Alpha1', 'Alpha2', 'Beta1', 'Beta2', 'Beta3', 'Norepinephrine', 'Epinephrine'],
        'Cholinergic': ['AChR', 'Cholinergic', 'Acetylcholine', 'Muscarinic', 'Nicotinic'],
        'Glutamatergic': ['Glutamate', 'NMDA', 'AMPA', 'Kainate', 'mGluR'],
        'GABAergic': ['GABA', 'GABA-A', 'GABA-B', 'Benzodiazepine'],
        'Other Neurotransmitter': ['Glycine', 'Histamine', 'Melatonin', 'Orexin', 'Cannabinoid', 'Opioid'],
        
        # Ion channels and transporters
        'Calcium Channels': ['Calcium_Channel', 'Ca2+', 'VGCC', 'L-type', 'N-type', 'P/Q-type', 'T-type', 'R-type'],
        'Potassium Channels': ['Potassium_Channel', 'K+', 'KATP', 'Kv', 'Kir', 'BK', 'SK', 'HCN'],
        'Sodium Channels': ['Sodium_Channel', 'Na+', 'Nav', 'ENaC'],
        'Other Ion Channels': ['Ion_Channel', 'TRP', 'Chloride', 'HCN', 'Cl-', 'CaCC'],
        'Transporters': ['Transporter', 'SLC', 'ABC', 'Pump', 'Exchanger', 'Na+/K+_ATPase', 'SERT', 'DAT', 'NET'],
        
        # Kinase subfamilies
        'MAPK Pathway': ['MAPK', 'ERK', 'JNK', 'p38', 'MEK', 'RAF'],
        'Cell Cycle Kinases': ['CDK', 'PLK', 'Aurora', 'Checkpoint', 'Cell_Cycle'],
        'PI3K/AKT/mTOR': ['PI3K', 'AKT', 'mTOR', 'PDK1', 'TSC'],
        'Tyrosine Kinases': ['Tyrosine_Kinase', 'EGFR', 'VEGFR', 'PDGFR', 'FGFR', 'TK', 'JAK', 'SRC'],
        'Other Kinases': ['Kinase', 'PKA', 'PKC', 'PKG', 'GSK', 'AMPK', 'CK', 'IKK', 'ROCK', 'LIM_Kinase'],
        
        # Nuclear receptor subfamilies
        'Steroid Receptors': ['Estrogen', 'Androgen', 'Glucocorticoid', 'Progesterone', 'Mineralocorticoid'],
        'Non-Steroid Receptors': ['Thyroid', 'Retinoic_Acid', 'Vitamin_D', 'RAR', 'RXR', 'VDR'],
        'Metabolism Receptors': ['PPAR', 'FXR', 'LXR', 'PXR', 'CAR', 'HNF'],
        'Cytokine Receptors': ['Cytokine_Receptor', 'IL_Receptor', 'TNF_Receptor', 'Interferon_Receptor'],
        'Growth Factor Receptors': ['Growth_Factor_Receptor', 'NGF', 'EGF', 'FGF', 'IGF', 'VEGF', 'PDGF'],
        
        # GPCRs
        'GPCR Class A': ['GPCR', 'G-Protein', 'Rhodopsin', 'Adrenergic', 'Serotonin', 'Dopamine', 'Histamine', 'Muscarinic'],
        'GPCR Class B': ['Secretin', 'Glucagon', 'Calcitonin', 'PTH', 'CRF', 'VIP', 'PACAP'],
        'GPCR Class C': ['Metabotropic', 'Glutamate', 'mGluR', 'GABA-B', 'Taste', 'Calcium-sensing'],
        'Other GPCRs': ['Frizzled', 'Smoothened', 'Prostanoid', 'Cannabinoid', 'Adenosine', 'Opioid'],
        
        # Enzymes by type
        'Proteases': ['Protease', 'Peptidase', 'Proteinase', 'Caspase', 'Cathepsin', 'MMP', 'ADAM', 'Elastase', 'Trypsin',
                    'Chymotrypsin', 'Serine_Protease', 'Metalloprotease', 'Cysteine_Protease', 'Aspartic_Protease'],
        'Phosphatases': ['Phosphatase', 'PTP', 'Ser/Thr_Phosphatase', 'PP1', 'PP2A', 'PP2B', 'PP2C', 'PTEN'],
        'Oxidoreductases': ['Oxidoreductase', 'Dehydrogenase', 'Reductase', 'Oxidase', 'Oxygenase', 'CYP', 'Cytochrome',
                          'NADPH', 'Dihydroorotate_Dehydrogenase', 'Xanthine_Oxidase', 'SOD', 'Catalase', 'Peroxidase'],
        'Transferases': ['Transferase', 'Kinase', 'Methyltransferase', 'Acetyltransferase', 'Transaminase', 'COMT'],
        'Hydrolases': ['Hydrolase', 'Esterase', 'Lipase', 'Amylase', 'Nuclease', 'Phosphodiesterase', 'PDE'],
        'Ligases/Synthetases': ['Ligase', 'Synthetase', 'Synthase', 'Carboxylase', 'DNA_Ligase', 'RNA_Ligase', 'Ubiquitin_Ligase'],
        'Isomerases': ['Isomerase', 'Racemase', 'Epimerase', 'Topoisomerase', 'cis-trans_Isomerase'],
        
        # Metabolic pathways
        'Glycolysis/Gluconeogenesis': ['Glycolysis', 'Gluconeogenesis', 'PFK', 'Pyruvate', 'HK', 'Hexokinase', 'GAPDH', 'PK', 'Enolase'],
        'TCA Cycle': ['TCA', 'Krebs', 'Citrate', 'Succinate', 'SDH', 'IDH', 'Fumarate', 'Malate', 'Oxaloacetate'],
        'Lipid Metabolism': ['Fatty_Acid', 'Lipid', 'Cholesterol', 'Lipoprotein', 'HMG-CoA', 'Statin', 'Sterol', 'Phospholipid', 
                           'Acyl-CoA', 'Carnitine', 'Acetyl-CoA_Carboxylase', 'Fatty_Acid_Synthase'],
        'Nucleotide Metabolism': ['Nucleotide', 'Purine', 'Pyrimidine', 'Ribonucleotide_Reductase', 'Thymidylate', 'IMPDH', 
                                'PRPP', 'IMP', 'GMP', 'AMP', 'CMP', 'UMP', 'Thymidine'],
        'Amino Acid Metabolism': ['Amino_Acid', 'Transaminase', 'Protease', 'Aminopeptidase', 'Decarboxylase', 'Glutaminase', 
                                'Tyrosine', 'Tryptophan', 'Phenylalanine', 'DOPA'],
        'Energy Metabolism': ['ATP', 'Mitochondria', 'Electron_Transport', 'OXPHOS', 'F1F0-ATPase', 'Complex I', 'Complex II', 
                            'Complex III', 'Complex IV', 'Complex V', 'UCP', 'Uncoupler'],
        
        # Cytoskeleton and cellular processes
        'Cytoskeleton': ['Actin', 'Tubulin', 'Myosin', 'Kinesin', 'Dynein', 'Microtubule', 'Cytoskeleton', 'Spectrin', 
                       'Microtubule_Associated', 'Microfilament', 'Intermediate_Filament'],
        'Cell Cycle/Division': ['Cell_Cycle', 'Mitosis', 'Meiosis', 'CDC', 'Cyclin', 'CDK', 'Checkpoint', 'Centrosome', 
                              'Spindle', 'APC/C', 'Anaphase', 'Metaphase', 'Prophase', 'Telophase'],
        'DNA Processes': ['DNA', 'Polymerase', 'Helicase', 'Topoisomerase', 'DNA_repair', 'Nuclease', 'Recombinase', 
                         'Telomerase', 'Gyrase', 'Replication', 'Replication_Fork'],
        'RNA Processes': ['RNA', 'RNA_Polymerase', 'Splicing', 'Spliceosome', 'rRNA', 'mRNA', 'tRNA', 'Translation', 
                         'Transcription', 'Ribonuclease', 'RNase', 'RNA_processing'],
        
        # Cell death and survival
        'Apoptosis': ['Apoptosis', 'Caspase', 'Bcl', 'Death', 'Survival', 'p53', 'Bax', 'Bak', 'Fas', 'TNF', 'TRAIL'],
        'Autophagy': ['Autophagy', 'ATG', 'ULK', 'mTOR', 'Beclin', 'LC3', 'Lysosome', 'Phagosome', 'Autolysosome'],
        
        # Immune and inflammatory systems
        'Innate Immunity': ['Toll', 'TLR', 'NOD', 'Inflammasome', 'NF-kB', 'MyD88', 'IKK', 'Macrophage', 'Neutrophil', 'NK_cell'],
        'Cytokines/Chemokines': ['Cytokine', 'Chemokine', 'Interleukin', 'IL-', 'Interferon', 'IFN', 'TNF', 'TGF', 'CCL', 'CXCL'],
        'Adaptive Immunity': ['T_cell', 'B_cell', 'TCR', 'BCR', 'Antibody', 'MHC', 'HLA', 'Antigen', 'CD3', 'CD4', 'CD8', 'CD19'],
        'Complement System': ['Complement', 'C1', 'C2', 'C3', 'C4', 'C5', 'MAC', 'Properdin', 'Factor_B', 'Factor_D'],
        
        # Miscellaneous important categories
        'Epigenetic Regulation': ['Epigenetic', 'Methylation', 'Acetylation', 'Histone', 'HDAC', 'HAT', 'DNMT', 'Chromatin'],
        'Microbiome/Pathogen': ['Bacterial', 'Viral', 'Fungal', 'Antibiotic', 'Antimicrobial', 'Antibacterial', 'Antiviral', 
                               'Antifungal', 'Microbiome', 'Microbiota', 'Probiotic'],
        'Redox Systems': ['Redox', 'Glutathione', 'GSH', 'GSSG', 'Thioredoxin', 'ROS', 'Oxidative_Stress', 'Antioxidant', 
                        'Free_Radical', 'Reactive_Oxygen', 'Reactive_Nitrogen', 'NOS', 'Superoxide', 'Glutathione_Metabolism'],
    }
    
    # Categorize each target
    target_categories = {}
    for target in targets:
        target_str = str(target)
        assigned = False
        
        # First try exact match with category keywords
        for category, keywords in pathway_categories.items():
            for keyword in keywords:
                if (keyword.lower() in target_str.lower() or 
                    re.search(r'\b' + keyword.lower() + r'\b', target_str.lower())):
                    target_categories[target] = category
                    assigned = True
                    break
            if assigned:
                break
        
        # If no match found, check for broader patterns - add more rules for specific cases
        if not assigned:
            # Special case pattern matching
            if re.search(r'kinase|ase', target_str.lower()):
                if 'phosphatase' in target_str.lower():
                    target_categories[target] = 'Phosphatases'
                elif re.search(r'dehydrogenase|oxidase|reductase', target_str.lower()):
                    target_categories[target] = 'Oxidoreductases'
                elif re.search(r'synthetase|synthase|ligase', target_str.lower()):
                    target_categories[target] = 'Ligases/Synthetases'
                elif re.search(r'transferase|transaminase', target_str.lower()):
                    target_categories[target] = 'Transferases'
                elif re.search(r'hydrolase|esterase|lipase', target_str.lower()):
                    target_categories[target] = 'Hydrolases'
                elif re.search(r'protease|peptidase|proteinase', target_str.lower()):
                    target_categories[target] = 'Proteases'
                elif re.search(r'isomerase|mutase|epimerase', target_str.lower()):
                    target_categories[target] = 'Isomerases'
                else:
                    target_categories[target] = 'Other Kinases'
            elif 'receptor' in target_str.lower():
                if re.search(r'steroid|androgen|estrogen|glucocorticoid|progesterone', target_str.lower()):
                    target_categories[target] = 'Steroid Receptors'
                elif re.search(r'growth|egf|fgf|pdgf|igf|ngf|vegf', target_str.lower()):
                    target_categories[target] = 'Growth Factor Receptors'
                elif re.search(r'cytokine|interleukin|il-|ifn|tnf', target_str.lower()):
                    target_categories[target] = 'Cytokine Receptors'
                elif re.search(r'gpcr|g-protein|adrenergic|serotonin|dopamine', target_str.lower()):
                    target_categories[target] = 'GPCR Class A'
                else:
                    target_categories[target] = 'Other GPCRs'
            elif re.search(r'channel|transporter|exchanger|pump', target_str.lower()):
                if 'calcium' in target_str.lower() or 'ca2+' in target_str.lower():
                    target_categories[target] = 'Calcium Channels'
                elif 'potassium' in target_str.lower() or 'k+' in target_str.lower():
                    target_categories[target] = 'Potassium Channels'
                elif 'sodium' in target_str.lower() or 'na+' in target_str.lower():
                    target_categories[target] = 'Sodium Channels'
                else:
                    target_categories[target] = 'Transporters'
            elif re.search(r'dna|rna|transcription|replication', target_str.lower()):
                if 'rna' in target_str.lower():
                    target_categories[target] = 'RNA Processes'
                else:
                    target_categories[target] = 'DNA Processes'
            else:
                target_categories[target] = 'Other'
        
    return target_categories

def analyze_pathway_enrichment(data, top_n=50):
    """Analyze pathway enrichment among targets with high entropy differences."""
    # Get targets more consolidated in PMA (negative entropy diff)
    pma_consolidated = data[data['Entropy_Diff'] < 0].sort_values('Entropy_Diff').head(top_n)
    
    # Get targets more consolidated in noPMA (positive entropy diff)
    nopma_consolidated = data[data['Entropy_Diff'] > 0].sort_values('Entropy_Diff', ascending=False).head(top_n)
    
    # Categorize targets
    pma_categories = categorize_targets(pma_consolidated['Target'])
    nopma_categories = categorize_targets(nopma_consolidated['Target'])
    
    # Count categories
    pma_category_counts = Counter(pma_categories.values())
    nopma_category_counts = Counter(nopma_categories.values())
    
    # Convert to DataFrames
    categories = list(set(list(pma_category_counts.keys()) + list(nopma_category_counts.keys())))
    
    pma_counts = [pma_category_counts.get(cat, 0) for cat in categories]
    nopma_counts = [nopma_category_counts.get(cat, 0) for cat in categories]
    
    # Calculate percentages
    pma_percentages = [count / sum(pma_counts) * 100 for count in pma_counts]
    nopma_percentages = [count / sum(nopma_counts) * 100 for count in nopma_counts]
    
    # Calculate enrichment (ratio of percentages)
    enrichment = []
    for i in range(len(categories)):
        if nopma_percentages[i] == 0:
            enrichment.append(float('inf'))
        else:
            enrichment.append(pma_percentages[i] / nopma_percentages[i])
    
    # Create DataFrame
    pathway_df = pd.DataFrame({
        'Category': categories,
        'PMA_Count': pma_counts,
        'PMA_Percentage': pma_percentages,
        'noPMA_Count': nopma_counts,
        'noPMA_Percentage': nopma_percentages,
        'PMA_vs_noPMA_Enrichment': enrichment
    })
    
    # Sort by enrichment
    pathway_df = pathway_df.sort_values('PMA_vs_noPMA_Enrichment', ascending=False)
    
    return pathway_df, pma_consolidated, nopma_consolidated

def plot_pathway_distribution(pathway_df, concentration):
    """Plot pathway distribution comparison."""
    # Filter out categories with few counts
    min_total = 3
    filtered_df = pathway_df[(pathway_df['PMA_Count'] + pathway_df['noPMA_Count']) >= min_total]
    
    # Handle case when filtered_df is empty
    if filtered_df.empty:
        print(f"Warning: No pathways with sufficient counts for {concentration}µM")
        filtered_df['Log2_Enrichment'] = []
        return filtered_df
    
    # Sort by enrichment
    filtered_df = filtered_df.sort_values('PMA_vs_noPMA_Enrichment', ascending=True)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    
    # Plot counts
    counts_df = pd.DataFrame({
        'PMA': filtered_df['PMA_Count'],
        'noPMA': filtered_df['noPMA_Count']
    }, index=filtered_df['Category'])
    
    counts_df.plot(kind='barh', ax=axes[0])
    axes[0].set_title('Target Counts by Pathway')
    axes[0].set_xlabel('Number of Targets')
    axes[0].set_ylabel('Pathway Category')
    
    # Plot percentages
    percentage_df = pd.DataFrame({
        'PMA': filtered_df['PMA_Percentage'],
        'noPMA': filtered_df['noPMA_Percentage']
    }, index=filtered_df['Category'])
    
    percentage_df.plot(kind='barh', ax=axes[1])
    axes[1].set_title('Target Percentage by Pathway')
    axes[1].set_xlabel('Percentage of Targets (%)')
    axes[1].set_ylabel('Pathway Category')
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/pathway_distribution_{concentration}uM.png", dpi=300)
    plt.close()
    
    # Create enrichment plot
    plt.figure(figsize=(10, 8))
    
    # Filter out infinite values and ensure data exists
    plot_df = filtered_df[filtered_df['PMA_vs_noPMA_Enrichment'] != float('inf')].copy()
    
    if not plot_df.empty:
        # Calculate log2 enrichment
        plot_df['Log2_Enrichment'] = np.log2(plot_df['PMA_vs_noPMA_Enrichment'])
        
        # Sort by log2 enrichment
        plot_df = plot_df.sort_values('Log2_Enrichment')
        
        # Create bar plot
        colors = ['red' if x < 0 else 'blue' for x in plot_df['Log2_Enrichment']]
        plt.barh(plot_df['Category'], plot_df['Log2_Enrichment'], color=colors)
        
        plt.axvline(0, color='black', linestyle='--')
        plt.title(f'Pathway Enrichment: PMA vs noPMA ({concentration}µM)')
        plt.xlabel('Log2(PMA/noPMA Enrichment)')
        plt.ylabel('Pathway Category')
        
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/pathway_enrichment_{concentration}uM.png", dpi=300)
    else:
        print(f"Warning: No valid enrichment data for creating the enrichment plot for {concentration}µM")
    
    plt.close()
    
    # Add Log2_Enrichment to filtered_df if it doesn't exist
    if 'Log2_Enrichment' not in filtered_df.columns:
        filtered_df['Log2_Enrichment'] = filtered_df['PMA_vs_noPMA_Enrichment'].apply(
            lambda x: np.log2(x) if x != float('inf') and x > 0 else float('nan'))
    
    return filtered_df

def list_top_targets(pma_consolidated, nopma_consolidated, category_filter=None):
    """List top targets for each condition, optionally filtered by category."""
    # Categorize targets
    pma_categories = categorize_targets(pma_consolidated['Target'])
    nopma_categories = categorize_targets(nopma_consolidated['Target'])
    
    # Add category to DataFrames
    pma_with_category = pma_consolidated.copy()
    pma_with_category['Category'] = pma_with_category['Target'].map(pma_categories)
    
    nopma_with_category = nopma_consolidated.copy()
    nopma_with_category['Category'] = nopma_with_category['Target'].map(nopma_categories)
    
    # Filter by category if specified
    if category_filter:
        pma_with_category = pma_with_category[pma_with_category['Category'] == category_filter]
        nopma_with_category = nopma_with_category[nopma_with_category['Category'] == category_filter]
    
    # Select columns to display
    display_cols = ['Target', 'Category', 'Entropy_PMA', 'Entropy_noPMA', 'Entropy_Diff', 'Avg_Compound_Count']
    
    return pma_with_category[display_cols], nopma_with_category[display_cols]

def plot_target_entropy_by_category(data, concentration):
    """Plot target entropy by category."""
    # Categorize all targets
    data['Category'] = data['Target'].map(categorize_targets(data['Target']))
    
    # Group by category and calculate mean entropy difference
    category_stats = data.groupby('Category').agg({
        'Entropy_Diff': ['mean', 'std', 'count'],
        'Avg_Compound_Count': 'mean'
    })
    
    # Flatten column names
    category_stats.columns = ['Mean_Entropy_Diff', 'Std_Entropy_Diff', 'Count', 'Avg_Compound_Count']
    
    # Filter to categories with at least 5 targets
    category_stats = category_stats[category_stats['Count'] >= 5]
    
    # Sort by mean entropy difference
    category_stats = category_stats.sort_values('Mean_Entropy_Diff')
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    # Generate colors based on sign of mean entropy difference
    colors = ['red' if x < 0 else 'blue' for x in category_stats['Mean_Entropy_Diff']]
    
    # Create bar plot
    plt.barh(category_stats.index, category_stats['Mean_Entropy_Diff'], 
             xerr=category_stats['Std_Entropy_Diff'] / np.sqrt(category_stats['Count']),
             color=colors, alpha=0.7)
    
    plt.axvline(0, color='black', linestyle='--')
    plt.title(f'Mean Entropy Difference by Pathway Category ({concentration}µM)')
    plt.xlabel('Mean Entropy Difference (PMA - noPMA)')
    plt.ylabel('Pathway Category')
    
    # Add counts to labels
    labels = [f"{idx} (n={int(category_stats.loc[idx, 'Count'])})" for idx in category_stats.index]
    plt.yticks(range(len(labels)), labels)
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/category_entropy_diff_{concentration}uM.png", dpi=300)
    plt.close()
    
    return category_stats

def main():
    """Main function to analyze pathway enrichment."""
    concentrations = ['1', '10']
    
    for conc in concentrations:
        print(f"\n=== Analyzing pathway enrichment for {conc}µM concentration ===")
        
        # Load data
        data = load_entropy_data(conc)
        
        # Analyze pathway enrichment
        pathway_df, pma_consolidated, nopma_consolidated = analyze_pathway_enrichment(data)
        
        # Plot pathway distribution
        filtered_pathway_df = plot_pathway_distribution(pathway_df, conc)
        
        # Plot target entropy by category
        category_stats = plot_target_entropy_by_category(data, conc)
        
        # Save results
        pathway_df.to_csv(f"{RESULTS_DIR}/pathway_enrichment_{conc}uM.csv", index=False)
        category_stats.to_csv(f"{RESULTS_DIR}/category_entropy_stats_{conc}uM.csv")
        
        # Check if we have valid data for reporting results
        if filtered_pathway_df.empty or 'Log2_Enrichment' not in filtered_pathway_df.columns:
            print("No valid pathway enrichment results to report.")
            continue
            
        # Handle NaN values in Log2_Enrichment
        filtered_pathway_df_valid = filtered_pathway_df.dropna(subset=['Log2_Enrichment'])
        
        if filtered_pathway_df_valid.empty:
            print("No valid enrichment scores to report.")
            continue
        
        # Print top pathways
        print("\nTop pathways more consolidated in PMA:")
        try:
            top_pma = filtered_pathway_df_valid.sort_values('Log2_Enrichment', ascending=False).head(5)
            for _, row in top_pma.iterrows():
                print(f"  {row['Category']}: {row['PMA_Count']} targets, {row['PMA_Percentage']:.1f}% " +
                    f"(vs {row['noPMA_Count']} targets, {row['noPMA_Percentage']:.1f}% in noPMA), " +
                    f"Enrichment: {row['PMA_vs_noPMA_Enrichment']:.2f}x")
        except (KeyError, ValueError) as e:
            print(f"Error displaying top PMA pathways: {e}")
        
        print("\nTop pathways more consolidated in noPMA:")
        try:
            top_nopma = filtered_pathway_df_valid.sort_values('Log2_Enrichment').head(5)
            for _, row in top_nopma.iterrows():
                print(f"  {row['Category']}: {row['noPMA_Count']} targets, {row['noPMA_Percentage']:.1f}% " +
                    f"(vs {row['PMA_Count']} targets, {row['PMA_Percentage']:.1f}% in PMA), " +
                    f"Enrichment: {1/row['PMA_vs_noPMA_Enrichment']:.2f}x")
        except (KeyError, ValueError) as e:
            print(f"Error displaying top noPMA pathways: {e}")
        
        # List example targets for top categories
        try:
            if len(top_pma) > 0:
                top_pma_category = top_pma.iloc[0]['Category']
                pma_targets, _ = list_top_targets(pma_consolidated, nopma_consolidated, top_pma_category)
                if not pma_targets.empty:
                    print(f"\nExample targets for top PMA category '{top_pma_category}':")
                    for _, row in pma_targets.head(5).iterrows():
                        print(f"  {row['Target']}: PMA entropy {row['Entropy_PMA']:.2f}, noPMA entropy {row['Entropy_noPMA']:.2f}, diff {row['Entropy_Diff']:.2f}")
        except (IndexError, KeyError) as e:
            print(f"Error displaying PMA targets: {e}")
        
        try:
            if len(top_nopma) > 0:
                top_nopma_category = top_nopma.iloc[0]['Category']
                _, nopma_targets = list_top_targets(pma_consolidated, nopma_consolidated, top_nopma_category)
                if not nopma_targets.empty:
                    print(f"\nExample targets for top noPMA category '{top_nopma_category}':")
                    for _, row in nopma_targets.head(5).iterrows():
                        print(f"  {row['Target']}: PMA entropy {row['Entropy_PMA']:.2f}, noPMA entropy {row['Entropy_noPMA']:.2f}, diff {row['Entropy_Diff']:.2f}")
        except (IndexError, KeyError) as e:
            print(f"Error displaying noPMA targets: {e}")

if __name__ == "__main__":
    main() 