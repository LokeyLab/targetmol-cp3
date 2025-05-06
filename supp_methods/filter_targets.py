import pandas as pd
import os
from collections import Counter

def filter_targets():
    # Define paths
    input_folder = "/Users/rslokey/Documents/Projects/CP3-0/Pathway_enrichment/Data/UMAP_loadings_above1e-4"
    output_folder = "/Users/rslokey/Documents/Projects/CP3-0/Pathway_enrichment/Data/UMAP_loadings_above1e-4_filtered"
    
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Define conditions and concentrations
    conditions = ["PMA", "noPMA"]
    concentrations = ["1", "10"]
    
    # Process each file
    for condition in conditions:
        for conc in concentrations:
            print(f"\nProcessing {condition} {conc}µM...")
            
            # Read the input file
            input_file = os.path.join(input_folder, f"{condition}_{conc}uM_UMAP_loadings.csv")
            df = pd.read_csv(input_file)
            
            # Extract targets and count them
            targets = [row.split("; ")[1] for row in df["Name_AL"]]
            target_counts = Counter(targets)
            
            # Filter targets with 3 or more members and exclude "Others"
            valid_targets = {target: count for target, count in target_counts.items() 
                           if count >= 3 and target != "Others"}
            
            # Filter the dataframe
            filtered_df = df[df["Name_AL"].apply(lambda x: x.split("; ")[1] in valid_targets)]
            
            # Save filtered data
            output_file = os.path.join(output_folder, f"{condition}_{conc}uM_UMAP_loadings.csv")
            filtered_df.to_csv(output_file, index=False)
            
            # Print statistics
            print(f"Original compounds: {len(df)}")
            print(f"Filtered compounds: {len(filtered_df)}")
            print(f"Original targets: {len(target_counts)}")
            print(f"Filtered targets: {len(valid_targets)}")
            
            # Save target distribution for filtered data
            results_file = os.path.join(output_folder, f"{condition}_{conc}uM_target_distribution.txt")
            with open(results_file, 'w') as f:
                f.write(f"Target Distribution Analysis - {condition} {conc}µM (Filtered)\n")
                f.write("==================================================\n\n")
                f.write(f"Total unique targets: {len(valid_targets)}\n")
                f.write(f"Total compounds: {len(filtered_df)}\n\n")
                
                # Sort targets by count in descending order
                for target, count in sorted(valid_targets.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"{target}: {count} compounds\n")

if __name__ == "__main__":
    filter_targets() 