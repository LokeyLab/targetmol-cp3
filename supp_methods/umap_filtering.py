# @command run_umap_filtering
# @desc Filter UMAP data based on key files and create new CSV files
import pandas as pd
import os

def run_umap_filtering():
    # Define paths
    data_folder = "Data"
    output_folder = "/Users/rslokey/Documents/Projects/CP3-0/Pathway_enrichment/Data/UMAP_loadings_above1e-4"
    
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Define file names
    data_files = {
        "PMA": "FinalClean_TM_1+10_PMA_dropna_UMAP_longtrain_with_metadata_reordered.csv",
        "noPMA": "FinalClean_TM_1+10_noPMA_dropna_UMAP_longtrain_with_metadata_reordered.csv"
    }

    key_files = {
        ("PMA", "10"): "PMA_10uM_above_1e-4_thresholdCPscores.csv",
        ("PMA", "1"): "PMA_1uM_above_1e-4_thresholdCPscores.csv",
        ("noPMA", "10"): "noPMA_10uM_above_1e-4_thresholdCPscores.csv",
        ("noPMA", "1"): "noPMA_1uM_above_1e-4_thresholdCPscores.csv"
    }

    # Process each condition (PMA/noPMA) and concentration (1µM/10µM)
    for condition in ["PMA", "noPMA"]:
        print(f"\nProcessing {condition} condition...")
        # Read the full data file
        data_df = pd.read_csv(os.path.join(data_folder, data_files[condition]))
        
        # Print unique concentration values
        print("\nUnique concentration values in data file:")
        print(data_df["metadata;Concentration"].unique())
        
        # Get UMAP column names
        umap_cols = [col for col in data_df.columns if col.startswith("UMAP_")]
        
        # Print first few rows of Name_AL column
        print("\nFirst few Name_AL values:")
        print(data_df["Name_AL"].head())
        
        for conc in ["1", "10"]:
            print(f"\nProcessing {conc}µM concentration...")
            # Read the corresponding key file
            key_df = pd.read_csv(os.path.join(data_folder, key_files[(condition, conc)]))
            
            # Create compound-target pairs from key file using 3rd column (Name) and 11th column (AL_CONSOLIDATED)
            key_pairs = key_df.apply(lambda row: f"{row.iloc[2]}; {row.iloc[10]}", axis=1).tolist()
            
            # Print some debug info
            print(f"Found {len(key_pairs)} compounds in key file")
            if len(key_pairs) > 0:
                print(f"Example key pair: {key_pairs[0]}")
            
            # Filter data for current concentration and matching compounds
            filtered_df = data_df[
                (data_df["metadata;Concentration"] == f"{conc}uM") &
                (data_df["Name_AL"].isin(key_pairs))
            ]
            
            # Print some debug info about the filtering
            print(f"Total rows in data file: {len(data_df)}")
            print(f"Rows matching concentration {conc}µM: {len(data_df[data_df['metadata;Concentration'] == f'{conc}uM'])}")
            print(f"Rows matching both concentration and key pairs: {len(filtered_df)}")
            
            if len(filtered_df) > 0:
                print(f"Example Name_AL from filtered data: {filtered_df['Name_AL'].iloc[0]}")
            
            # Select only the required columns
            output_df = filtered_df[["Name_AL"] + umap_cols]
            
            # Save to output file
            output_filename = f"{condition}_{conc}uM_UMAP_loadings.csv"
            output_path = os.path.join(output_folder, output_filename)
            output_df.to_csv(output_path, index=False)
            
            print(f"Created {output_filename} with {len(output_df)} compounds")

    print("\nProcessing complete!")

if __name__ == "__main__":
    run_umap_filtering() 