import os
import pandas as pd
from collections import defaultdict

def get_parquet_files(folder_path):
    """
    Retrieves all parquet files from the given folder path and its nested subfolders.
    """
    parquet_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.parquet'):
                parquet_files.append(os.path.join(root, file))
    print(f"Found {len(parquet_files)} parquet files.")
    return parquet_files

def process_files(input_folder, output_folder, target_inchikeys, metadata_filter):
    """
    Processes parquet files, extracts target compounds, and saves them incrementally.
    """
    parquet_files = get_parquet_files(input_folder)
    compounds_by_source = defaultdict(set)
    total_files = len(parquet_files)

    for index, file in enumerate(parquet_files, 1):
        print(f"Processing file {index}/{total_files}: {file}")
        try:
            df = pd.read_parquet(file)
            source = os.path.basename(os.path.dirname(os.path.dirname(file)))
            
            df = pd.merge(df, metadata_filter, how='inner', on=['Metadata_Source', 'Metadata_Plate', 'Metadata_Well'])
            
            filtered_df = df[df['Metadata_InChIKey'].isin(target_inchikeys)]
            
            if not filtered_df.empty:
                for inchikey, group in filtered_df.groupby('Metadata_InChIKey'):
                    output_path = os.path.join(output_folder, source, f"{inchikey}.parquet")
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    
                    if os.path.exists(output_path):
                        existing_df = pd.read_parquet(output_path)
                        combined_df = pd.concat([existing_df, group])
                        combined_df.to_parquet(output_path, index=False)
                        print(f"Appended data for {inchikey} in {source}")
                    else:
                        group.to_parquet(output_path, index=False)
                        print(f"Created new file for {inchikey} in {source}")
                    
                    compounds_by_source[source].add(inchikey)
        
        except Exception as e:
            print(f"Error processing file {file}: {e}")

    # Print summary
    print("\nProcessing complete. Summary:")
    for source, compounds in compounds_by_source.items():
        print(f"{source}: Processed {len(compounds)} unique compounds")

if __name__ == "__main__":
    input_folder = '../data/'
    output_folder = '../filtered_data/'
    
    # Read target InChIKeys
    with open("../data/metadata/poscon_cp_InChIKey.txt", 'r') as file:
        poscon_cp_InChIKey = [line.strip() for line in file.readlines()]
    with open("../data/metadata/known_moa_InChIKey.txt", 'r') as file:
        known_moa_InChIKey = [line.strip() for line in file.readlines()]
    
    target_inchikeys = list(set(poscon_cp_InChIKey + known_moa_InChIKey))
    print(f"Loaded {len(target_inchikeys)} target InChIKeys")
    
    metadata_filter = pd.read_csv('../data/metadata/metadata_compound_of_interest.csv', dtype=str)
    
    process_files(input_folder, output_folder, target_inchikeys, metadata_filter)