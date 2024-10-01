import os
import pandas as pd
from collections import defaultdict
import csv
import logging
from datetime import datetime

# Set up logging
log_filename = f"process_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_filename),
                        logging.StreamHandler()
                    ])

def get_parquet_files(folder_path):
    """
    Retrieves all parquet files from the given folder path and its nested subfolders,
    excluding folders containing 'ORF' or 'CRISPR'.
    """
    parquet_files = []
    for root, dirs, files in os.walk(folder_path):
        if 'ORF' in root or 'CRISPR' in root:
            logging.info(f"Skipping folder: {root}")
            continue
        for file in files:
            if file.endswith('.parquet'):
                parquet_files.append(os.path.join(root, file))
    logging.info(f"Found {len(parquet_files)} parquet files.")
    return parquet_files

def process_files(input_folder, output_folder, target_inchikeys, meta_compound):
    """
    Processes parquet files, extracts target compounds, and saves them by source, batch, and plate.
    """
    parquet_files = get_parquet_files(input_folder)
    log_data = []
    total_files = len(parquet_files)
    processed_files = 0
    skipped_files = 0
    error_files = 0

    for index, file in enumerate(parquet_files, 1):
        logging.info(f"Processing file {index}/{total_files}: {file}")
        try:
            df = pd.read_parquet(file)
            source = os.path.basename(os.path.dirname(os.path.dirname(file)))
            
            # Merge with meta_compound
            df = pd.merge(df, meta_compound, how='inner', on=['Metadata_Source', 'Metadata_Plate', 'Metadata_Well'])
            
            filtered_df = df[df['Metadata_InChIKey'].isin(target_inchikeys)]
            
            if not filtered_df.empty:
                for (batch, plate), group in filtered_df.groupby(['Metadata_Batch', 'Metadata_Plate']):
                    output_path = os.path.join(output_folder, source, batch, f"{plate}.parquet")
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    
                    group.to_parquet(output_path, index=False)
                    logging.info(f"Created/Updated file for {source}, batch {batch}, plate {plate}")
                    
                    log_data.append({
                        'Source': source,
                        'Batch': batch,
                        'Plate': plate,
                        'Total_Compounds': group['Metadata_InChIKey'].nunique(),
                    })
                processed_files += 1
            else:
                logging.info(f"No matching compounds found in {file}. Skipping.")
                skipped_files += 1
        
        except Exception as e:
            logging.error(f"Error processing file {file}: {e}")
            error_files += 1

    # Generate log file
    log_df = pd.DataFrame(log_data)
    log_path = os.path.join(output_folder, 'data_log.csv')
    log_df.to_csv(log_path, index=False)
    logging.info(f"Data log file generated: {log_path}")

    # Log summary
    logging.info(f"Processing complete. Summary:")
    logging.info(f"Total files: {total_files}")
    logging.info(f"Processed files: {processed_files}")
    logging.info(f"Skipped files: {skipped_files}")
    logging.info(f"Error files: {error_files}")

if __name__ == "__main__":
    input_folder = '../data_parquet/'
    output_folder = '../data_cp/'
    
    logging.info("Starting the process...")

    # Read target InChIKeys
    with open("../data/metadata/poscon_cp_JUMP.txt", 'r') as file:
        target_inchikeys = [line.strip() for line in file.readlines()]
    logging.info(f"Loaded {len(target_inchikeys)} target InChIKeys")
    
    # Load metadata
    logging.info("Loading metadata...")
    plates = pd.read_csv("../data/metadata/plate.csv.gz")
    wells = pd.read_csv("../data/metadata/well.csv.gz")
    compound = pd.read_csv("../data/metadata/compound.csv.gz")
    meta = pd.merge(plates.dropna(), wells.dropna(), how='inner', on=['Metadata_Source', 'Metadata_Plate'])
    meta_compound = pd.merge(meta.dropna(), compound.dropna(), how='inner', on=['Metadata_JCP2022'])
    logging.info(f"Metadata loaded. Shape of meta_compound: {meta_compound.shape}")
    
    process_files(input_folder, output_folder, target_inchikeys, meta_compound)

    logging.info("Process completed. Check the log file for details.")