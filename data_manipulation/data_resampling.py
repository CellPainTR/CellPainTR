import os
import glob
import random
import argparse
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import logging
import concurrent.futures
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def transform_file(file, output_folder, rows_per_file=384):
    df = pq.read_table(file).to_pandas()
    base_name = os.path.splitext(os.path.basename(file))[0]
    file_counter = 0
    for i in range(0, len(df), rows_per_file):
        chunk = df.iloc[i:i+rows_per_file]
        output_file = os.path.join(output_folder, f"{base_name}_chunk{file_counter}.parquet")
        table = pa.Table.from_pandas(chunk)
        pq.write_table(table, output_file, compression='snappy')
        file_counter += 1
    return len(df)

def transform_files(input_folder, output_folder, file_names, rows_per_file=384, process_all=False):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    all_files = glob.glob(f"{input_folder}/**/*.parquet", recursive=True)
    if process_all:
        filtered_files = all_files
    else:
        filtered_files = [f for f in all_files if os.path.splitext(os.path.basename(f))[0] in file_names]
    
    logging.info(f"Processing {len(filtered_files)} Parquet files")
    
    total_rows = 0
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(transform_file, file, output_folder, rows_per_file) for file in filtered_files]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(filtered_files), desc="Transforming files"):
            total_rows += future.result()
    
    logging.info(f"Total rows processed: {total_rows}")

def shuffle_folder(folder, x, rows_per_file=384):
    all_files = glob.glob(f"{folder}/*.parquet")
    random.shuffle(all_files)
    
    logging.info(f"Shuffling folder containing {len(all_files)} files")
    
    temp_folder = f"{folder}_temp"
    os.makedirs(temp_folder, exist_ok=True)
    
    total_rows = 0
    output_files = []
    
    for i in tqdm(range(0, len(all_files), x), desc="Processing batches"):
        batch_files = all_files[i:i+x]
        tables = [pq.read_table(f) for f in batch_files]
        combined_table = pa.concat_tables(tables)
        combined_df = combined_table.to_pandas()
        combined_df = combined_df.sample(frac=1).reset_index(drop=True)
        
        total_rows += len(combined_df)
        
        # Write shuffled data back to files
        for j in range(0, len(combined_df), rows_per_file):
            chunk = combined_df.iloc[j:j+rows_per_file]
            output_file = os.path.join(temp_folder, f"shuffled_file{len(output_files)}.parquet")
            table = pa.Table.from_pandas(chunk)
            pq.write_table(table, output_file, compression='snappy')
            output_files.append(output_file)
    
    # Verify row count
    shuffled_rows = sum(pq.read_metadata(f).num_rows for f in output_files)
    original_rows = sum(pq.read_metadata(f).num_rows for f in all_files)
    
    if shuffled_rows != original_rows:
        raise ValueError(f"Row count mismatch: original {original_rows}, shuffled {shuffled_rows}")
    
    # Replace original files with shuffled files
    for f in all_files:
        os.remove(f)
    
    for f in output_files:
        shutil.move(f, folder)
    
    os.rmdir(temp_folder)
    
    logging.info(f"Folder shuffling completed. Total rows: {total_rows}")

def main(input_folder, output_folder, x, k, file_names, skip_transform=False, process_all=False):
    if not skip_transform:
        logging.info("Step 1: Transforming files")
        transform_files(input_folder, output_folder, file_names, process_all=process_all)
    else:
        logging.info("Skipping transformation step")
    
    logging.info(f"Step 2: Shuffling files (x={x}, k={k})")
    for i in range(k):
        logging.info(f"Starting shuffle iteration {i+1}/{k}")
        shuffle_folder(output_folder, x)
        logging.info(f"Completed shuffle iteration {i+1}/{k}")
    
    logging.info("Process completed successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform and shuffle Parquet files.")
    parser.add_argument("input_folder", help="Input folder containing Parquet files")
    parser.add_argument("output_folder", help="Output folder for transformed and shuffled files")
    parser.add_argument("x", type=int, help="Number of files to load at once for shuffling")
    parser.add_argument("k", type=int, help="Number of shuffling iterations")
    parser.add_argument("file_names", nargs='*', help="List of file names to process (without extension)")
    parser.add_argument("--skip-transform", action="store_true", help="Skip the transformation step")
    parser.add_argument("--process-all", action="store_true", help="Process all Parquet files in the input folder")
    
    args = parser.parse_args()
    
    main(args.input_folder, args.output_folder, args.x, args.k, args.file_names, args.skip_transform, args.process_all)