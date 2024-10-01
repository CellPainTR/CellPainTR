import os
import pandas as pd

def remove_duplicates(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.parquet'):
                file_path = os.path.join(root, file)
                df = pd.read_parquet(file_path)
                original_count = len(df)
                df = df.drop_duplicates()
                new_count = len(df)
                if new_count < original_count:
                    print(f"Removed {original_count - new_count} duplicates from {file_path}")
                    df.to_parquet(file_path, index=False)
                else:
                    print(f"No duplicates found in {file_path}")

if __name__ == "__main__":
    output_folder = '../train_data/'
    remove_duplicates(output_folder)