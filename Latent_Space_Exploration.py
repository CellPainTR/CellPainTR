import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import yaml
from glob import glob
import random
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import torch
from dataset import RobustMAD, cap_outliers

pio.renderers.default = 'browser'

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def get_files(folder_path, max_files_per_source, file_suffix='.parquet', random_seed=42):
    random.seed(random_seed)
    
    all_files = glob(os.path.join(folder_path, '**', f'*{file_suffix}'), recursive=True)
    files_by_source = {}
    for file in all_files:
        source = os.path.basename(os.path.dirname(file))
        if source not in files_by_source:
            files_by_source[source] = []
        files_by_source[source].append(file)
    
    if max_files_per_source != 0:
        selected_files = []
        for source, files in files_by_source.items():
            selected_files.extend(random.sample(files, min(max_files_per_source, len(files))))
    else:
        selected_files = all_files
    
    return selected_files

def preprocess_data(df, config):
    feature_cols = [col for col in df.columns if any(dye in col for dye in config['dyes']) and any(part in col for part in config['parts'])]
    feature_data = df[feature_cols].replace([np.inf, -np.inf], 0).fillna(0).astype('float64')
    
    robust_mad_instance = RobustMAD()
    robust_mad_instance.fit(feature_data[df['Metadata_InChIKey'] == config['nega_con']])
    feature_data = robust_mad_instance.transform(feature_data)
    
    feature_data = feature_data.replace([np.inf, -np.inf], 0).fillna(0)
    
    feature_data = feature_data.values
    feature_data = cap_outliers(feature_data)
    
    return pd.DataFrame(feature_data, columns=feature_cols)

def apply_filters(df, filters):
    if not filters:
        return df
    
    mask = pd.Series([True] * len(df))
    for column, values in filters.items():
        mask &= df[column].isin(values)
    
    return df[mask]

def load_data(files, config, data_type, filters):
    all_data = []
    all_metadata = []
    for file in tqdm(files, desc=f"Loading {data_type} files"):
        df = pd.read_parquet(file)
        
        metadata_cols = [col for col in df.columns if col.startswith('Metadata_')]
        non_metadata_cols = [col for col in df.columns if col not in metadata_cols]
        
        # Apply filters
        df = apply_filters(df, filters)
        
        # Skip to next file if no data remains after filtering
        if df.empty:
            continue
        
        metadata = df[metadata_cols]
        
        if data_type == 'dl':
            data = df[non_metadata_cols]
        elif data_type == 'preprocessed':
            data = preprocess_data(df, config)
        else:
            raise ValueError("Invalid data_type. Choose 'dl' or 'preprocessed'.")
        
        # Drop rows where Metadata_Source is None
        rows_to_keep = metadata['Metadata_Source'].notna()
        metadata = metadata[rows_to_keep]
        data = data[rows_to_keep]
        
        # Ensure data and metadata have the same index
        data = data.reset_index(drop=True)
        metadata = metadata.reset_index(drop=True)
        
        # Identify rows with NaN values in non-metadata columns
        rows_to_keep = ~data.isna().any(axis=1)
        
        # Keep only the rows without NaN values in both data and metadata
        data_without_nan = data[rows_to_keep]
        metadata_without_nan = metadata[rows_to_keep]
        
        all_data.append(data_without_nan)
        all_metadata.append(metadata_without_nan)
    
    if not all_data:
        return pd.DataFrame(), pd.DataFrame()
    
    return pd.concat(all_data, ignore_index=True), pd.concat(all_metadata, ignore_index=True)

def plot_embedding(embedding, metadata, method_name, color_column):
    df = pd.DataFrame(embedding, columns=['x', 'y'])
    
    # Add the specified metadata column to the dataframe
    df[color_column] = metadata[color_column].fillna('Unknown')
    
    # Create the scatter plot
    fig = px.scatter(
        df, x='x', y='y',
        color=color_column,
        hover_data=[color_column],
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    
    fig.update_traces(marker=dict(size=2))
    
    fig.update_layout(
        title=f'{method_name} Visualization - Color: {color_column}',
        xaxis_title='Dimension 1',
        yaxis_title='Dimension 2'
    )
    
    fig.show()

def main(args):
    config = load_config(args.config)
    color_column = args.color_column
    filters = yaml.safe_load(args.filters) if args.filters else {}
    
    dl_files = get_files(args.dl_folder, args.max_files_per_source, '_dl.parquet') if args.dl_folder else []
    preprocessed_files = get_files(args.preprocessed_folder, args.max_files_per_source) if args.preprocessed_folder else []

    data_list = []
    metadata_list = []

    if dl_files:
        dl_data, dl_metadata = load_data(dl_files, config, 'dl', filters)
        if not dl_data.empty:
            data_list.append(dl_data)
            metadata_list.append(dl_metadata)

    if preprocessed_files:
        preprocessed_data, preprocessed_metadata = load_data(preprocessed_files, config, 'preprocessed', filters)
        if not preprocessed_data.empty:
            data_list.append(preprocessed_data)
            metadata_list.append(preprocessed_metadata)

    if not data_list:
        raise ValueError("No data loaded after applying filters. Check your filter criteria or input data.")

    data = pd.concat(data_list, axis=1)
    metadata = metadata_list[0]  # Assuming metadata is the same for both types

    # Dimensionality reduction
    pca = PCA(n_components=2)
    tsne = TSNE(n_components=2, random_state=42)
    umap = UMAP(random_state=42)

    print("Performing PCA...")
    pca_result = pca.fit_transform(data)
    print("Performing t-SNE...")
    tsne_result = tsne.fit_transform(data)
    print("Performing UMAP...")
    umap_result = umap.fit_transform(data)

    # Plotting
    print("Generating plots...")
    plot_embedding(pca_result, metadata, 'PCA', color_column)
    plot_embedding(tsne_result, metadata, 't-SNE', color_column)
    plot_embedding(umap_result, metadata, 'UMAP', color_column)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to the pretrain config file')
    parser.add_argument('--dl_folder', help='Path to the folder containing deep learning inference data files')
    parser.add_argument('--preprocessed_folder', help='Path to the folder containing preprocessed data files')
    parser.add_argument('--max_files_per_source', type=int, default=1, help='Maximum number of files to sample from each source')
    parser.add_argument('--color_column', default='Metadata_Source', help='Metadata column to use for coloring the plots')
    parser.add_argument('--filters', help='YAML-formatted string of filters to apply to the data')
    args = parser.parse_args()

    if not args.dl_folder and not args.preprocessed_folder:
        parser.error("At least one of --dl_folder or --preprocessed_folder must be specified.")

    main(args)