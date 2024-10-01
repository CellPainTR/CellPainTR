#main

import argparse
import os
import logging
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
from preprocessing import read_parquet, preprocess_data, to_anndata, get_common_features, align_features, feature_selection
from batch_correction import combat_func, harmony_func, mnn_func, scanorama_func, scvi_func, desc_func, sphering_func
from metrics import calculate_metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import plotly.graph_objects as go
import plotly.io as pio

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_parquet_files(directory):
    """Get all parquet files in the directory structure."""
    return [os.path.join(root, file) for root, _, files in os.walk(directory) 
            for file in files if file.endswith('.parquet')]

def process_file(file_path, preprocess=False):
    """Process a single parquet file."""
    logger.info(f"Processing file: {file_path}")
    metadata, features = read_parquet(file_path)
    if preprocess:
        metadata, features = preprocess_data(metadata, features)
    return metadata, features

def parallel_process_files(file_list, preprocess=False):
    """Process files in parallel."""
    with Pool(cpu_count()) as pool:
        results = pool.map(partial(process_file, preprocess=preprocess), file_list)
    return zip(*results)

def run_batch_correction(method, adata, batch_key, label_key):
    """Run the specified batch correction method on combined data."""
    # Ensure batch_key is categorical
    adata.obs[batch_key] = adata.obs[batch_key].astype('category')
    
    if method == 'combat':
        adata = combat_func(adata, batch_key)
    elif method == 'harmony':
        adata = harmony_func(adata, batch_key)
    elif method == 'mnn':
        adata = mnn_func(adata, batch_key)
    elif method == 'scanorama':
        adata = scanorama_func(adata, batch_key)
    elif method == 'scvi':
        adata = scvi_func(adata, batch_key)
    elif method == 'desc':
        adata = desc_func(adata, batch_key)
    elif method == 'sphering':
        adata = sphering_func(adata)
    elif method == 'baseline':
        return adata
    else:
        raise ValueError(f"Unknown batch correction method: {method}")

    # Check for NaN values and handle them
    if np.any(np.isnan(adata.X)):
        adata.X = np.nan_to_num(adata.X, nan=1e-9)
        print("NaN values detected in the corrected data. Replaced with ~0.")

    return adata

def load_microscope_config(file_path):
    config = pd.read_csv(file_path)
    return dict(zip(config['Metadata_Source'], config['Metadata_Microscope_Name']))

def plot_embedding(embedding, adata, color_key, title, output_file):
    color_values = adata.obs[color_key]
    unique_colors = color_values.unique()
    color_map = {color: f'rgb({int(r)}, {int(g)}, {int(b)})' for color, (r, g, b) in 
                 zip(unique_colors, plt.cm.get_cmap('tab20')(np.linspace(0, 1, len(unique_colors)))[:, :3] * 255)}
    
    traces = []
    for color in unique_colors:
        mask = color_values == color
        traces.append(go.Scatter(
            x=embedding[mask, 0],
            y=embedding[mask, 1],
            mode='markers',
            name=color,
            marker=dict(color=color_map[color], size=1),
            hoverinfo='text',
            text=[f"{color_key}: {color}<br>Index: {i}" for i in adata.obs.index[mask]]
        ))

    layout = go.Layout(
        title=dict(
            text=title,
            font=dict(color='black')  # Black title
        ),
        xaxis=dict(
            title=dict(
                text=f'{title.split("-")[0]} 1',
                font=dict(color='black')  # Black x-axis label
            ),
            showgrid=False,
            zeroline=False,
            tickfont=dict(color='black')  # Black x-axis tick labels
        ),
        yaxis=dict(
            title=dict(
                text=f'{title.split("-")[0]} 2',
                font=dict(color='black')  # Black y-axis label
            ),
            showgrid=False,
            zeroline=False,
            tickfont=dict(color='black')  # Black y-axis tick labels
        ),
        hovermode='closest',
        legend=dict(
            x=1.05, 
            y=1, 
            bordercolor='Black', 
            borderwidth=1,
            font=dict(color='black')  # Black legend text
        ),
        width=900,
        height=700,
        margin=dict(l=50, r=50, t=50, b=50),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    fig = go.Figure(data=traces, layout=layout)
    pio.write_image(fig, output_file, scale=2)

def generate_embeddings(adata):
    pca = PCA(n_components=2).fit_transform(adata.X)
    tsne = TSNE(n_components=2, random_state=42).fit_transform(adata.X)
    umap_embedding = umap.UMAP(random_state=42).fit_transform(adata.X)
    return pca, tsne, umap_embedding

def plot_embeddings(adata, method, output_dir):
    pca, tsne, umap_embedding = generate_embeddings(adata)
    
    for embedding, name in zip([pca, tsne, umap_embedding], ['PCA', 't-SNE', 'UMAP']):
        for color_key in ['Metadata_InChIKey', 'Metadata_Source', 'Metadata_Microscope_Name']:
            title = f"{name} - {method} - Colored by {color_key}"
            output_file = os.path.join(output_dir, 'figures', method, f"{method}_{name}_{color_key}.png")
            plot_embedding(embedding, adata, color_key, title, output_file)

def main():
    parser = argparse.ArgumentParser(description="Run batch correction and calculate metrics on all files together.")
    parser.add_argument("input_dir", help="Directory containing parquet files")
    parser.add_argument("--label_key", default="Metadata_InChIKey", help="Column name for label information")
    parser.add_argument("--batch_key", default="Metadata_Batch", help="Column name for batch information")
    parser.add_argument("--output_dir", default="results", help="Directory to save results")
    parser.add_argument("--methods", nargs='+', choices=['combat', 'harmony', 'mnn', 'scanorama', 'scvi', 'desc', 'sphering', 'baseline'],
                        required=True, help="Batch correction methods to use")
    parser.add_argument("--preprocess", action="store_true", help="Apply preprocessing pipeline before batch correction")
    
    args = parser.parse_args()
    
    # Get all parquet files
    logger.info("Finding all parquet files in the input directory...")
    parquet_files = get_parquet_files(args.input_dir)
    logger.info(f"Found {len(parquet_files)} parquet files.")
    
    # Process all files in parallel
    logger.info("Processing all files in parallel...")
    all_metadata, all_features = parallel_process_files(parquet_files, preprocess=args.preprocess)
    logger.info("All files processed.")

    # Get common features across all files
    logger.info("Finding common features across all files...")
    common_features = get_common_features(all_features)
    logger.info(f"Found {len(common_features)} common features.")

    # Align features across all files
    logger.info("Aligning features across all files...")
    aligned_features = align_features(all_features, common_features)

    # Combine all data
    logger.info("Combining all data...")
    combined_metadata = pd.concat(all_metadata, ignore_index=True)
    combined_features = pd.concat(aligned_features, ignore_index=True)

    # Convert batch_key and label_key to categorical
    combined_metadata[args.batch_key] = combined_metadata[args.batch_key].astype('category')
    combined_metadata[args.label_key] = combined_metadata[args.label_key].astype('category')

    # Add logging for debugging
    logger.info(f"Total rows in combined metadata: {len(combined_metadata)}")
    logger.info(f"Total rows in combined features: {len(combined_features)}")

    # Select Features
    logger.info("Selecting Features...")
    combined_features = feature_selection(combined_features)

    # Handle any remaining NaN values
    logger.info("Handling remaining NaN values...")
    combined_features = combined_features.fillna(combined_features.median())

    # Create AnnData object
    logger.info("Creating AnnData object...")
    adata = to_anndata(combined_metadata, combined_features)

    df_data_print = pd.DataFrame(adata.X)
    print(df_data_print.describe())

    print(f"NaN count adata: {np.isnan(adata.X).sum()}")
    
    # Load microscope configuration
    microscope_config = load_microscope_config('../data/metadata/microscope_config.csv')
    
    # Process each method
    for method in args.methods:
        logger.info(f"Applying {method} batch correction to all data...")
        corrected_adata = run_batch_correction(method, adata.copy(), args.batch_key, args.label_key)
        
        # Add microscope information to adata
        corrected_adata.obs['Metadata_Microscope_Name'] = corrected_adata.obs['Metadata_Source'].map(microscope_config)
        
        # Generate and save embeddings
        logger.info(f"Generating and saving embeddings for {method}...")
        plot_embeddings(corrected_adata, method, args.output_dir)
        
        # Calculate metrics
        logger.info("Calculating metrics...")
        results = calculate_metrics(corrected_adata, args.label_key, args.batch_key)
        
        # Save results
        logger.info("Saving results...")
        os.makedirs(args.output_dir, exist_ok=True)
        output_file = os.path.join(args.output_dir, f'metric_results_{method}.csv')
        pd.DataFrame([results]).to_csv(output_file, index=False)
        logger.info(f"Results for {method} saved to {output_file}")

if __name__ == "__main__":
    main()