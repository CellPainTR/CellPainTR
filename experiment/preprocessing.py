#Preprocessing

import pandas as pd
import anndata as ad
import numpy as np
from scipy.stats import median_abs_deviation

def read_parquet(file_path):
    """Read a parquet file and split it into metadata and features."""
    df = pd.read_parquet(file_path)

    metadata_cols = [col for col in df.columns if col.startswith('Metadata_')]
    feature_cols = [col for col in df.columns if not col.startswith('Metadata_')]
    
    metadata = df[metadata_cols]
    features = df[feature_cols]
    
    # Convert features to float, replacing non-convertible values with NaN
    features = features.apply(pd.to_numeric, errors='coerce')

    # Calculate the median for each feature across all non-NaN values
    feature_medians = features.median()

    # Fill NaN values with the calculated medians
    features = features.fillna(feature_medians)
    
    return metadata, features

def get_common_features(feature_dfs):
    """Get the set of common features across all DataFrames."""
    common_features = set.intersection(*[set(df.columns) for df in feature_dfs])
    return list(common_features)

def align_features(features_list, common_features):
    """Align all feature DataFrames to have the same columns."""
    aligned_features = []
    for features in features_list:
        missing_cols = set(common_features) - set(features.columns)
        for col in missing_cols:
            features[col] = np.nan
        aligned_features.append(features[common_features])
    return aligned_features

def to_anndata(metadata, features):
    """Convert metadata and features to AnnData object."""
    return ad.AnnData(X=features, obs=metadata)

def filter_dmso(metadata, features):
    """Filter out DMSO samples."""
    non_dmso_ix = metadata['Metadata_JCP2022'] != 'DMSO'
    return metadata[non_dmso_ix].reset_index(drop=True), features[non_dmso_ix].reset_index(drop=True)

def variation_filtering(features, threshold=1e-4):
    """Filter out features with low variance."""
    X = features.values
    X_median = np.nanmedian(X, axis=0)
    X_mad = np.nanmedian(np.abs(X - X_median), axis=0)
    C_var = np.divide(X_mad, X_median, out=np.zeros_like(X_mad), where=X_median!=0)
    return features.iloc[:, C_var >= threshold]


def robust_mad_normalization(features, epsilon=1e-6):
    """
    Perform Robust MAD normalization on the data.
    
    Parameters:
    -----------
    features : pandas.DataFrame
        The DataFrame containing the features to normalize.
    epsilon : float, optional
        A small constant to avoid division by zero. Default is 1e-6.
    
    Returns:
    --------
    normalized_features : pandas.DataFrame
        A DataFrame with normalized features.
    """
    # Compute median and MAD
    median = features.median()
    mad = features.apply(lambda x: median_abs_deviation(x, nan_policy="omit", scale=1/1.4826))
    
    # Perform normalization
    normalized_features = (features - median) / (mad + epsilon)
    
    return normalized_features

def rank_based_inverse_normal_transform(features):
    """Apply rank-based inverse normal transformation."""
    from scipy import stats
    def int_transform(x):
        return stats.norm.ppf((stats.rankdata(x) - 0.5) / len(x))
    
    return features.apply(int_transform)

def feature_selection(features, correlation_threshold=0.9):
    """Select features based on correlation threshold."""
    corr_matrix = features.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > correlation_threshold)]
    return features.drop(columns=to_drop)

def preprocess_data(metadata, features):
    """Apply the full preprocessing pipeline to both metadata and features."""
    # Preprocess features
    features = variation_filtering(features)
    features = robust_mad_normalization(features)
    features = rank_based_inverse_normal_transform(features)

    return metadata, features