import os

from torch.utils.data import Dataset
import torch

import pandas as pd
import numpy as np
import json

import functools

from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import median_abs_deviation

class RobustMAD(BaseEstimator, TransformerMixin):
    """Class to perform a "Robust" normalization with respect to median and mad

        scaled = (x - median) / mad

    Attributes
    ----------
    epsilon : float
        fudge factor parameter
    """

    def __init__(self, epsilon=1e-6):
        self.epsilon = epsilon

    def fit(self, X, y=None):
        """Compute the median and mad to be used for later scaling.

        Parameters
        ----------
        X : pandas.core.frame.DataFrame
            dataframe to fit RobustMAD transform

        Returns
        -------
        self
            With computed median and mad attributes
        """
        # Get the mean of the features (columns) and center if specified
        self.median = X.median()
        # The scale param is required to preserve previous behavior. More info at:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.median_absolute_deviation.html#scipy.stats.median_absolute_deviation
        self.mad = pd.Series(
            median_abs_deviation(X, nan_policy="omit", scale=1 / 1.4826),
            index=self.median.index,
        )
        return self

    def transform(self, X, copy=None):
        """Apply the RobustMAD calculation

        Parameters
        ----------
        X : pandas.core.frame.DataFrame
            dataframe to fit RobustMAD transform

        Returns
        -------
        pandas.core.frame.DataFrame
            RobustMAD transformed dataframe
        """
        return (X - self.median) / (self.mad + self.epsilon)

def cap_outliers(data, lower_percentile=1, upper_percentile=99):
    lower = np.percentile(data, lower_percentile)
    upper = np.percentile(data, upper_percentile)
    return np.clip(data, lower, upper)

def load_csv_files(directories):
    """Load and return a shuffled list of CSV file paths from the provided directories."""
    csv_files = [os.path.join(directory, file_name)
                 for directory in directories
                 for file_name in os.listdir(directory)
                 if file_name.endswith('.parquet')]
    return np.random.permutation(csv_files)

def get_csv_files(folder_path):
    """
    Retrieves all CSV files from the given folder path and its nested subfolders,
    excluding files in folders named "metadata".

    Args:
        folder_path (str): The path to the folder to search for CSV files.

    Returns:
        list: A list of file paths for all CSV files found, excluding those in "metadata" folders.
    """
    csv_files = []
    for root, dirs, files in os.walk(folder_path):
        dirs[:] = [d for d in dirs if d != 'metadata']  # Skip "metadata" folders
        for file in files:
            if file.endswith('.parquet'):
                csv_files.append(os.path.join(root, file))
    return csv_files

def get_csv_files_names(folder_path, folder_keywords):
    """
    Retrieves all CSV files from the given folder path and its nested subfolders,
    including only folders that contain any of the specified keywords.

    Args:
        folder_path (str): The path to the folder to search for CSV files.
        folder_keywords (list): A list of strings that the folder name should contain.

    Returns:
        list: A list of file paths for all CSV files found in matching folders.
    """
    csv_files = []
    for root, dirs, files in os.walk(folder_path):
        # Check if the current folder contains any of the keywords
        if not folder_keywords == None:
            if any(keyword.lower() in os.path.basename(root).lower() for keyword in folder_keywords):
                for file in files:
                    if file.endswith('.parquet'):
                        csv_files.append(os.path.join(root, file))
        else:
            for file in files:
                if file.endswith('.parquet'):
                    csv_files.append(os.path.join(root, file))
    return csv_files

def get_feature_columns(header, dyes, parts):
    """Return the feature columns that match the given dyes and parts."""
    return [col for col in header if any(dye in col for dye in dyes) and any(part in col for part in parts)]


def merge_with_meta(meta, df):
    """Merge the data frame with meta data."""
    return pd.merge(meta, df, on=['Metadata_Source', 'Metadata_Plate', 'Metadata_Well'], how='right')


def get_normalization_params(data, feature_cols):
    """Compute and return median and MAD for normalization."""
    median = np.median(data[feature_cols], axis=0)
    mad = np.median(np.abs(data[feature_cols] - median), axis=0)
    return median, mad

def get_group_mask(feature_cols, dyes, parts):
    """Return a mask array representing groups of features."""
    feature_groups = [''.join([part + dye for dye in dyes for part in parts if dye in col and part in col]) for col in feature_cols]
    group_dict = {group: idx for idx, group in enumerate(set(feature_groups))}
    return np.array([group_dict[group] for group in feature_groups])


def encode_labels(labels, encoding_dict):
    """Encode labels using the provided encoding dictionary."""
    vectorized_mapping = np.vectorize(lambda x: encoding_dict.get(x, x))
    return vectorized_mapping(labels)

def transform_column_names(column_names):
    return ' '.join(col.strip().lower().replace('_', '') for col in column_names)


def load_json(path):
    with open(path, 'r') as json_file:
        data = json.load(json_file)
    return data

class CsvDataset(Dataset):
    def __init__(self, csv_dirs, config):
        self.csv_files = get_csv_files_names(csv_dirs, config['folder_keywords'])
        self.dyes = config['dyes']
        self.parts = config['parts']
        self.neg_control = config['nega_con']
        self.Pheno_ids_fast_load_path = config['Pheno_ids_fast_load']
        self.data_cache = {}
        self.compound_encoding_dict = load_json(config['compound_encoding_path'])

        
        with open(self.Pheno_ids_fast_load_path, "r") as file:
            self.pheno_ids = file.read()
        
    @functools.lru_cache(maxsize=None)
    def load_csv(self, file_path):
        df = pd.read_parquet(file_path)
        df['Metadata_Plate'] = df['Metadata_Plate'].astype(str)
        return df

    def __len__(self):
        return len(self.csv_files)

    def __getitem__(self, idx):
        file_path = self.csv_files[idx]
        
        if file_path not in self.data_cache:
            df = self.load_csv(file_path)
    
            header = df.columns
            feature_cols = get_feature_columns(header, self.dyes, self.parts)
    
            # Labels and source numbers
            labels = encode_labels(df.Metadata_InChIKey.values, self.compound_encoding_dict)
            source_num = df.Metadata_Source.apply(lambda x: int(x.split('_')[1])).values
    
            # Feature data
            feature_data = df[feature_cols].replace([np.inf, -np.inf], 0).fillna(0).astype('float64')
            
            robust_mad_instance = RobustMAD()
            robust_mad_instance.fit(feature_data[df['Metadata_InChIKey'] == self.neg_control])
            feature_data = robust_mad_instance.transform(feature_data)
            
            feature_data = feature_data.values
            
            feature_data = cap_outliers(feature_data)
            
            feature_data = torch.from_numpy(feature_data)
            
            
            group_mask = get_group_mask(feature_cols, self.dyes, self.parts)
    
            data = {
                'Pheno_ids': self.pheno_ids,
                'Pheno_seq': feature_data,
                'file_name': file_path,
                'mask_group': group_mask,
                'labels': labels,
                'source_num': source_num
            }
            self.data_cache[file_path] = data
        else:
            data = self.data_cache[file_path]
        return data

class PretrainDataset(Dataset):
    def __init__(self, data):
        self.pheno_seq = data['Pheno_seq'].squeeze(0)
        self.pheno_ids = data['Pheno_ids'][0]
        self.pheno_group_mask = data['mask_group'].squeeze(0)
        self.labels = data['labels'].squeeze(0)
        self.source_num = data['source_num'].squeeze(0)

    def __len__(self):
        return len(self.pheno_seq)

    def __getitem__(self, idx):
        return {
            'Pheno_ids': self.pheno_ids,
            'Pheno_seq': self.pheno_seq[idx],
            'Pheno_group_mask': self.pheno_group_mask,
            'labels': self.labels[idx],
            'source_num': self.source_num[idx]
        }

class evaluate_dataset(Dataset):
    def __init__(self, data):
        
        self.Pheno_seq = data['Pheno_seq']
        self.Pheno_ids = data['Pheno_ids']
        self.file_name = data['file_name']

    def __len__(self):
        return len(self.Pheno_seq)

    def __getitem__(self, idx):
        
        return self.Pheno_ids[idx], self.Pheno_seq[idx], self.file_name[0]