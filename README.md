# CellPainTR: Contrastive Batch Corrected Transformer for Large Scale Cell Painting

## Abstract

Cell Painting, a high-content imaging-based profiling method, has emerged as a powerful tool for understanding cellular phenotypes and drug responses. However, batch effects severely constrain the integration and interpretation of data collected across different laboratories and experimental conditions. 

To mitigate this issue, we introduce **CellPainTR**, a novel embedding approach through Transformer for unified batch correction and representation learning of Cell Painting data, thereby addressing a critical challenge in the field of image-based profiling. Our approach employs a Transformer-like architecture with Hyena operators, positional encoding via morphological-feature-embedding, and a special source context token for batch correction, combined with a multi-stage training process that incorporates masked token prediction and supervised contrastive learning.

Experiments on the JUMP Cell Painting dataset demonstrate that CellPainTR significantly outperforms existing approaches such as Combat and Harmony across multiple evaluation metrics, while maintaining strong biological information retention as evidenced by improved clustering metrics and qualitative UMAP visualizations. Moreover, our method effectively reduces the feature space from thousands of dimensions to just 256, addressing the curse of dimensionality while maintaining high performance.

These advancements enable more robust integration of multi-source Cell Painting data, potentially accelerating progress in drug discovery and cellular biology research.

## Usage

### Custom Use

You can load any of the saved states found in `results/[step[1,2,3]]/checkpoint_[].pth` using the `models/model_evaluate.py` implementation.

### Inference

To calculate features for a whole data folder, use the `Inference_data_save_feature_dl.py` script. Make sure to fill in the following variables:

```python
config_path = './configs/Pretrain.yaml'
checkpoint_path = '...'
output_dir = '...'
```

### Visualization

Use the `Latent_Space_Exploration.py` file to create your own data visualizations. You can use both raw data or model representations.

Example command:

```bash
python Latent_Space_Exploration.py --config ./configs/Pretrain.yaml --dl_folder /path/to/dl/inference/data --preprocessed_folder /path/to/preprocessed/data --max_files_per_source 5 --color_column Metadata_Source --filters "{'Metadata_Plate': ['Plate1', 'Plate2']}"
```

#### Command-line Arguments

- `--config`: Path to the pretrain config file (required)
- `--dl_folder`: Path to the folder containing deep learning inference data files
- `--preprocessed_folder`: Path to the folder containing preprocessed data files
- `--max_files_per_source`: Maximum number of files to sample from each source (default: 1)
- `--color_column`: Metadata column to use for coloring the plots (default: 'Metadata_Source')
- `--filters`: YAML-formatted string of filters to apply to the data

Note: At least one of `--dl_folder` or `--preprocessed_folder` must be specified.
