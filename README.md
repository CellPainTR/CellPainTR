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

## Training from Scratch

(The complete process, from scratch training to full inference of the data and visualization as described in the paper, using one NVIDIA Quadro RTX 4000, took approximately 1 month)

### Dataset Preparation

1. Download the dataset:
   - Use the `data download/Data_Download.ipynb` notebook to download all Cell Profiles.
   - Execute up to "Join features with metadata" if you're only interested in downloading the data.

2. Retrieve metadata for future curation:
   - Use the `data download/Metadata Analysis and Selection.ipynb` notebook to generate a CSV file for dataset curation.
   - The current setup will filter only the Positive Control related data.

3. Curate the dataset:
   - Use `data_manipulation/filter_data_make_new_folder.py` to retrieve Compound data with metadata of interest.
     ```python
     input_folder = '../data/'
     output_folder = '../filtered_data/'
     ```
   - Use `data_manipulation/make_train_data.py` to retrieve only Positive Control Compounds.
     ```python
     input_folder = '../data_parquet/'
     output_folder = '../data_cp/'
     ```

4. Shuffle the data for Inter-Source supervised contrastive learning:
   - Use `data_manipulation/data_resampling.py`
   - Example command:
     ```bash
     python data_manipulation/data_resampling.py /path/to/input_folder /path/to/output_folder 5 3 file1 file2 file3 --skip-transform --process-all
     ```

5. Remove duplicate entries:
   - Use `data_manipulation/remove_duplicate_entries_filtered_data.py` for each created file.

### Training Process

1. Step One: Pretraining
   - Use `./pretrain.py`
   - Set `--checkpoint` to 'False' for training from scratch
   - Customize other arguments as needed

2. Step Two: Fine-tuning
   - Use `./Finetune_contrastive_stable.py`
   - Use the last checkpoint from step one as the initial checkpoint
   - For the first epoch, freeze all model parameters except the source token (uncomment freezing code in `models/model_contrastive_stable.py`)

3. Step Three: Inter-Source Training
   - Use the same training script as step two
   - Use the shuffled dataset (specify in the `pretrain.yaml` file)

### Experiments

To generate results as presented in the paper:

1. For baseline batch correction methods:
   - Use `experiment/main.py`
   - Example command:
     ```bash
     python experiment/main.py /path/to/input_dir --methods combat harmony mnn --label_key Metadata_InChIKey --batch_key Metadata_Batch --output_dir /path/to/results --preprocess
     ```

2. For model representations:
   - Use `experiment/main_dl.py`
   - Example command:
     ```bash
     python experiment/main_dl.py /path/to/input_dir --label_key Metadata_InChIKey --batch_key Metadata_Batch --output_dir /path/to/results --method deep_learning
     ```

Note: Ensure all paths and parameters are correctly set according to your specific setup and requirements.
