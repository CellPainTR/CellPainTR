import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from dataset import create_CSV_dataset, create_loader, create_dataset
from models.model_evaluate import HyenaMorph
from models.tokenization import RNATokenizer
import yaml
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path):
    logging.info(f"Loading configuration from {config_path}")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    logging.info(f"Configuration loaded successfully")
    return config

def load_model(config, checkpoint_path, device):
    logging.info(f"Loading model from {checkpoint_path}")
    model = HyenaMorph(config=config)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)
    logging.info(f"Model loaded successfully and moved to {device}")
    return model.to(device).eval()

def process_batch(model, tokenizer, Morph_ids, Feature_seq, Source_num, device):
    try:
        with torch.no_grad():
            Morph_input = tokenizer(Morph_ids, padding='longest', return_tensors="pt").to(device)
            Feature_seq = Feature_seq.to(device)
            Source_num = Source_num.to(device)
            output = model(Morph_input, Feature_seq, Source_num)
            
            # Ensure output is 2D tensor
            if output.dim() == 1:
                output = output.unsqueeze(0)
            
            return output
    except Exception as e:
        logging.error(f"Error processing batch: {e}")
        return None
    
def get_output_file_path(original_file, output_dir, config):
    relative_path = os.path.relpath(original_file, config['train_file'])
    output_file = os.path.join(output_dir, relative_path)
    output_file = output_file.rsplit('.', 1)[0] + '_dl.parquet'
    return output_file

def save_results(output, meta_df, original_file, output_dir, config):
    output_file = get_output_file_path(original_file, output_dir, config)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    output_np = output.cpu().numpy()

    # Create a DataFrame from output_np with named columns
    output_df = pd.DataFrame(output_np, columns=[f'output_{i}' for i in range(output_np.shape[1])])
    
    # Ensure index alignment
    output_df.index = meta_df.index
    
    # Concatenate horizontally
    result_df = pd.concat([meta_df, output_df], axis=1)

    assert result_df.shape[0] == meta_df.shape[0], "Mismatch between output and metadata row counts in save_results"
    assert result_df.shape[1] == meta_df.shape[1] + output_np.shape[1], "Unexpected number of columns in result_df"
    
    result_df.to_parquet(output_file, index=False)
    logging.info(f"Results saved to {output_file}")

def main(config_path, checkpoint_path, output_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    config = load_config(config_path)
    model = load_model(config, checkpoint_path, device)
    
    logging.info("Initializing tokenizer")
    tokenizer = RNATokenizer(vocab_file=os.path.join(config['data_path'], config['vocab']))

    logging.info("Creating CSV dataset")
    CSVdatasets = [create_CSV_dataset('pretrain', config)]
    CSVdataloader = create_loader(CSVdatasets, [None], batch_size=[1], num_workers=[4], is_trains=[False], collate_fns=[None])[0]

    total_files = len(CSVdataloader)
    logging.info(f"Total number of files to process: {total_files}")

    for file_idx, data in enumerate(tqdm(CSVdataloader, desc="Processing files"), 1):
        logging.info(f"Processing file {file_idx}/{total_files}: {data['file_name'][0]}")
        
        # Check if output file already exists
        output_file = get_output_file_path(data['file_name'][0], output_dir, config)
        if os.path.exists(output_file):
            logging.info(f"Output file already exists. Skipping to next file: {output_file}")
            continue
        
        datasets = [create_dataset('pretrain', data)]
        dataloader = create_loader(datasets, [None], batch_size=[config['batch_size']], num_workers=[4], is_trains=[False], collate_fns=[None])[0]

        all_outputs = []
        total_batches = len(dataloader)
        logging.info(f"Number of batches for this file: {total_batches}")

        for batch_idx, subdata in enumerate(dataloader, 1):
            Morph_ids = subdata['Pheno_ids']
            Feature_seq = subdata['Pheno_seq']
            Source_num = subdata['source_num']
            
            output = process_batch(model, tokenizer, Morph_ids, Feature_seq, Source_num, device)
            if output is not None:
                all_outputs.append(output)
            
            if batch_idx % 1 == 0:  # Log every batch
                logging.info(f"Processed batch {batch_idx}/{total_batches}")

        if not all_outputs:
            logging.warning(f"No valid outputs for file {data['file_name'][0]}. Skipping.")
            continue

        logging.info("Concatenating outputs")
        logging.info(f"Total processed items: {sum(o.shape[0] for o in all_outputs)}")
        full_output = torch.cat(all_outputs, dim=0)

        logging.info("Loading metadata")
        meta_df = pd.read_parquet(data['file_name'][0])
        meta_col = [col for col in meta_df.columns if col.startswith('Metadata_')]
        meta_df = meta_df[meta_col]

        logging.info("Saving results")
        assert full_output.shape[0] == meta_df.shape[0], "Mismatch between output and metadata row counts"
        save_results(full_output, meta_df, data['file_name'][0], output_dir, config)

        logging.info(f"Finished processing file {file_idx}/{total_files}")

    logging.info("All files processed successfully")

if __name__ == "__main__":
    config_path = './configs/Pretrain.yaml'
    checkpoint_path = '...'
    output_dir = '...'
    
    logging.info("Starting the inference process")
    main(config_path, checkpoint_path, output_dir)
    logging.info("Inference process completed")
