from models.hyena_source_token import HyenaModel

import torch
import json
import numpy as np
from torch import nn


class HyenaMorph(nn.Module):
    def __init__(self,
                 tokenizer=None,
                 config=None
                 ):
        super().__init__()

        hyena_config = json.load(open(config['hyena_config']))

        self.tokenizer = tokenizer
        self.mrm_probability = config['mrm_probability']
        assert self.mrm_probability >= 0.05

        self.encoder = HyenaModel(**hyena_config, use_head=False)
        self.regressor = nn.Sequential(
            nn.Linear(hyena_config['d_model'], hyena_config['d_model']),
            nn.GELU(),
            nn.LayerNorm(hyena_config['d_model'], 1e-5),
            nn.Linear(hyena_config['d_model'], 1)
        )
    
    def make_masks(self, Pheno_group_mask, mrm_probability):
        Pheno_group_mask = Pheno_group_mask.cpu().numpy()
        unique_values = np.unique(Pheno_group_mask)
        
        final_mask = np.zeros(Pheno_group_mask.shape)
        
        for uniq_val in unique_values:
            Pheno_group_mask_val = (Pheno_group_mask == uniq_val)*1
            mask = (np.random.random_sample(Pheno_group_mask.shape) > mrm_probability)*1
            
            mask_i = Pheno_group_mask_val * mask
            
            final_mask = final_mask + mask_i
            
        return torch.from_numpy(abs(final_mask-1)).bool()
        
    def forward(self, Morph_input, Feature_seq, Pheno_group_mask, Source_num, alpha=0):
        Feature_seq_input = Feature_seq.clone().to(torch.float32)
        Feature_seq_target = Feature_seq.clone().to(torch.float32)
        RNA_ids = Morph_input.input_ids[:, 1:].clone()
        
        mrm_probability = np.random.random() * (self.mrm_probability - 0.05) + 0.05
        masked_indices = self.make_masks(Pheno_group_mask, mrm_probability)
        
        Feature_seq_input, Feature_seq_target, masked_indices = self.mask(Feature_seq_input, RNA_ids, self.tokenizer, targets=Feature_seq_target, masked_indices=masked_indices)
        
        Context_vector = self.encoder(Morph_input.input_ids, Feature_seq_input, Source_num, masked_indices=masked_indices)
        mrm_output = self.regressor(Context_vector)[:, 1:-1, 0]

        loss = (mrm_output - Feature_seq_target) ** 2
        loss = loss[Feature_seq_target != -100].mean()
        
        return loss
    
    def mask(self, input_seq, input_ids, tokenizer, targets=None, masked_indices=None, probability_matrix=None):
        if masked_indices is None:
            masked_indices = torch.bernoulli(probability_matrix).bool()
        
        masked_indices[input_ids == tokenizer.pad_token_id] = False
        masked_indices[input_seq == 0.0] = False

        if targets is not None:
            targets[~masked_indices] = -100

        input_seq[masked_indices] = 0.0

        if targets is not None:
            return input_seq, targets, masked_indices
        else:
            return input_seq, masked_indices
