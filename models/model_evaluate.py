from models.hyena_source_token import HyenaModel

import torch
import json
from torch import nn


class HyenaMorph(nn.Module):
    def __init__(self,
                 tokenizer=None,
                 config=None
                 ):
        super().__init__()

        hyena_config = json.load(open(config['hyena_config']))

        self.encoder = HyenaModel(**hyena_config, use_head=False)

    
    def forward(self, Morph_input, Feature_seq, Source_num):
        Feature_seq_input = Feature_seq.clone().to(torch.float32)
        
        Context_vector = self.encoder(Morph_input.input_ids, Feature_seq_input, Source_num)
        
        Context_vector = Context_vector[:, 0, :]
            
        return Context_vector
