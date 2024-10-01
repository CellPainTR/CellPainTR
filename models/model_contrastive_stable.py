from models.hyena_source_token import HyenaModel

import torch
import json
import numpy as np
from torch import nn
import torch.nn.functional as F
from pytorch_metric_learning import losses


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
        
        '''
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        self.encoder.backbone.embeddings.source_token.weight.requires_grad = True
        '''
        
        self.regressor = nn.Sequential(
            nn.Linear(hyena_config['d_model'], hyena_config['d_model']),
            nn.GELU(),
            nn.LayerNorm(hyena_config['d_model'], 1e-5),
            nn.Linear(hyena_config['d_model'], 1)
        )
        
        '''
        for param in self.regressor.parameters():
            param.requires_grad = False
        '''
        
        self.SupLoss = SupConLoss()
        self.lossweight = 1
    
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
        
    def forward(self, Morph_input, Feature_seq, Pheno_group_mask, Source_num, labels, alpha=0):
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
        
        suploss = self.SupLoss(Context_vector[:, 0, :], labels)
        
        return loss + suploss
    
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

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        
        features = F.normalize(features, p=2, dim=1)
        features = features.unsqueeze(1)
        
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
