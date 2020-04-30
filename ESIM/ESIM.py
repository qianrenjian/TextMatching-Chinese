# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from Encoder import Encoder
from Attention import SoftmaxAttention
from utils import replace_masked

class ESIM(nn.Module):
    def __init__(self, embed_size, hidden_size, dropout_rate, out_dim, pretrained_weight,
                padding_idx):
        super(ESIM, self).__init__()
        self.embed = nn.Embedding.from_pretrained(pretrained_weight,
                                                  freeze=False,
                                                  padding_idx=padding_idx)
        self.input_encode = Encoder(input_size=embed_size,
                                   hidden_size=hidden_size,
                                   dropout_rate=dropout_rate)
        self.proj = nn.Sequential(nn.Linear(8 * hidden_size, hidden_size), nn.ReLU())
        self.attention = SoftmaxAttention()
        self.inference_comp = Encoder(input_size=2 * hidden_size,
                                     hidden_size=hidden_size,
                                     dropout_rate=dropout_rate)
        self.classify = nn.Sequential(nn.Linear(8 * hidden_size, hidden_size),
                                      nn.ReLU(),
                                      nn.Dropout(p=dropout_rate),
                                      nn.Linear(hidden_size, hidden_size // 2),
                                      nn.ReLU(),
                                      nn.Dropout(p=dropout_rate),
                                      nn.Linear(hidden_size // 2, out_dim))
        self.padding_idx = padding_idx

    def forward(self, sent_a, sent_b):
        """
        sent_a: [batch_size, max_len_a]
        sent_b: [batch_size, max_len_b]
        """
        batch_size, max_len_a = sent_a.shape
        sent_a_mask = (sent_a != self.padding_idx).float()
        len_a = torch.sum(sent_a != self.padding_idx, dim=-1)
        
        _, max_len_b = sent_b.shape
        sent_b_mask = (sent_b != self.padding_idx).float()
        len_b = torch.sum(sent_b != self.padding_idx, dim=-1)

        # Embedding
        embed_a = self.embed(sent_a).float() # [batch_size, max_len_a, embed_size]
        embed_b = self.embed(sent_b).float() # [batch_size, max_len_b, embed_size]
        
        # Input encoding
        output_a = self.input_encode(embed_a, len_a) # [batch_size, max_len_a, 2 * hidden_size]
        output_b = self.input_encode(embed_b, len_b) # [batch_size, max_len_b, 2 * hidden_size]
        
        # Local inference modeling
        infer_a, infer_b = self.attention(output_a, sent_a_mask, output_b, sent_b_mask)
        ma = torch.cat([output_a, infer_a, output_a - infer_a, output_a * infer_a], dim=-1) # [batch_size, max_len_a, 8 * hidden_size]
        ma = self.proj(ma) # [batch_size, max_len_a, hidden_size]
        mb = torch.cat([output_b, infer_b, output_b - infer_b, output_b * infer_b], dim=-1) # [batch_size, max_len_b, 8 * hidden_size]
        mb = self.proj(mb) # [batch_size, max_len_b, hidden_size]
        
        # Inference Composition
        va = self.inference_comp(output_a, len_a) # [batch_size, max_len_a, 2 * hidden_size]
        vb = self.inference_comp(output_b, len_b) # [batch_size, max_len_b, 2 * hidden_size]
        
        vaave = torch.sum(va * sent_a_mask.unsqueeze(2), dim=1) / torch.sum(sent_a_mask, dim=1, keepdim=True) # [batch_size, 2 * hidden_size]
        vamax = replace_masked(va, sent_a_mask, -1e7).max(dim=1)[0] # [batch_size, 2 * hidden_size]
        vbave = torch.sum(vb * sent_b_mask.unsqueeze(2), dim=1) / torch.sum(sent_b_mask, dim=1, keepdim=True) # [batch_size, 2 * hidden_size]
        vbmax = replace_masked(vb, sent_b_mask, -1e7).max(dim=1)[0] # [batch_size, 2 * hidden_size]
        v = torch.cat([vaave, vamax, vbave, vbmax], dim=-1) # [batch_size, 8 * hidden_size]
        
        # FNN
        return self.classify(v)