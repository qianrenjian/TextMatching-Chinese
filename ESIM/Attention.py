# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from utils import masked_softmax, weighted_sum

class SoftmaxAttention(nn.Module):
    def forward(self, sent_a, sent_a_mask, sent_b, sent_b_mask):
        """
        输入：
        sent_a: [batch_size, seq_a_len, vec_dim]
        sent_a_mask: [batch_size, seq_a_len]
        sent_b: [batch_size, seq_b_len, vec_dim]
        sent_b_mask: [batch_size, seq_b_len]
        输出：
        sent_a_att: [batch_size, seq_a_len, seq_b_len]
        sent_b_att: [batch_size, seq_b_len, seq_a_len]
        """
        # similarity matrix
        similarity_matrix = torch.matmul(sent_a, sent_b.transpose(1, 2).contiguous()) # [batch_size, seq_a, seq_b]
        sent_a_b_attn = masked_softmax(similarity_matrix, sent_b_mask) # [batch_size, seq_a, seq_b]
        sent_b_a_attn = masked_softmax(similarity_matrix.transpose(1, 2).contiguous(), sent_a_mask) # [batch_size, seq_b, seq_a]
        sent_a_att = weighted_sum(sent_b, sent_a_b_attn, sent_a_mask) # [batch_size, seq_a, vec_dim]
        sent_b_att = weighted_sum(sent_a, sent_b_a_attn, sent_b_mask) # [batch_size, seq_b, vec_dim]
        return sent_a_att, sent_b_att