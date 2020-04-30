# -*- coding: utf-8 -*-
import torch

def masked_softmax(similarity_matrix, mask):
    """
    输入：
    similarity_matrix: [batch_size, seq_a, seq_b]
    mask: [batch_size, seq_b]
    输出：
    被mask掩盖后经过softmax运算的similarity matrix
    """
    batch_size, seq_len_a, seq_len_b = similarity_matrix.shape
    reshape_sim = similarity_matrix.view(-1, seq_len_b) # [batch_size * seq_a, seq_b]
    mask = mask.unsqueeze(1) # [batch_size, 1, seq_b]
    mask = mask.expand_as(similarity_matrix).contiguous().float() # [batch_size, seq_a, seq_b]
    reshape_mask = mask.view(-1, seq_len_b) # [batch_size * seq_a, seq_b]
    reshape_sim.masked_fill_(reshape_mask == 0, -1e7)
    result = torch.softmax(reshape_sim, dim=-1)
    result = result * reshape_mask # [batch_size * seq_a, seq_b]
    return result.view(batch_size, seq_len_a, seq_len_b)

def replace_masked(tensor, mask, value):
    """
    用value替换tensor中被mask的位置
    输入：
    tensor: [batch_size, seq_len, vec_dim]
    mask: [batch_size, seq_len]
    value: float
    """
    mask = mask.unsqueeze(2) # [batch_size, seq_len, 1]
    reverse_mask = 1.0 - mask
    values_to_add = value * reverse_mask
    return tensor * mask + values_to_add

def weighted_sum(tensor, weights, mask):
    """
    输入：
    tensor: [batch_size, seq_b, vec_dim]
    weights: [batch_size, seq_a, seq_b]
    mask: [batch_size, seq_a]
    """
    weighted_sum = torch.matmul(weights, tensor) # [batch_size, seq_a, vec_dim]
    mask = mask.unsqueeze(2) # [batch_size, seq_a, 1]
    mask = mask.expand_as(weighted_sum).contiguous().float() # [batch_size, seq_a, vec_dim]
    return weighted_sum * mask

def sort_by_seq_lens(batch, sequences_lengths, descending=True):
    sorted_seq_lens, sorting_index = sequences_lengths.sort(0, descending=descending)
    sorted_batch = batch.index_select(0, sorting_index)
    idx_range = torch.arange(0, len(sequences_lengths)).type_as(sequences_lengths)
    _, reverse_mapping = sorting_index.sort(0, descending=False)
    restoration_index = idx_range.index_select(0, reverse_mapping)
    return sorted_batch, sorted_seq_lens, sorting_index, restoration_index