import torch

def sort_by_seq_lens(batch, sequences_lengths, descending=True):
    sorted_seq_lens, sorting_index = sequences_lengths.sort(0, descending=descending)
    sorted_batch = batch.index_select(0, sorting_index)
    idx_range = torch.arange(0, len(sequences_lengths)).type_as(sequences_lengths)
    _, reverse_mapping = sorting_index.sort(0, descending=False)
    restoration_index = idx_range.index_select(0, reverse_mapping)
    return sorted_batch, sorted_seq_lens, sorting_index, restoration_index