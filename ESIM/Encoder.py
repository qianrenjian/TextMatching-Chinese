# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from utils import sort_by_seq_lens

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.encoder = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                             num_layers=2, dropout=dropout_rate, bidirectional=True,
                             batch_first=True)
    
    def forward(self, sequence_batch, sequence_lengths):
        sorted_batch, sorted_seq_lens, _, restoration_index = sort_by_seq_lens(sequence_batch, sequence_lengths)
        packed_batch = nn.utils.rnn.pack_padded_sequence(sorted_batch, sorted_seq_lens,
                                                        batch_first=True)
        output, _ = self.encoder(packed_batch)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output.index_select(0, restoration_index)