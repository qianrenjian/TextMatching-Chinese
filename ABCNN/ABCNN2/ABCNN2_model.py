import torch
import torch.nn as nn
from Embed import Embedding
from ABCNN2_layer import ABCNN_2

class ABCNN(nn.Module):
    def __init__(self, embed_size, pretrained, seq_len, width, num_layers, dropout_rate,
                padding_idx, out_dim):
        super(ABCNN, self).__init__()
        self.embed = Embedding(padding_idx=padding_idx, pretrained=pretrained,
                              dropout_rate=dropout_rate)
        self.convs = nn.ModuleList([ABCNN_2(seq_len, embed_size, width) for _ in range(num_layers)])
        self.proj = nn.Sequential(nn.Linear(embed_size * 2, embed_size),
                                  nn.ReLU(),
                                  nn.Dropout(p=dropout_rate),
                                  nn.Linear(embed_size, embed_size // 2),
                                  nn.ReLU(),
                                  nn.Dropout(p=dropout_rate),
                                  nn.Linear(embed_size // 2, out_dim)
                                 )
        self.dropout = nn.Dropout(p=dropout_rate)
        self.padding_idx = padding_idx
        
    def forward(self, sent0, sent1):
        """
        Input:
            (1) sent0: [batch_size, seq_len]
            (2) sent1: [batch_size, seq_len]
            (3) sent0_mask: [batch_size, seq_len]
            (4) sent1_mask: [batch_size, seq_len]
        """
        sent0_mask = (sent0 != self.padding_idx).float()
        sent1_mask = (sent1 != self.padding_idx).float()
        len0 = torch.sum(sent0_mask, dim=-1)
        len1 = torch.sum(sent1_mask, dim=-1)
        out0 = self.embed(sent0) # [batch_size, seq_len, embed_size]
        out1 = self.embed(sent1) # [batch_size, seq_len, embed_size]
        for conv in self.convs:
            out0, out1 = conv(out0, out1, sent0_mask, sent1_mask)
        sent0_mask = sent0_mask.unsqueeze(2).expand_as(out0) # [batch_size, seq_len, embed_size]
        sent1_mask = sent1_mask.unsqueeze(2).expand_as(out1) # [batch_size, seq_len, embed_size]
        out0 = out0 * sent0_mask
        out1 = out1 * sent1_mask
        out0 = torch.sum(out0, dim=1) / len0.unsqueeze(1) # [batch_size, embed_size]
        out1 = torch.sum(out1, dim=1) / len1.unsqueeze(1) # [batch_size, embed_size]
        feature = torch.cat([out0, out1], dim=-1) # [batch_size, embed_size * 2]
        return self.proj(self.dropout(feature))