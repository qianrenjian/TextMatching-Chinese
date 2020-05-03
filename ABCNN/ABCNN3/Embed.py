import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, padding_idx, pretrained, dropout_rate):
        super(Embedding, self).__init__()
        self.embed = nn.Embedding.from_pretrained(pretrained,
                                                 freeze=False,
                                                 padding_idx=padding_idx)
        self.dropout = nn.Dropout(p=dropout_rate)
    
    def forward(self, text):
        """
        Input:
        text size = [batch_size, seq_len]
        
        Output:
        embed text size = [batch_size, seq_len, embed_size]
        """
        return self.dropout(self.embed(text)).float()