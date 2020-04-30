import torch
import torch.nn as nn
from Encoder import Encoder

class SiameseGRU(nn.Module):
    def __init__(self, embed_size, hidden_size, bidirectional, dropout_rate,
                pretrained_weight, padding_idx, out_dim):
        super(SiameseGRU, self).__init__()
        self.embed = nn.Embedding.from_pretrained(pretrained_weight,
                                                 freeze=False,
                                                 padding_idx=padding_idx)
        self.gru = Encoder(input_size=embed_size, hidden_size=hidden_size,
                          dropout_rate=dropout_rate)
        self.proj = nn.Linear(8 * hidden_size, out_dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.padding_idx = padding_idx
        self.hidden_size = hidden_size
        self.init_params()
        
    def init_params(self):
        nn.init.xavier_normal_(self.proj.weight)
        nn.init.constant_(self.proj.bias, 0)
    
    def forward(self, texta, textb):
        """
        Input:
            texta: [batch_size, max_len_a]
            textb: [batch_size, max_len_b]
        """
        batch_size, max_len_a = texta.shape
        _, max_len_b = textb.shape
        lens_a = torch.sum(texta != self.padding_idx, dim=-1) #[batch_size]
        lens_b = torch.sum(textb != self.padding_idx, dim=-1) #[batch_size]
        
        # Embedding
        embed_a = self.dropout(self.embed(texta)) #[batch_size, max_len_a, embed_size]
        embed_b = self.dropout(self.embed(textb)) #[batch_size, max_len_b, embed_size]
        
        # GRU Encoding
        # sentence a
        output_a = self.gru(embed_a.float(), lens_a) # [batch_size, seq_len, 2 * hidden_size]
        output_a = torch.max(output_a, dim=1)[0] # [batch_size, 2 * hidden_size]
        
        # sentence b
        output_b = self.gru(embed_b.float(), lens_b) # [batch_size, seq_len, 2 * hidden_size]
        output_b = torch.max(output_b, dim=1)[0] # [batch_size, 2 * hidden_size]
        
        # Similarity Computing
        sim = torch.cat([output_a, output_a * output_b, torch.abs(output_a - output_b), output_b], dim=1)
        # [batch_size, 8 * hidden_size]
        
        # FNN Layer
        return self.proj(sim)