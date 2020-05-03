import torch
import torch.nn as nn

class ABCNN_2(nn.Module):
    def __init__(self, seq_len, vec_dim, width):
        super(ABCNN_2, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(width, 1),
                             padding=(width-1, 0), stride=1)
        self.tanh = nn.Tanh()
        self.avg_pool = nn.AvgPool2d(kernel_size=(width, 1), padding=0, stride=1)
        self.seq_len = seq_len
        self.vec_dim = vec_dim
        self.width = width
    
    def _match_score(self, F0r, F1r):
        batch_size, _, cur_len, _ = F0r.shape
        F0r = F0r.squeeze(1).unsqueeze(2).repeat(1, 1, cur_len, 1) # [batch_size, seq_len, seq_len, vec_dim]
        F1r = F1r.repeat(1, cur_len, 1, 1) # [batch_size, seq_len, seq_len, vec_dim]
        a = F0r - F1r
        a = 1.0 + torch.norm(a, dim=-1, p=2) # [batch_size, seq_len, seq_len]
        return 1.0 / a
    
    def _attention(self, F0, F1):
        """
        Input:
            F0 size [batch_size, 1, seq_len, vec_dim]
            F1 size [batch_size, 1, seq_len, vec_dim]
        """
        A = self._match_score(F0, F1) # [batch_size, seq_len, seq_len]
        a0j = torch.sum(A, dim=-1).unsqueeze(2) # row-wise sum [batch_size, seq_len, 1]
        a1j = torch.sum(A, dim=1).unsqueeze(2) # col-wise sum [batch_size, seq_len, 1]
        F0 = F0.squeeze(1) # [batch_size, seq_len, vec_dim]
        F1 = F1.squeeze(1) # [batch_size, seq_len, vec_dim]
        F0 = F0 * a0j
        F1 = F1 * a1j
        return F0, F1
    
    def forward(self, F0r, F1r, sent0_mask, sent1_mask):
        """
        Input:
            (1) F0r: [batch_size, seq_len, vec_dim]
            (2) F1r: [batch_size, seq_len, vec_dim]
            (3) sent0_mask: [batch_size, seq_len]
            (4) sent1_mask: [batch_size, seq_len]
        """
        sent0_mask = sent0_mask.unsqueeze(2).expand_as(F0r) # [batch_size, seq_len, vec_dim]
        sent1_mask = sent1_mask.unsqueeze(2).expand_as(F1r) # [batch_size, seq_len, vec_dim]
        F0r = (F0r * sent0_mask).unsqueeze(1) # [batch_size, 1, seq_len, vec_dim]
        F1r = (F1r * sent1_mask).unsqueeze(1) # [batch_size, 1, seq_len, vec_dim]
        F0r = self.tanh(self.conv(F0r)) # [batch_size, 1, seq_len + width - 1, vec_dim]
        F1r = self.tanh(self.conv(F1r)) # [batch_size, 1, seq_len + width - 1, vec_dim]
        F0r, F1r = self._attention(F0r, F1r) # [batch_size, seq_len + width - 1, vec_dim]
        F0r = F0r.unsqueeze(1) # [batch_size, 1, seq_len + width - 1, vec_dim]
        F1r = F1r.unsqueeze(1) # [batch_size, 1, seq_len + width - 1, vec_dim]
        out0 = self.avg_pool(F0r).squeeze(1)
        out1 = self.avg_pool(F1r).squeeze(1)
        return out0, out1