import torch
from torch import nn
import math

class MultiheadAttention(nn.Module):
    def __init__(self, d_model, head_size):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // head_size
        self.head_size = head_size
        
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, q:torch.Tensor, k:torch.Tensor, v:torch.Tensor):
        b, seq_l, d_m = q.size()
        
        query = self.Wq(q) # b, seq, d_m
        key = self.Wk(k)
        value = self.Wv(v)
        
        query = query.view(b, seq_l, self.head_size, self.d_k).transpose(-3, -2) #b, head_size, seq_l, d_k
        key = key.view(b, seq_l, self.head_size, self.d_k).transpose(-3, -2) #b, head_size, seq_l, d_k
        value = value.view(b, seq_l, self.head_size, self.d_k).transpose(-3, -2) #b, head_size, seq_l, d_k
        
        attention = query @ key.transpose(-2, -1) / math.sqrt(self.d_k)   # b, head_size, seq, seq
        mask = torch.tril(torch.ones(seq_l, seq_l, device=q.device)).unsqueeze(0).unsqueeze(0)
        attention.masked_fill_(mask == 0, float('-inf'))
        attention = attention.softmax(dim=-1) # b, head_size, seq, seq
        
        out = attention @ value # b, head_size, seq, d_k
        out = out.transpose(-2, -3).contiguous().view(b, seq_l, d_m) # b, seq, d_m

        return self.Wo(out)
    
if __name__ == "__main__":
    i = torch.rand(1, 6, 512)
    m = MultiheadAttention(512, 8)
    out = m(i, i, i)
    print(out.shape)