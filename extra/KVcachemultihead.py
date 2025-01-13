import torch
from torch import nn
import math

class MHAwithKVCache(nn.Module):
    def __init__(self, d_model, head_size):
        super().__init__()
        
        self.d_model = d_model
        self.head_size = head_size
        self.d_k = d_model // head_size
        
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, q:torch.Tensor, k:torch.Tensor, v:torch.Tensor, kv_cache=None):
        b = q.shape[0]
        
        query = self.Wq(q) #(b, seq, d_m)
        key = self.Wk(k)
        value = self.Wv(v)
        
        query = query.view(b, -1, self.head_size, self.d_k).transpose(-3, -2) # (b, head, seq, d_k)
        key = key.view(b, -1, self.head_size, self.d_k).transpose(-3, -2)
        value = value.view(b, -1, self.head_size, self.d_k).transpose(-3, -2)
        
        if kv_cache is not None:
            c_k, c_v = kv_cache
            key = torch.cat([c_k, key], dim=2)
            value = torch.cat([c_v, value], dim=2)
        
        attention = query @ key.transpose(-2, -1) / math.sqrt(self.d_k) # (b, head, seq, seq)
        mask = torch.tril(torch.ones(attention.shape[-2:], device=q.device)).unsqueeze(0).unsqueeze(0)
        attention.masked_fill_(mask == 0, float('-inf'))
        
        attention = attention.softmax(dim=-1)
        
        out = attention @ value #(b, head, seq, d_k)
        out = out.transpose(1, 2).contiguous().view(b, -1, self.d_model)
        out = self.Wo(out)
        return out, (key, value)
    
if __name__ == "__main__":
    b = 1
    seq_len = 6
    d_model = 512
    head_size = 8
    
    m = MHAwithKVCache(d_model, head_size)
    i = torch.rand(b, seq_len, d_model)
    kv_cache = None
    
    out, kv_cache = m(i, i, i, kv_cache)
    
    next_word = torch.rand(b,1,d_model)
    out2, kv_cache = m(next_word, next_word, next_word, kv_cache)
    
    print("First output:", out.shape)
    print("Using the old KV-cache",out2.shape)
        
