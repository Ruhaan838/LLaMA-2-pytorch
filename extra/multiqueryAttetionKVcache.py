import torch
from torch import nn

import math

class MQAwithKVCache(nn.Module):
    def __init__(self, d_model, num_heads, groups):
        super().__init__()
        
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.groups = groups
        self.head_per_grp = num_heads // groups
        
    def _reshape_qkv(self, x):
        b, seq_len, _ = x.shape
        x = x.view(b, seq_len, self.num_heads, self.d_k) # (b, seq, head_size, dk)
        x = x.view(b, seq_len, self.groups, self.head_per_grp, self.d_k)# (b, seq, grups, head_per_grp, dk)
        x = x.permute(0, 2, 3, 1, 4) # (b, grups, head_pre_grp, seq, dk)
        return x

    def forward(self, q:torch.Tensor, k:torch.Tensor, v:torch.Tensor, mask = None, kv_cache=None):
        
        b = q.shape[0]
        
        query = self.Wq(q) # (b, seq, d_model)
        key = self.Wk(k)
        value = self.Wv(v)
        
        query = self._reshape_qkv(query) #  (b, grups, head_pre_grp, seq, dk)
        key = self._reshape_qkv(key) # (b, grups, head_pre_grp, seq, dk)
        value = self._reshape_qkv(value)  # (b, grups, head_pre_grp, seq, dk)
        
        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            key = torch.cat([cache_k, key], dim=3)
            value = torch.cat([cache_v, value], dim=3)
        
        attention = query @ key.transpose(-2, -1) / math.sqrt(self.d_k)  # (b, grups, head_pre_grp, seq, seq)
        
        if mask:
            causal_mask = torch.tril(torch.ones(key.shape[-2], key.shape[-2], device=q.device)) 
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(1).unsqueeze(1)
            attention = attention.masked_fill(causal_mask[:, :, :, -attention.shape[-2]:, :] == 0, float('-inf'))
            
        attention = attention.softmax(dim=-1)
        
        out = attention @ value  # (b, grups, head_pre_grp, seq, dk)
        out = out.permute(0, 3, 1, 2, 4).contiguous().view(b, -1, self.d_model)
        out = self.Wo(out)
        return out, (key, value)
    
if __name__ == "__main__":
    i = torch.rand(1, 6, 512)
    m = MQAwithKVCache(512, 8, 4)
    kv_cache = None
    out, kv_cache = m(i, i, i, True, kv_cache=kv_cache)
    print(out.shape)
    
    next_word = torch.rand(1, 1, 512)
    out2, kv_cache = m(next_word, next_word, next_word, True, kv_cache=kv_cache)
    print(out2.shape)