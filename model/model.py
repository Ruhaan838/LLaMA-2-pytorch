import torch
from torch import Tensor
from torch import nn

from torch.nn import functional as F
import math

from typing import Tuple, Optional
from .config import LlamaConfig

from .embedding import rotary_embedding

typed = Tuple[Tensor, Tensor]

class RMSNorm(nn.Module):
    def __init__(self, args:LlamaConfig) -> None:
        super().__init__()
        
        self.eps = args.eps
        self.weight = nn.Parameter(torch.ones(args.d_model))
        
    def _norm(self, x:Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x:Tensor) -> Tensor:
        out = self._norm(x.float()).type_as(x)
        return out * self.weight
    
class Attention(nn.Module):
    def __init__(self, args:LlamaConfig) -> None:
        super().__init__()
        
        assert args.d_model % args.num_heads == 0, "d_model is must divisible by num_heads"
        assert args.num_heads % args.groups == 0, "num_heads is must divisible by the groups"
        
        self.d_model = args.d_model
        self.num_heads = args.num_heads
        self.groups = args.groups
        
        self.Wq = nn.Linear(self.d_model, self.d_model)
        self.Wk = nn.Linear(self.d_model, self.d_model)
        self.Wv = nn.Linear(self.d_model, self.d_model)
        
        self.Wo = nn.Linear(self.d_model, self.d_model)
        
        self.d_k = self.d_model // self.num_heads
        self.heads_per_groups = self.num_heads // self.groups
        
    def _reshape_qkv(self, x:Tensor) -> Tensor:
        b, seq_len, d_model = x.size()
        
        x = x.view(b, seq_len, self.num_heads, self.d_k)
        x = x.view(b, seq_len, self.groups, self.heads_per_groups, self.d_k)
        x = x.permute(0, 2, 3, 1, 4)
        return x
    
    def forward(self, q:Tensor, k:Tensor, v:Tensor, kv_cache:Optional[typed] = None) -> Tuple[Tensor, typed]:
        batch_size = q.shape[0]
        
        query = self._reshape_qkv(self.Wq(q))
        key = self._reshape_qkv(self.Wk(k))
        value = self._reshape_qkv(self.Wv(v))
        
        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            key = torch.cat([cache_k, key], dim=3)
            value = torch.cat([cache_v, value], dim=3)
        
        attention = query @ key.transpose(-2, -1) / math.sqrt(self.d_k)
        cacusal_mask = torch.tril(torch.ones(key.shape[-2], key.shape[-2], device=q.device))
        cacusal_mask = cacusal_mask[..., :attention.shape[-2], :attention.shape[-1]]
        attention = attention.masked_fill_(cacusal_mask == 0, float('-inf'))
        attention = attention.softmax(dim=-1)
        
        out = attention @ value
        out = out.permute(0, 3, 1, 2, 4).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        
        return self.Wo(out), (key, value)
    
class FeedForward(nn.Module):
    def __init__(self, args:LlamaConfig):
        super().__init__()
        
        hidden_dim = args.d_model * 4
        d_model = args.d_model
        self.W1 = nn.Linear(d_model, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, d_model)
        self.W3 = nn.Linear(d_model, hidden_dim)
        
    def forward(self, x:Tensor) -> Tensor:
        return self.W2(F.silu(self.W1(x)) * self.W3(x))

class TransformerBlock(nn.Module):
    def __init__(self, args:LlamaConfig):
        super().__init__()
        
        self.attention = Attention(args)
        self.attention_norm = RMSNorm(args)
        self.feed_forward = FeedForward(args)
        self.ff_norm = RMSNorm(args)
    
    def forward(self, x:Tensor, kv_cache:typed) -> typed:
        
        res = x
        x = self.attention_norm(x) # norm
        q, k = rotary_embedding(x), rotary_embedding(x) #rotary embed
        x, kv_cache = self.attention(q, k, x, kv_cache) #cached attetnion
        x = x + res
        x = x + self.feed_forward(self.ff_norm(x)) # feed-forward
        return x, kv_cache
    
class Llama(nn.Module):
    def __init__(self, args:LlamaConfig):
        super().__init__()
        
        self.embedding = nn.Embedding(args.vocab_size, args.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(args) for _ in range(args.n_blocks)])
        self.final_norm = RMSNorm(args)
        self.output_layer = nn.Linear(args.d_model, args.vocab_size, bias=False)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x:Tensor, kv_cache:typed = None) -> Tensor:
        
        x = self.embedding(x)
        for layer in self.blocks:
            x, kv_cache = layer(x, kv_cache)

        x = self.final_norm(x)
        x = self.output_layer(x)
        
        return x, kv_cache
    