from model import Llama, load_from_checkpoint, LlamaConfig

import torch
from torch.nn import functional as F

@torch.inference_mode()
def generate(model:Llama, args:LlamaConfig,context:torch.Tensor, new_tokens_len:int, temperature:int=1.0, top_k:int=None):
    
    params = args.num_params
    args.model_name.format(params, args.last_epoch)
    load_from_checkpoint(model, args.model_name)
    
    for _ in range(new_tokens_len):
        idx_cond = context if context.size(1) <= args.block_size else context[:, -args.block_size:]
        
        kv_cache = None
        logits, kv_cache = model(idx_cond, kv_cache)
        logits = logits[:, -1, :] / temperature
        
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")
            
        prob = F.softmax(logits, dim=-1)
        next_word = torch.multinomial(prob, num_samples=1)
        context = torch.cat((context, next_word), dim=1)
        
    return context

