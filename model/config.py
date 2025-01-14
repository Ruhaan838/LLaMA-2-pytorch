
from dataclasses import dataclass
import torch

@dataclass
class LlamaConfig:
    d_model: int = 512
    eps: float = 1e-6
    num_heads: int = 8
    groups: int = 4
    vocab_size:int = 300
    n_blocks:int = 2
    
    block_size:int = 6
    batch_size:int = 8
    num_workers:int = 0
    
    num_params:str = "7B"
    model_name:str = "checkpoint/llama-{}-{}"
    last_epoch:int = 10
    lr :float =0.001
    
def load_from_checkpoint(model, path:str):
    model.load_state_dict(torch.load(path, weights_only=True))
        
def save_model(model, name:str):
    torch.save(model.state_dict(), name)
