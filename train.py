import torch
from torch.utils.data import DataLoader
from torch import optim, nn

from typing import Any
from tqdm import tqdm

from model import LlamaConfig, Llama, save_model
from dataset import get_dataloaders

def train(args:LlamaConfig, model:Llama, dataloader:DataLoader, device:str, loss_fn:nn, optimizer:optim, epoch:int, save_steps:int=5):
    model.train()
    losses = []
    
    for token, next_token in (pbar := tqdm(dataloader, desc="Training")):
        
        token, next_token = token.to(device), next_token.to(device)
        
        kv_cache = None
        pred, kv_cache = model(token, kv_cache) #(b, seq_len, vocab_size)
        loss = loss_fn(pred.view(-1, args.vocab_size), next_token.view(-1))
        pbar.set_postfix(Loss = loss.item())
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        losses.append(loss.item())
    
    if epoch % save_steps == 0:
        params = args.num_params
        args.model_name.format(params, epoch)
        save_model(model, args.model_name)
        
    return losses
        
if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    llamacomfig = LlamaConfig()
    train_dataset, val_dataset, vocab_size = get_dataloaders(llamacomfig)
    llamacomfig.vocab_size = vocab_size
    
    model = Llama(llamacomfig)
    model = model.to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters())
    
    train(llamacomfig, model, train_dataset, device, loss_fn, optimizer)
    
    