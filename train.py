import torch
from torch.utils.data import DataLoader
from torch import optim, nn

from typing import Any
from tqdm import tqdm
from argparse import ArgumentParser
import subprocess
import sys

from model import LlamaConfig, Llama, save_model
from dataset import get_dataloaders

def train(args:LlamaConfig, model:Llama, dataloader:DataLoader, device:str, loss_fn:nn, optimizer:optim, epoch:int, save_steps:int=2):
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
        
    return losses / len(dataloader)

def safety_function_for_device(device: str, args: LlamaConfig):
    try:
        params = (12 * (args.d_model ** 2) * args.n_blocks) + (args.d_model * args.vocab_size) 
        model_memory_gb = params * 4 / (1024 ** 3)
        print("Model Size:", model_memory_gb, "GB")

        if device == "cuda":
            if torch.cuda.device_count() > 2:
                return
                
            device_name = torch.cuda.get_device_name()
            device_props = torch.cuda.get_device_properties(0)
            total_memory_gb = device_props.total_memory / (1024 ** 3)
            
            print("=" * 5)
            print(f"You have {device_name} with {total_memory_gb:.2f} GB of memory.")
            
            if total_memory_gb < model_memory_gb:
                print(
                    f"Your device {device_name} does not have sufficient memory "
                    f"({total_memory_gb:.2f} GB) to load the whole model ({model_memory_gb} GB)."
                )
                sys.exit(1)
        
        elif device == "mps":
            result = subprocess.run(
                ["sysctl", "hw.memsize"], capture_output=True, text=True
            )
            if result.returncode != 0:
                print("Failed to retrieve memory information for MPS.")
                sys.exit(1)
            
            total_memory = int(result.stdout.split(":")[1].strip())
            total_memory_gb = total_memory / (1024 ** 3)
            print(f"You have MPS with {total_memory_gb:.2f} GB of memory.")
            
            if total_memory_gb < model_memory_gb:
                print(
                    f"Your MPS device does not have sufficient memory "
                    f"({total_memory_gb:.2f} GB) to load the whole model ({model_memory_gb} GB)."
                )
                sys.exit(1)
        
        else:
            print(f"Unsupported device type: {device}")
            sys.exit(1)

    except Exception as e:
        print("Error:", str(e))
        sys.exit(1)
        
if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    args = ArgumentParser(description="Take the Arguments of the Model, Traning.")
    args.add_argument('--epoch', type=int, help="Epochs", default=1)
    args.add_argument('--model', type=str, help="Which one 7B, 13B, 34B etc.", default="7B")
    args = args.parse_args()
    
    if args.model == "7B":
        llamaconfig = LlamaConfig(d_model=4096, num_heads=32, n_blocks=32, lr=3e-4, last_epoch=args.epoch)
    elif args.model == "13B":
        llamaconfig = LlamaConfig(d_model=5120, num_heads=40, n_blocks=40, lr=3e-4, last_epoch=args.epoch)
    elif args.model == "34B":
        llamaconfig = LlamaConfig(d_model=6656, num_heads=52, n_blocks=60, lr=1.5e-4, last_epoch=args.epoch)
    elif args.model == "70B":
        llamaconfig = LlamaConfig(d_model=8192, num_heads=64, n_blocks=80, lr=1.5e-4, last_epoch=args.epoch)
    elif args.model == "T":
        llamaconfig = LlamaConfig(d_model=768, num_heads=8, n_blocks=4, lr=3e-4, last_epoch=args.epoch)
    else:
        raise ValueError(f"Given args for model is not support {args.model}")
        
    safety_function_for_device(device, llamaconfig)
    
    train_dataset, val_dataset, vocab_size = get_dataloaders(llamaconfig)
    llamaconfig.vocab_size = vocab_size
    
    model = Llama(llamaconfig)
    model = model.to(device)
    
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=llamaconfig.lr)
    losses = []
    
    for epoch in range(llamaconfig.last_epoch):
        print(f"Epoch {epoch+1}/{llamaconfig.last_epoch}")
        loss = train(llamaconfig, model, train_dataset, device, loss_fn, optimizer, epoch)
        losses.append(loss)
        