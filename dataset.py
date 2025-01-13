import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from model import LlamaConfig

class CharDataset(Dataset):
    def __init__(self, titles, blog_posts, block_size):

        texts = [f"{title} ### {post}" for title, post in zip(titles, blog_posts)]
        
        chars = sorted(list(set(''.join(texts))))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        
        data = ''.join(texts)
        self.encoded_data = torch.tensor(self.encode(data), dtype=torch.long)
        self.block_size = block_size

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, l):
        return ''.join(self.itos[i.item() if torch.is_tensor(i) else i] for i in l)

    def __len__(self):
        return len(self.encoded_data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.encoded_data[idx:idx + self.block_size]
        target = self.encoded_data[idx + 1:idx + self.block_size + 1]
        return chunk, target


class TokenizedDataset(Dataset):
    def __init__(self, titles, blog_posts, tokenizer, block_size):

        texts = [f"{title} ### {post}" for title, post in zip(titles, blog_posts)]
        
        self.tokenizer = tokenizer
        tokenized = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=block_size
        )
        self.input_ids = tokenized["input_ids"]
        self.block_size = block_size

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        chunk = self.input_ids[idx, :-1]
        target = self.input_ids[idx, 1:]
        return chunk, target


def get_dataloaders(args:LlamaConfig, use_tokenizer=False):
    
    block_size = args.block_size
    batch_size = args.batch_size
    num_workers = args.num_workers
    
    data = load_dataset("fdaudens/hf-blog-posts-split")
    train, val = data['train'], data['test']
    
    if use_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        train_dataset = TokenizedDataset(train['targets'], train['inputs'], tokenizer, block_size)
        val_dataset = TokenizedDataset(val['targets'], val['inputs'], tokenizer, block_size)
        vocab_size = tokenizer.vocab_size
    else:
        train_dataset = CharDataset(train['targets'], train['inputs'], block_size)
        val_dataset = CharDataset(val['targets'], val['inputs'], block_size)
        vocab_size = train_dataset.vocab_size
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, vocab_size


if __name__ == "__main__":
    llamaconfig = LlamaConfig()
    use_tokenizer = False
    train_loader, val_loader, vocab_size = get_dataloaders(llamaconfig, use_tokenizer=use_tokenizer)
    print(f"Vocabulary Size: {vocab_size}")
    
    for batch in train_loader:
        x, y = batch
        print(f"Input shape: {x.shape}, Target shape: {y.shape}")
        break
