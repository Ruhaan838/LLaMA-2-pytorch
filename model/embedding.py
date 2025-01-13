import torch

def rotary_embedding(x):
    b, seq_len, d_model = x.shape

    position_ids = torch.arange(0, seq_len, dtype=torch.float, device=x.device).reshape(1, -1) 
    indices = torch.arange(0, d_model // 2, dtype=torch.float, device=x.device)
    theta = 10000.0 ** (-indices / (d_model // 2))  
    rotary_emb = torch.matmul(position_ids.transpose(0, 1), theta.reshape(1, -1)).to(x.dtype) 

    cos_emb = torch.cos(rotary_emb)
    sin_emb = torch.sin(rotary_emb)

    x_reshape = x.reshape(b, seq_len, d_model // 2, 2)
    x_rotated = torch.stack(
        [
            x_reshape[..., 0] * cos_emb - x_reshape[..., 1] * sin_emb,
            x_reshape[..., 1] * cos_emb + x_reshape[..., 0] * sin_emb,
        ],
        dim=-1,
    )
    x_rotated = x_rotated.reshape(b, seq_len, d_model)

    return x_rotated
