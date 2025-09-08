import torch

def add_mask(x):
    with torch.no_grad():
        x[:, :, 13:15] = 0.5
        x[:, :, :, 13:15] = 0.5
    return x