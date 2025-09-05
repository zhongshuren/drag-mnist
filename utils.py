def add_mask(x):
    x[:, :, 13:15] = 0.5
    x[:, :, :, 13:15] = 0.5
    return x