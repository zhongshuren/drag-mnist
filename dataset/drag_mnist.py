import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from einops import rearrange
from utils import add_mask

image_file_train = 'train-images.idx3-ubyte'
image_file_test = 't10k-images.idx3-ubyte'


class DragMNIST(Dataset):
    def __init__(self,
                 configs,
                 mode: str='test',
                 seq_len: int=64,):
        self.configs = configs
        image_file = image_file_train if mode == 'train' else image_file_test
        self.data_file = f'{self.configs.dataset_dir}/{image_file}'
        self.sample_size = 60000 if mode == 'train' else 100
        self.seq_len = seq_len
        self.images = self._read_data()

    def __len__(self):
        return self.sample_size

    def __getitem__(self, idx):
        init = self.sample_init(idx, mask=False)
        img_seq = []
        ctrl_seq = torch.FloatTensor(self.sample_ctrl(length=self.seq_len))
        self.accumulated_dxdy = torch.zeros(2)
        for i in range(self.seq_len):
            self.accumulated_dxdy += ctrl_seq[i] * 2

            x = (torch.arange(0, 28) / 14 - 1).unsqueeze(1).repeat(1, 28) - self.accumulated_dxdy[0]
            y = (torch.arange(0, 28) / 14 - 1).unsqueeze(0).repeat(28, 1) - self.accumulated_dxdy[1]

            grid = torch.stack([x, y], dim=-1).unsqueeze(0)

            out = F.grid_sample(init, grid, padding_mode='reflection')
            out = add_mask(out)
            out = rearrange(out, '1 1 h w -> 1 h w')

            img_seq.append(out)

        img_seq = torch.stack(img_seq, dim=0) # t, 1, h, w

        return img_seq, ctrl_seq

    def _read_data(self):
        with open(self.data_file, 'rb') as f:
            image_set = f.read()
        image = []
        for i in range(0, self.sample_size):
            tmp_image = np.array([item for item in image_set[16 + i * 784: 16 + 784 + i * 784]],
                                 dtype=np.float32).reshape(28, 28)
            tmp_image /= 255
            image.append(tmp_image)
        return image

    def sample_init(self, idx=None, mask=True):
        if idx is None:
            idx = np.random.choice(self.sample_size, 1).item()
        ret = self.images[idx]
        if mask:
            ret = add_mask(ret)
        return rearrange(torch.tensor(ret), 'h w -> 1 1 h w')

    def sample_ctrl(self, length=1):
        return np.random.randn(length, 2) * 0.01