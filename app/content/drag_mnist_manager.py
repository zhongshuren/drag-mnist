import time

import cv2
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
from threading import Thread

from .base import ContentManager
from dataset.drag_mnist import DragMNIST
from utils import add_mask


# used to test the app interface
class FakeModel(nn.Module):
    def __init__(self):
        super(FakeModel, self).__init__()
        self.init_image = torch.zeros(28, 28)
        self.accumulated_dxdy = torch.zeros(2)

    def forward(self, dxdy):
        self.accumulated_dxdy += dxdy * 2

        x = (torch.arange(0, 28) / 14 - 1).unsqueeze(1).repeat(1, 28) - self.accumulated_dxdy[0]
        y = (torch.arange(0, 28) / 14 - 1).unsqueeze(0).repeat(28, 1) - self.accumulated_dxdy[1]

        grid = torch.stack([x, y], dim=-1).unsqueeze(0)

        out = F.grid_sample(self.init_image, grid, padding_mode='reflection')

        out = add_mask(out)
        return out


class DragMNISTManager(ContentManager):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.resize_w = config.resize_w
        self.resize_h = config.resize_h
        self.use_model = config.use_model
        self.generate_input = config.generate_input
        self.dataset = DragMNIST(config)

        if self.use_model:
            self.model = None
        else:
            self.model = FakeModel()

        self.reset_state()

        if self.generate_input:
            def _run_loop():
                while True:
                    ctrl = self.dataset.sample_ctrl()[0]
                    self.update_state(ctrl)
                    time.sleep(1 / 60)
            self.thread = Thread(target=_run_loop)
            self.thread.daemon = True
            self.thread.start()

    def init_image(self):
        return self.dataset.sample_init(mask=self.use_model)

    def render(self):
        dxdy = self.new_xy - self.xy
        self.xy = self.new_xy
        image = self.model(dxdy)
        image = rearrange(image, '1 1 h w -> h w').numpy()
        frame = cv2.resize(image, (self.resize_w, self.resize_h), interpolation=cv2.INTER_NEAREST)
        return (frame * 255).astype(np.uint8)

    # receive data from the api wrapper
    def update_state(self, data):
        if self.generate_input:
            dxdy = data
        else:
            dxdy = np.array(data['dxdy'])
        self.new_xy = self.xy + dxdy
        return

    def reset_state(self):
        self.model.init_image = self.init_image()
        self.xy = np.array([0., 0.])
        self.new_xy = np.array([0., 0.])