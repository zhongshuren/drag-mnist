import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class DiffusionForcingLoss(nn.Module):
    def __init__(self, model, cfg=False, cfg_w=1.0, cfg_dropout=0.9, num_labels=0):
        super(DiffusionForcingLoss, self).__init__()
        self.model = model
        self.cfg = cfg
        self.cfg_w = cfg_w
        self.cfg_dropout = cfg_dropout
        self.num_labels = num_labels

        self.past = lambda x: torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        self.v = lambda x1, t1, t2, c, past_x: self.model(x1, t1, t2, c, past_x=self.past(past_x))[0] - x1
        self.usq = lambda x: rearrange(x, 'b t c -> b t c 1 1')

    def sample_t(self, x):
        B, T = x.shape[:2]
        samples = torch.rand(B, T, 3, device=x.device)
        samples = 3 * samples ** 2 - 2 * samples ** 3
        t0 = samples[:, :, 0:1]
        t1 = torch.minimum(samples[:, :, 1:2], samples[:, :, 2:3])
        t2 = torch.maximum(samples[:, :, 1:2], samples[:, :, 2:3])
        tm = (t1 + t2) / 2
        return t0, t1, t2, tm

    def forward(self, x, c=None):
        t0, t1, t2, tm = self.sample_t(x)
        e = torch.randn_like(x)
        z0, z1, z2 = torch.lerp(e, x, self.usq(t0)), torch.lerp(e, x, self.usq(t0)), torch.lerp(e, x, self.usq(t2))
        v_tgt = x - e

        if self.cfg and c is not None:
            uncond = torch.ones_like(c) * self.num_labels
            cfg_mask = torch.rand(*c.shape, device=c.device) < self.cfg_dropout
            c = torch.where(cfg_mask, c, uncond)
            v_uncond = self.v(z0, t0, t0, uncond, z0)
            v_tgt = torch.where(cfg_mask.unsqueeze(-1), v_tgt * self.cfg_w + v_uncond * (1 - self.cfg_w), v_tgt)

        v_t0 = self.v(z0, t0, t0, c, z2)
        v_t1t2 = self.v(z1, t1, t2, c, z2)
        v_t1tm = self.v(z1, t1, tm, c, z2)
        v_tmt2 = self.v(z1 + v_t1tm * self.usq(tm - t1), tm, t2, c, z2)
        v_t1tmt2 = v_t1tm + v_tmt2

        loss1 = F.mse_loss(v_t0, v_tgt) # Flow Matching loss
        loss2 = F.mse_loss(v_t1t2 * 2, v_t1tmt2.detach())   # shortcut loss

        loss = loss1 + loss2
        return loss