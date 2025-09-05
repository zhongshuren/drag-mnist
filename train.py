import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from dataset.drag_mnist import DragMNIST
from model import DiT
from algorithm import DiffusionForcingLoss
from accelerate import Accelerator

configs = OmegaConf.load('config/config_train.yaml')
accelerator = Accelerator()
device = accelerator.device
torch.manual_seed(configs.seed)

data = DataLoader(DragMNIST(configs, seq_len=64), batch_size=configs.batch_size, shuffle=True)
model = DiT(dim=configs.dim)
diffusion_loss = DiffusionForcingLoss(model)

optimizer = AdamW(model.parameters(), lr=configs.lr)
scheduler = CosineAnnealingLR(optimizer, T_max=configs.epochs, eta_min=1e-5)
model, optimizer, data = accelerator.prepare(model, optimizer, data)


if __name__ == '__main__':
    info = {'train_loss': '#.###'}
    for epoch in range(configs.epochs):
        p_bar = tqdm(total=len(data), postfix=info, )
        model.train()
        for i, x in enumerate(data):
            x = x.to(device)
            loss = diffusion_loss(x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            p_bar.update(1)
            info['train_loss'] = '%.3f' % loss.item()
            p_bar.set_postfix(info, refresh=True)
        p_bar.close()

        num_eval = configs.num_eval
        eval_time_steps = 256
        model.eval()
        with torch.no_grad():
            past_h = torch.zeros(2, num_eval, configs.dim).to(device)
            past_x = torch.zeros(num_eval, 1, 2).to(device)
            x_seq = []
            for i in range(eval_time_steps):
                if configs.flow_steps == 1:
                    e = torch.randn(num_eval, 1, 2)
                    x, h = model(e, h=past_h, past_x=past_x)
                x_seq.append(x)
                past_x = x
                past_h = h

        scheduler.step()
