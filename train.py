import torch
import os
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

drag_mnist = DragMNIST(configs,
                       seq_len=configs.dataset.seq_len,
                       mode=configs.dataset.mode,
                       num_samples=configs.dataset.num_samples,)
data = DataLoader(drag_mnist, batch_size=configs.training.batch_size, shuffle=True)
model = DiT(dim=configs.model.dim,
            in_channels=configs.model.in_channels,
            input_size=configs.model.input_size,
            patch_size=configs.model.patch_size,
            depth=configs.model.depth,)
diffusion_loss = DiffusionForcingLoss(model)

optimizer = AdamW(model.parameters(), lr=configs.training.lr)
scheduler = CosineAnnealingLR(optimizer, T_max=configs.training.epochs, eta_min=1e-5)
model, optimizer, data = accelerator.prepare(model, optimizer, data)


if __name__ == '__main__':
    info = {'train_loss': '#.###'}
    for epoch in range(configs.training.epochs):

        model.train()
        train_loss = 0
        p_bar = tqdm(total=len(data), postfix=info, )
        for i, (x, ctrl_seq) in enumerate(data):
            x = x.to(device)
            ctrl_seq = ctrl_seq.to(device)
            loss = diffusion_loss(x, c=ctrl_seq)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            p_bar.update(1)
            info['train_loss'] = '%.3f' % (train_loss / (i + 1))
            p_bar.set_postfix(info, refresh=True)
        p_bar.close()

        model.eval()
        with torch.no_grad():
            num_eval = configs.training.num_eval
            eval_seq_len = configs.training.eval_seq_len
            past_h = torch.zeros(configs.model.depth, 1, 1, configs.model.dim).to(device)# depth, 1, B, D
            past_x = drag_mnist.sample_init().to(device).unsqueeze(1)
            x_seq = [past_x]
            for i in range(eval_seq_len):
                noise = torch.randn_like(past_x)
                ctrl = drag_mnist.sample_ctrl().to(device)
                x, h = model(noise, c=ctrl, h=past_h, past_x=x_seq[-1])
                x_seq.append(x)
                past_x = x
                past_h = h

        # save checkpoint
        accelerator.wait_for_everyone()
        if (epoch + 1) % configs.training.checkpoint_every == 0 or epoch + 1 == configs.training.epochs:
            model_weights = model.state_dict()
            checkpoint = {
                "model": model_weights,
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": scheduler.state_dict(),
            }
            os.makedirs(configs.training.ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(configs.training.ckpt_dir, f"ckpt_{(epoch+1):04}.pt")
            accelerator.save(checkpoint, ckpt_path)
            print(f"Saved checkpoint at step {epoch} to {os.path.abspath(ckpt_path)}")
            torch.cuda.empty_cache()
            accelerator.free_memory()
            accelerator.wait_for_everyone()

        scheduler.step()

