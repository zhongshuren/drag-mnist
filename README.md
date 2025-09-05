# drag-mnist

A toy task for interactive diffusion models.

It is designed for beginners of interactive diffusion models who don't have access to high-grade GPU clusters.

The training can be done on a laptop gpu within few hours.

The inference can be performed at nearly 60 fps locally.

## Technical Details

### 1. Dataset


### 2. Model Architecture

It uses RNNs to store history information so the memory usage won't increase linearly as inference timestep increases.

### 3. Training pipeline

To obtain minimal latency, the model uses frame-wise autoregressive one-step diffusion for video generation.

To make this possible, we implement the recent SOTA method from one-step diffusion and autoregressive diffusion, namely, shortcut model and diffusion forcing. And put them to a unified, end-to-end training framework.

It turns out 1-NFE diffusion has great potential to be a stable interactive generative model.

## Training
```shell
python train.py
```

## Interactive Demo

Train the model first, or download the pretrained model and put it to /checkpoint.

```shell
python app.py
```

The demo will launch a web interface. You can drag mouse to move the MNIST sample in real time. See how long the model can hold shape consistency.

