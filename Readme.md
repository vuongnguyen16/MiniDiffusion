# MiniDiffusion

A DDPM baseline for unconditional CIFAR-10 generation, trained for
**200,000 optimizer updates**. Based on
[w86763777/pytorch-ddpm](https://github.com/w86763777/pytorch-ddpm), an unofficial
PyTorch implementation of [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239).

This repository contains training, EMA sampling, and FID/Inception Score code.
Datasets, trained weights, generated images, and logs are created locally and
excluded from Git.

## Setup

Use **Python 3.11**. An NVIDIA GPU is recommended for full training; CPU
execution is available for small checks.

```bash
git clone https://github.com/vuongnguyen16/MiniDiffusion.git
cd MiniDiffusion
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS
source .venv/bin/activate
```

Install PyTorch, then the remaining dependencies:

```bash
python -m pip install --upgrade pip
# NVIDIA GPU, Linux / Windows, CUDA 12.8 wheel:
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

For CPU only, replace the PyTorch installation command with:

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
```

For other GPU/OS combinations, use the matching command in the
[PyTorch 2.7.1 installation instructions](https://pytorch.org/get-started/previous-versions/#v271),
then install `requirements.txt`. TensorFlow is not required.

## Train

```bash
python main.py --train --flagfile ./config/CIFAR10.txt
```

CIFAR-10 is downloaded automatically to `data/` on the first run. Training
starts from random initialization and stops after **exactly 200,000 updates**.
The `step` field in a new checkpoint counts completed optimizer updates.

| Setting | Default |
| --- | ---: |
| Training updates | 200,000 |
| Batch size | 128 |
| Image size | 32 x 32 |
| Diffusion steps | 1,000 |
| Learning rate | 0.0002 |
| Warmup updates | 5,000 |
| EMA decay | 0.9999 |
| Dropout | 0.1 |
| Random seed | 42 |

The default output directory is `logs/DDPM_CIFAR10_EPS/`:

```text
logs/DDPM_CIFAR10_EPS/
  ckpt.pt          # latest checkpoint, including the final update
  flagfile.txt     # saved model configuration and arguments
  sample/          # periodic EMA sample grids
  events.*         # TensorBoard events
```

Periodic saves replace `ckpt.pt`; earlier weights are not retained. The final
checkpoint is always saved, even with `--save_step=0`. An existing training
`sample/` directory is not overwritten: use a different `--logdir` for a new
experiment. Training does not automatically resume from a checkpoint.

Examples:

```bash
# Use a separate output directory.
python main.py --flagfile ./config/CIFAR10.txt --logdir ./logs/run2

# Reduce sampling overhead while retaining 200k training updates.
python main.py --flagfile ./config/CIFAR10.txt --sample_step 50000

# Reduce batch size for a smaller GPU; report this change in comparisons.
python main.py --flagfile ./config/CIFAR10.txt --batch_size 64 --logdir ./logs/batch64

# Use multiple visible CUDA GPUs; batch size is the total across devices.
python main.py --flagfile ./config/CIFAR10.txt --parallel
```

The upstream sampling interval is retained: 64 images every 1,000 updates. Each
image requires 1,000 denoising steps, so sampling can add substantial runtime.
Set `--sample_step=0` to disable periodic samples and sample the final checkpoint
separately.

## Sample the trained EMA model

```bash
python main.py --flagfile ./logs/DDPM_CIFAR10_EPS/flagfile.txt --notrain --sample --sample_size 64
```

This loads `ckpt.pt` and saves a grid to
`logs/DDPM_CIFAR10_EPS/samples_ema.png`. It does not compute FID/IS and does not
require reference statistics or Inception weights. `--seed` controls the sampling
randomness. The default device is CUDA when available, otherwise CPU; use
`--device=cpu` or `--device=cuda` to select explicitly.

With a training flagfile, include `--notrain` when sampling/evaluating. Exactly
one of `--train`, `--sample`, and `--eval` must be enabled.

## Evaluate FID and Inception Score

Download the upstream [CIFAR-10 training-set statistics](https://drive.google.com/drive/folders/1UBdzl6GtNMwNQ5U-4ESlIer43tNjiGJC?usp=sharing)
and place the file at:

```text
stats/cifar10.train.npz
```

Then run:

```bash
python main.py --flagfile ./logs/DDPM_CIFAR10_EPS/flagfile.txt --notrain --eval --num_images 50000
```

Evaluation generates **50,000 images from the EMA model**, computes FID/IS using
the upstream scoring implementation, and saves a preview grid to
`samples_ema.png`. Inception weights are downloaded automatically on the first
evaluation. Reference statistics are checked before generating the evaluation
images. `--fid_use_torch` optionally performs score statistics calculations on
the selected device.

FID is lower-is-better; Inception Score is higher-is-better. Scores from small
sample grids are not directly comparable to FID-50k. This 200k baseline does not
claim to reproduce the upstream 800k configuration's published scores.

## Files and upstream changes

```text
config/CIFAR10.txt       # 200k baseline configuration
main.py                 # train, EMA sampling, and evaluation
model.py                # upstream UNet architecture
diffusion.py            # forward diffusion and DDPM sampler
score/                  # upstream Inception/FID implementation
requirements.txt
LICENSE
```

The UNet and original training/scoring structure are retained. Changes include
the 200k default, completed-update numbering and final checkpoint save,
standalone sampling, device/seed selection, modern dependencies, EMA eval mode
during sampling, and sampler fixes (valid mean-type choices and clipping
predicted x0 before computing the posterior mean).

The upstream project's [WTFPL license](LICENSE) is retained. Credit to
[w86763777](https://github.com/w86763777) for the original implementation and
the cited sources in `score/` for the Inception/FID utilities.
