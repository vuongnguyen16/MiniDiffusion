# Pretrained CIFAR-10 checkpoint: 200k updates

This folder contains the configuration and downloader for the trained
**200,000-update EMA model**. Training is not required to use it.

From the repository root, after installing `requirements.txt`:

```bash
python checkpoints/200k/download.py
python main.py --flagfile ./checkpoints/200k/flagfile.txt --sample
```

The first command downloads `ckpt.pt` into this folder and verifies its size
and SHA-256. The second generates 64 images and writes a grid to
`checkpoints/200k/samples_ema.png`. CUDA is used when available; CPU also works
but 1,000-step sampling is slow.

```text
checkpoints/200k/
  README.md
  download.py
  flagfile.txt
  metadata.json
  ckpt.pt             # downloaded locally, excluded from Git
  samples_ema.png      # generated locally, excluded from Git
```

## Download directly

- [200k EMA checkpoint](https://github.com/vuongnguyen16/MiniDiffusion/releases/download/cifar10-200k/ckpt.pt)
- [Release and all assets](https://github.com/vuongnguyen16/MiniDiffusion/releases/tag/cifar10-200k)

If downloading manually, save the asset as `checkpoints/200k/ckpt.pt`.
The weights are hosted as a Release asset because the file exceeds GitHub's
100 MiB limit for regular Git files; cloning the source stays small.

## Model details

| Property | Value |
| --- | --- |
| Dataset | CIFAR-10 train split, 32 x 32 RGB |
| Completed optimizer updates | 200,000 |
| Weights | EMA, decay 0.9999 |
| Stored tensor precision | FP32, unchanged from the training checkpoint |
| Training precision | BF16 mixed precision |
| DDPM diffusion steps | 1,000 |
| Mean / variance parameterization | epsilon / fixedlarge |
| File size | 143,593,856 bytes (about 137 MiB) |
| SHA-256 | `0c8fe72c7fb3d53aa0b194a5663d9bc6e7992619ca1275c2dbdaf55b9099bb76` |

This is an inference checkpoint containing `ema_model`, `step`, and `config`.
EMA tensors are bitwise equal to the completed training run's FP32 EMA tensors.
The optimizer, raw training model and RNG state are omitted to reduce download
size. The original full training checkpoint remains separate. This asset is
intended for sampling/evaluation, not resuming the optimizer state.

This checkpoint was trained with BF16 mixed precision; training the standalone
FP32 implementation again is not expected to reproduce the same tensors bit
for bit. The UNet state dictionary loads strictly into this repository's model.
No upstream FID-50k score is claimed for this checkpoint.

For FID/IS evaluation, download the reference statistics described in the root
README, then select evaluation mode:

```bash
python main.py --flagfile ./checkpoints/200k/flagfile.txt --nosample --eval --num_images 50000
```
