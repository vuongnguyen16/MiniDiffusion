# Reverse-noise ablation

Compare DDPM `lambda=1`, intermediate `lambda=0.5` / `lambda=0.1`, and mean-only `lambda=0`, using the **same saved x_T
bank**, EMA weights and all 1,000 timesteps. Lambda multiplies the **standard
deviation**. Mean-only is not DDIM and is not a timestep acceleration method.
No training is performed. The original `main.py` entrypoints remain available.

The qualitative supplement probes `lambda=0.75, 0.9, 1.1, 1.25` with **128 images
per new condition**, reusing the first 128 evaluation inputs and the saved
`lambda=1` controls. It computes no FID/IS and does not extend the 5,000-image runs.

The source base is `f38fbe8ae1a465578b438ecdc3d058aa383981b7`. The runner records
SHA-256 hashes of source files, checkpoint and reference statistics. It requires
the 200k checkpoint, complete training config, epsilon/fixedlarge, linear beta
1e-4 to .02, RGB 32x32, float32, eval mode, deterministic operations and disabled
TF32. `model.py`, trainer and score implementation are unchanged.

## Saved report

The completed experiment is in [ablation/reverse_noise/ema200k](ablation/reverse_noise/ema200k/report_vi.md).
Its compact layout contains `report_vi.md`, `metrics.csv`, `figures/` and
`reproducibility.zip`. Metadata, tensor chunks, input banks, test logs and
reference statistics are inside the ZIP. Extract it into the run directory to
inspect the evidence or recalculate metrics; no new sampling is needed.
Reports, CSV and figures are tracked on GitHub. Reproducibility ZIPs stay local
and are excluded by `.gitignore`; they are not downloadable from the repository.
The report compares four conditions at N=5,000 each: lambda=1, 0.5, 0.1 and 0.
The supplement reuses the exact original input banks, checkpoint, reverse seeds
and Inception/reference pipeline; full-batch baseline replay on both sampling
GPUs matched the stored controls exactly. Original 1/0 results were reused.
`figures/four_conditions_64.png` groups the four conditions by the same sample ID.
The ZIP keeps original artifacts at its root and new artifacts under
`extension_05_01/`, including metric/source hashes and local integrity checks.
Historical execution and cost estimates are recorded in the report. Temporary
Modal launch/download scripts are removed after completing the experiment.

The [128-image qualitative supplement](ablation/reverse_noise/ema200k/qualitative_128/README.md)
contains all 128 IDs in fixed order: individual grids for five conditions,
eight paired comparison pages, and one ZIP with original PNGs, float tensors,
input banks and verification metadata. Each paired row uses the same x_T and
reverse seed. These use the evaluation bank, so their IDs do not refer to the
independent 64-image preview above.

## Local CLI

```bash
python ablation_sampling.py inspect --checkpoint checkpoints/200k/ckpt.pt --output-dir ablation/reverse_noise/local --device cpu
python ablation_sampling.py sanity --output-dir ablation/reverse_noise/local --device cuda
python ablation_sampling.py preview --output-dir ablation/reverse_noise/local --batch-size 64
python ablation_sampling.py generate --output-dir ablation/reverse_noise/local --num-images 5000 --batch-size 64
python ablation_sampling.py generate --output-dir ablation/reverse_noise/local --num-images 5000 --batch-size 64 --resume
# metrics takes the evaluation subdirectory (no model sampling):
python ablation_sampling.py metrics --output-dir ablation/reverse_noise/local/evaluation --num-images 5000 --fid-cache stats/cifar10.train.npz
python -m unittest discover -s tests -v
```

`--seed-init`, `--seed-reverse`, `--noise-scale 1 0` control the protocol.
The generation/metric runner supports scales `0, 0.1, 0.5, 0.75, 0.9, 1, 1.1, 1.25`
in a new run. Use `--num-images 128 --batch-size 64` for a small qualitative run;
do not invoke `metrics` for this visual comparison.
The original paired preview mode retains the fixed `[1, 0]` layout. The
0.5/0.1 supplement uses an independent output protocol with copied original x_T
banks, preserving the original 1/0 protocols and result manifests.
`--config-json` is required only when a checkpoint lacks config, and must be
the training config; conflicting supplied config is rejected. Missing or
unexpected EMA keys fail strict loading.

## Artifacts and resume

Each preview/evaluation directory contains `protocol.json`, `inputs/manifest.json`,
CPU x_T chunks, `lambda_1/` and `lambda_0/` float32 [0,1] chunks and manifests.
Bank batches are generated on CPU with seed_init + batch_index. Reverse seeds
are seed_reverse + batch_index on the sampling device. Both branches load the
same bank chunk. No full reverse-noise trajectories are stored.

Chunks always have the fixed full batch size. The final padded batch preserves
existing IDs when extending N; metrics consume exactly the first N images.
No image is selected by visual quality. Incomplete temporary files are ignored
and rewritten; published chunks and manifests use atomic rename. Resume checks
protocol, checkpoint/source/environment hashes, input IDs/hash, output IDs/hash,
shape, float32 dtype, finite values and range. It refuses incompatible or corrupt
results. Do not modify manifests manually. A completed batch is never resampled.

Preview images live at `preview/preview/`: `baseline_64.png`, `mean_only_64.png`,
`paired_64.png` and the two `*_reverse_seeds.png` grids. Paired tiles are adjacent
baseline/mean-only with the same ID; seed-study rows share x_T and columns vary
reverse seed. Mean-only repetitions within a row are expected, not evidence of
distribution-wide mode collapse.

Timing excludes model load, CPU transfer/save and warm-up; CUDA is synchronized
around each timed batch. `preview/preview/timing.json` reports warm-up separately.
Metrics read saved float tensors, never PNG grids. `metrics.csv` contains FID,
IS mean, **IS standard deviation over ten splits**, and sampling seconds/image.
`metrics_metadata.json` records the pipeline and reference hash.

Report delta_FID = FID_mean_only - FID_baseline. Compare the two conditions at
the same N. A single bank gives no across-run uncertainty; IS split std is not
a confidence interval. FID/IS do not separately establish diversity and fidelity.
Do not compare FID-5k directly with paper FID-50k. Changes reflect this checkpoint
and protocol, not all DDPMs. See the workspace plan for the full methodology.
