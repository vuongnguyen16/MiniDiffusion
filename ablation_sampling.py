"""Paired DDPM reverse-noise ablation. See ABLATION.md for the fixed protocol."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch

from diffusion import GaussianDiffusionSampler
from model import UNet

BASE_COMMIT = 'f38fbe8ae1a465578b438ecdc3d058aa383981b7'
ARCH_KEYS = ('T', 'ch', 'ch_mult', 'attn', 'num_res_blocks', 'dropout')
CONFIG_KEYS = ARCH_KEYS + ('beta_1', 'beta_T', 'mean_type', 'var_type', 'img_size')


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def atomic_tensor(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temp)
    os.replace(temp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def setup_device(device):
    device = torch.device(device)
    torch.set_num_threads(4)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return device


def environment(device):
    return {'torch': str(torch.__version__), 'cuda': torch.version.cuda,
            'device': str(device), 'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
            'dtype': 'float32', 'deterministic': torch.are_deterministic_algorithms_enabled(),
            'tf32': False}


def load_checkpoint(checkpoint, device, config_path=None):
    started = time.monotonic()
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=True)
    config = ckpt.get('config')
    if config is None:
        if config_path is None:
            raise ValueError('Checkpoint has no config; supply --config-json from the training run')
        config = read_json(config_path)
    elif config_path is not None:
        supplied = read_json(config_path)
        if any(supplied.get(key) != config.get(key) for key in CONFIG_KEYS):
            raise ValueError('Supplied config conflicts with checkpoint config')
    missing = [key for key in CONFIG_KEYS if key not in config]
    if missing:
        raise ValueError(f'Missing checkpoint config: {missing}')
    for key, expected in {'T': 1000, 'beta_1': 1e-4, 'beta_T': .02,
                          'mean_type': 'epsilon', 'var_type': 'fixedlarge', 'img_size': 32}.items():
        if config[key] != expected:
            raise ValueError(f'Fixed protocol requires {key}={expected}, got {config[key]}')
    if ckpt.get('step') != 200000:
        raise ValueError('This protocol requires exactly 200000 optimizer updates')
    model = UNet(**{key: config[key] for key in ARCH_KEYS})
    keys = model.load_state_dict(ckpt['ema_model'], strict=True)
    if any(t.dtype != torch.float32 for t in model.parameters()):
        raise ValueError('EMA must be float32')
    model = model.to(device).eval()
    sampler = GaussianDiffusionSampler(model, config['beta_1'], config['beta_T'], config['T'],
                                       config['img_size'], config['mean_type'], config['var_type']).to(device).eval()
    sync(device)
    meta = {'base_commit': BASE_COMMIT, 'checkpoint_sha256': sha256(checkpoint),
            'checkpoint_step': ckpt['step'], 'config': {k: config[k] for k in CONFIG_KEYS},
            'missing_keys': list(keys.missing_keys), 'unexpected_keys': list(keys.unexpected_keys),
            'environment': environment(device), 'load_seconds': time.monotonic()-started,
            'source_sha256': {name: sha256(Path(__file__).parent/name)
                              for name in ['diffusion.py', 'model.py', 'ablation_sampling.py']}}
    return sampler, meta


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def protocol_for(meta, batch_size, seed_init, seed_reverse, purpose):
    return {k: v for k, v in dict(meta, batch_size=batch_size, seed_init=seed_init,
                                seed_reverse=seed_reverse, noise_scales=[1., 0.], purpose=purpose,
                                input_seed_rule='seed_init + batch_index; full batches on CPU',
                                reverse_seed_rule='seed_reverse + batch_index',
                                image_storage='float32 NCHW [0,1]', metric_splits=10,
                                metric_use_torch=False).items() if k != 'load_seconds'}


def experiment_identity(meta):
    # Runner changes can add logging/reporting; sampler/model hashes bind the
    # tested numerical behavior, alongside weights, config and environment.
    return json_hash({key:meta[key] for key in
                      ['checkpoint_sha256','checkpoint_step','config','environment']} | {
                          'sampler':meta['source_sha256']['diffusion.py'],
                          'model':meta['source_sha256']['model.py']})


def require_sanity(result,meta):
    if not result.get('passed') or result.get('experiment_identity')!=experiment_identity(meta):
        raise ValueError('Sanity identity mismatch; rerun sanity for this checkpoint/code/environment')


def validate_tensor(tensor, shape, image=False):
    if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != tuple(shape):
        raise ValueError(f'Invalid tensor shape; expected {shape}')
    if tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
        raise ValueError('Tensor must be finite float32')
    if image and (tensor.min() < 0 or tensor.max() > 1):
        raise ValueError('Image outside [0,1]')


def checked_tensor(path, entry, shape, image=False):
    if not Path(path).is_file() or sha256(path) != entry['sha256']:
        raise ValueError(f'Corrupt or missing artifact: {path}')
    try:
        tensor = torch.load(path, map_location='cpu', weights_only=True)
    except Exception as exc:
        raise ValueError(f'Unreadable artifact: {path}') from exc
    validate_tensor(tensor, shape, image)
    return tensor


def prepare_run(root, protocol, count, resume=False):
    root = Path(root)
    if count <= 0 or protocol['batch_size'] <= 0:
        raise ValueError('count and batch_size must be positive')
    path = root/'protocol.json'
    if path.exists():
        if not resume:
            raise ValueError('Run already exists; use --resume')
        if read_json(path) != protocol:
            raise ValueError('Protocol/checkpoint/environment mismatch; use a new output directory')
    elif root.exists() and any(root.iterdir()):
        raise ValueError('Nonempty output directory without protocol')
    else:
        atomic_json(path, protocol)
    inputs = root/'inputs'
    manifest_path = inputs/'manifest.json'
    manifest = read_json(manifest_path) if manifest_path.exists() else {'batches': {}, 'requested_count': count}
    manifest['requested_count'] = max(count, manifest['requested_count'])
    size, side = protocol['batch_size'], protocol['config']['img_size']
    shape = (size, 3, side, side)
    for index in range(math.ceil(count/size)):
        key = str(index)
        file = inputs/f'batch_{index:06d}.pt'
        if key in manifest['batches']:
            checked_tensor(file, manifest['batches'][key], shape)
            continue
        x = torch.randn(shape, generator=torch.Generator().manual_seed(protocol['seed_init']+index))
        atomic_tensor(file, x)
        manifest['batches'][key] = {'file': file.name, 'sha256': sha256(file),
                                   'sample_ids': list(range(index*size, (index+1)*size))}
        atomic_json(manifest_path, manifest)
    atomic_json(manifest_path, manifest)


def bank_batch(root, index):
    root = Path(root)
    protocol = read_json(root/'protocol.json')
    entry = read_json(root/'inputs/manifest.json')['batches'][str(index)]
    size, side = protocol['batch_size'], protocol['config']['img_size']
    expected_ids = list(range(index*size, (index+1)*size))
    if entry['sample_ids'] != expected_ids:
        raise ValueError('Input sample ID mismatch')
    return checked_tensor(root/'inputs'/entry['file'], entry, (size, 3, side, side))


def condition_path(root, scale):
    if float(scale) not in (0., .1, .5, 1.):
        raise ValueError('Supported noise scales: 0, 0.1, 0.5, 1')
    return Path(root)/f'lambda_{float(scale):g}'


def validate_preview_scales(scales):
    if list(scales) != [1.,0.]:
        raise ValueError('Paired preview requires --noise-scale 1 0 in that order')


@torch.inference_mode()
def generate(sampler, root, scale, count, device, resume=False, commit=None, deadline=None):
    root = Path(root)
    protocol = read_json(root/'protocol.json')
    folder = condition_path(root, scale)
    path = folder/'manifest.json'
    if path.exists() and not resume:
        raise ValueError('Condition exists; use --resume')
    manifest = read_json(path) if path.exists() else {
        'protocol_sha256': json_hash(protocol), 'noise_scale': scale, 'batches': {}, 'requested_count': count}
    if manifest['protocol_sha256'] != json_hash(protocol) or manifest['noise_scale'] != scale:
        raise ValueError('Condition protocol mismatch')
    size, side = protocol['batch_size'], protocol['config']['img_size']
    manifest['requested_count'] = max(count, manifest['requested_count'])
    for index in range(math.ceil(count/size)):
        key = str(index)
        ids = list(range(index*size, (index+1)*size))
        file = folder/f'batch_{index:06d}.pt'
        x = bank_batch(root, index)
        input_hash = read_json(root/'inputs/manifest.json')['batches'][key]['sha256']
        if key in manifest['batches']:
            entry = manifest['batches'][key]
            if entry['sample_ids'] != ids or entry['input_sha256'] != input_hash:
                raise ValueError('Output sample IDs/input bank mismatch')
            checked_tensor(file, entry, (size, 3, side, side), image=True)
            continue
        prior_times = [e['seconds'] for e in manifest['batches'].values()]
        reserve = max(prior_times, default=120.) * 1.25
        if deadline is not None and time.monotonic()+reserve >= deadline:
            raise TimeoutError('Budget guard stopped before starting another batch; resume is safe')
        gen = torch.Generator(device=device).manual_seed(protocol['seed_reverse']+index)
        x = x.to(device)
        sync(device)
        started = time.monotonic()
        images = ((sampler(x, noise_scale=scale, generator=gen)+1)/2).float()
        sync(device)
        elapsed = time.monotonic()-started
        images = images.cpu()
        validate_tensor(images, (size, 3, side, side), image=True)
        atomic_tensor(file, images)
        manifest['batches'][key] = {'file': file.name, 'sha256': sha256(file), 'sample_ids': ids,
                                   'input_sha256': input_hash, 'reverse_seed': protocol['seed_reverse']+index,
                                   'seconds': elapsed, 'seconds_per_image': elapsed/size}
        atomic_json(path, manifest)
        if commit:
            commit()
        print(f'lambda={scale:g} batch={index} images={min((index+1)*size,count)}/{count} seconds={elapsed:.3f}', flush=True)
    atomic_json(path, manifest)
    return manifest


def read_images(root, scale, count):
    root = Path(root)
    protocol = read_json(root/'protocol.json')
    folder = condition_path(root, scale)
    manifest = read_json(folder/'manifest.json')
    if manifest['protocol_sha256'] != json_hash(protocol) or manifest['noise_scale'] != scale:
        raise ValueError('Image manifest protocol mismatch')
    size, side = protocol['batch_size'], protocol['config']['img_size']
    remaining = count
    for index in range(math.ceil(count/size)):
        entry = manifest['batches'].get(str(index))
        if entry is None or entry['sample_ids'] != list(range(index*size,(index+1)*size)):
            raise ValueError('Missing batch or sample ID mismatch')
        bank_batch(root, index)
        input_hash = read_json(root/'inputs/manifest.json')['batches'][str(index)]['sha256']
        if entry['input_sha256'] != input_hash:
            raise ValueError('Image manifest input hash mismatch')
        images = checked_tensor(folder/entry['file'], entry, (size,3,side,side), image=True)
        for img in images[:remaining]:
            yield img.numpy()
        remaining -= min(remaining, len(images))


@torch.inference_mode()
def sanity(sampler, root, device, seed_init=2026, seed_reverse=10001, meta=None):
    root = Path(root)
    x = torch.randn((2,3,sampler.img_size,sampler.img_size),
                    generator=torch.Generator().manual_seed(seed_init)).to(device)
    atomic_tensor(root/'x_T.pt', x.cpu())
    def run(scale, seed):
        return sampler(x, scale, torch.Generator(device=device).manual_seed(seed))
    def legacy():
        value = x.clone()
        for step in reversed(range(sampler.T)):
            t = value.new_full((len(x),), step, dtype=torch.long)
            mean, log_var = sampler.p_mean_variance(value,t)
            noise = torch.randn_like(value) if step > 0 else 0
            value = mean + torch.exp(.5*log_var)*noise
        return value.clamp(-1,1)
    # Restoring the same global RNG exercises the untouched original noise path.
    torch.manual_seed(seed_reverse)
    old = legacy()
    torch.manual_seed(seed_reverse)
    new = sampler(x,1.)
    a, b = run(0.,seed_reverse), run(0.,seed_reverse+1)
    c, d, e = run(1.,seed_reverse), run(1.,seed_reverse+1), run(1.,seed_reverse)
    def diff(left,right):
        return float((left-right).abs().max())
    result = {'T': sampler.T, 'images':len(x), 'atol':1e-6,'rtol':1e-5,
              'legacy_max_abs_diff': diff(old,new), 'mean_only_seed_max_abs_diff':diff(a,b),
              'stochastic_seed_max_abs_diff':diff(c,d), 'replay_max_abs_diff':diff(c,e),
              'baseline_compatible':torch.allclose(old,new,atol=1e-6,rtol=1e-5),
              'mean_only_invariant':torch.allclose(a,b,atol=1e-6,rtol=1e-5),
              'stochastic_changes':not torch.allclose(c,d,atol=1e-6,rtol=1e-5),
              'replay_matches':torch.allclose(c,e,atol=1e-6,rtol=1e-5),
              'finite_shape_range':all(v.shape==x.shape and bool(torch.isfinite(v).all()) and
                                       float(v.abs().max())<=1 for v in [old,new,a,b,c,d,e]),
              'environment':environment(device),'seed_init':seed_init,'seed_reverse':seed_reverse}
    result['passed'] = all(result[k] for k in ['baseline_compatible','mean_only_invariant',
                                              'stochastic_changes','replay_matches','finite_shape_range'])
    if meta is not None:
        result['experiment_identity'] = experiment_identity(meta)
    atomic_json(root/'results.json', result)
    if not result['passed']:
        raise RuntimeError(f'Sanity failed: {result}')
    return result


def save_grid(images, path, columns=8, labels=None):
    from PIL import Image, ImageDraw
    images = images.detach().cpu()
    side = images.shape[-1]*3
    rows = math.ceil(len(images)/columns)
    canvas = Image.new('RGB',(columns*(side+4),rows*(side+22)), 'white')
    draw = ImageDraw.Draw(canvas)
    for index,img in enumerate(images):
        pixels = (img.clamp(0,1)*255).round().byte().permute(1,2,0).numpy()
        tile = Image.fromarray(pixels).resize((side,side),Image.Resampling.NEAREST)
        xy = ((index%columns)*(side+4),(index//columns)*(side+22))
        canvas.paste(tile,xy)
        draw.text((xy[0],xy[1]+side+2),labels[index] if labels else str(index),fill='black')
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    canvas.save(path)


@torch.inference_mode()
def preview(sampler, root, device, commit=None, deadline=None):
    root = Path(root)
    validate_preview_scales(read_json(root/'protocol.json')['noise_scales'])
    timings = {}
    # Separate full-T warm-up from timed, persisted batches; use a different input.
    warm_x = torch.randn((2,3,32,32),generator=torch.Generator().manual_seed(997)).to(device)
    sync(device)
    started = time.monotonic()
    sampler(warm_x,1.,torch.Generator(device=device).manual_seed(998))
    sync(device)
    timings['warmup_seconds'] = time.monotonic()-started
    tensors = {}
    for scale,name in [(1.,'baseline'),(0.,'mean_only')]:
        manifest = generate(sampler,root,scale,64,device,resume=True,commit=commit,deadline=deadline)
        tensors[scale] = torch.stack([torch.from_numpy(img) for img in read_images(root,scale,64)])
        save_grid(tensors[scale],root/'preview'/f'{name}_64.png')
        times = [e['seconds'] for e in manifest['batches'].values()]
        timings[f'lambda_{scale:g}'] = {'batch_seconds':times,
                                      'seconds_per_image':sum(times)/(len(times)*read_json(root/'protocol.json')['batch_size'])}
    paired = torch.stack([tensors[1.],tensors[0.]],dim=1).flatten(0,1)
    labels = [f'ID {i} lambda={scale:g}' for i in range(64) for scale in [1.,0.]]
    save_grid(paired,root/'preview/paired_64.png',8,labels)
    x = bank_batch(root,0)[:4].to(device)
    if len(x)!=4:
        raise ValueError('Preview requires batch_size >= 4')
    for scale,name in [(1.,'baseline'),(0.,'mean_only')]:
        values = []
        for seed in [11001,11002,11003,11004]:
            if deadline is not None and time.monotonic()+120 >= deadline:
                raise TimeoutError('Budget guard before seed study')
            values.append(((sampler(x,scale,torch.Generator(device=device).manual_seed(seed))+1)/2).cpu())
        tensor = torch.stack(values,dim=1).flatten(0,1)
        atomic_tensor(root/'preview'/f'{name}_reverse_seeds.pt',tensor)
        save_grid(tensor,root/'preview'/f'{name}_reverse_seeds.png',4,
                  [f'ID {i} seed={s}' for i in range(4) for s in [11001,11002,11003,11004]])
    timings['load_and_io_excluded'] = True
    atomic_json(root/'preview/timing.json',timings)
    atomic_json(root/'preview/config.json',read_json(root/'protocol.json'))
    if commit:
        commit()
    return timings


def metrics(root, fid_cache, count, device, batch_size=50):
    from score.both import get_inception_and_fid_score
    from score.inception import FID_WEIGHTS_URL
    root, fid_cache = Path(root), Path(fid_cache)
    if count < 10 or not fid_cache.is_file():
        raise ValueError('Metrics require N >= 10 and an existing fid_cache')
    weights = Path(torch.hub.get_dir())/'checkpoints'/FID_WEIGHTS_URL.rsplit('/',1)[-1]
    with __import__('numpy').load(fid_cache) as stats:
        if stats['mu'].shape != (2048,) or stats['sigma'].shape != (2048,2048):
            raise ValueError('Reference statistics must have 2048-dimensional mu/sigma')
    meta = {'num_images':count,'splits':10,'use_torch':False,'metric_batch_size':batch_size,
            'fid_cache_sha256':sha256(fid_cache), 'inception_url':FID_WEIGHTS_URL,
            'protocol_sha256':json_hash(read_json(root/'protocol.json')),
            'score_source_sha256':{name:sha256(Path(__file__).parent/'score'/name)
                                    for name in ['both.py','inception.py','fid.py']},
            'environment':environment(device), 'preprocessing':'float32 NCHW [0,1]; repo Inception resize/normalize'}
    path = root/'metrics_metadata.json'
    if path.exists():
        prior = read_json(path)
        for key in meta:
            if key != 'num_images' and prior[key] != meta[key]:
                raise ValueError('Metric protocol changed; use a separate run')
    rows = []
    for scale in read_json(root/'protocol.json')['noise_scales']:
        # Fully validate every chunk before allocating Inception arrays.
        for _ in read_images(root,scale,count):
            pass
        started = time.monotonic()
        (is_mean,is_std),fid = get_inception_and_fid_score(
            read_images(root,scale,count),str(fid_cache),num_images=count,splits=10,
            batch_size=batch_size,use_torch=False,device=device,verbose=True)
        if not all(math.isfinite(float(x)) for x in [is_mean,is_std,fid]):
            raise ValueError('Nonfinite metric')
        sampling = read_json(condition_path(root,scale)/'manifest.json')['batches']
        seconds = sum(sampling[str(i)]['seconds'] for i in range(math.ceil(count/read_json(root/'protocol.json')['batch_size'])))
        generated = math.ceil(count/read_json(root/'protocol.json')['batch_size'])*read_json(root/'protocol.json')['batch_size']
        row = {'condition':'DDPM baseline' if scale==1 else 'DDPM mean-only' if scale==0 else 'DDPM intermediate',
               'lambda':scale,'N':count,'FID':float(fid),'IS_mean':float(is_mean),'IS_split_std':float(is_std),
               'seconds_per_image':seconds/generated,'metric_seconds':time.monotonic()-started}
        rows.append(row)
        atomic_json(root/f'metrics_lambda_{scale:g}_N{count}.json',row)
    meta['inception_sha256'] = sha256(weights)
    atomic_json(path,meta)
    temp = root/'metrics.csv.tmp'
    with temp.open('w',newline='',encoding='utf-8') as file:
        writer = csv.DictWriter(file,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp,root/'metrics.csv')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['inspect','sanity','preview','generate','metrics'])
    parser.add_argument('--checkpoint',default='checkpoints/200k/ckpt.pt')
    parser.add_argument('--config-json')
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--num-images',type=int,default=5000)
    parser.add_argument('--noise-scale',type=float,nargs='+',default=[1.,0.])
    parser.add_argument('--seed-init',type=int,default=2026)
    parser.add_argument('--seed-reverse',type=int,default=10001)
    parser.add_argument('--fid-cache',type=Path)
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    if args.mode == 'preview':
        validate_preview_scales(args.noise_scale)
    device = setup_device(args.device)
    if args.mode=='metrics':
        if args.fid_cache is None:
            parser.error('--fid-cache is required for metrics')
        print(json.dumps(metrics(args.output_dir,args.fid_cache,args.num_images,device),indent=2))
        return
    sampler,meta = load_checkpoint(args.checkpoint,device,args.config_json)
    print(json.dumps(meta,indent=2))
    if args.mode=='inspect':
        atomic_json(args.output_dir/'checkpoint_inspection.json',meta)
    elif args.mode=='sanity':
        atomic_json(args.output_dir/'checkpoint_inspection.json',meta)
        sanity(sampler,args.output_dir/'sanity',device,args.seed_init,args.seed_reverse,meta)
    else:
        result_path = args.output_dir/'sanity/results.json'
        if not result_path.exists() or not read_json(result_path)['passed']:
            raise ValueError('Run sanity first in this output directory')
        require_sanity(read_json(result_path),meta)
        # Sanity artifacts live next to independent preview/evaluation banks.
        purpose = 'preview' if args.mode=='preview' else 'evaluation'
        seed = args.seed_init if purpose=='preview' else args.seed_init+1_000_000
        protocol = protocol_for(meta,args.batch_size,seed,args.seed_reverse,purpose)
        protocol['noise_scales'] = args.noise_scale
        count = 64 if args.mode=='preview' else args.num_images
        root = args.output_dir/purpose
        prepare_run(root,protocol,count,args.resume)
        if args.mode=='preview':
            preview(sampler,root,device)
        else:
            for scale in args.noise_scale:
                generate(sampler,root,scale,count,device,args.resume)


if __name__=='__main__':
    main()
