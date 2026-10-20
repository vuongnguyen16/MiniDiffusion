import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from diffusion import GaussianDiffusionSampler


class Epsilon(nn.Module):
    def forward(self, x, t):
        return 0.1 * x


def original(sampler, x):
    for step in reversed(range(sampler.T)):
        t = x.new_full((len(x),), step, dtype=torch.long)
        mean, log_var = sampler.p_mean_variance(x, t)
        noise = torch.randn_like(x) if step > 0 else 0
        x = mean + torch.exp(0.5 * log_var) * noise
    return x.clamp(-1, 1)


class SamplerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.sampler = GaussianDiffusionSampler(Epsilon(), 1e-4, .02, 1000, img_size=4)
        self.x = torch.randn(2, 3, 4, 4)

    def test_legacy_baseline_exact(self):
        torch.manual_seed(19)
        expected = original(self.sampler, self.x)
        torch.manual_seed(19)
        actual = self.sampler(self.x, noise_scale=1.)
        self.assertTrue(torch.equal(expected, actual))

    def test_mean_only_ignores_reverse_seed_and_preserves_rng(self):
        gen = torch.Generator().manual_seed(11)
        state = gen.get_state().clone()
        a = self.sampler(self.x, 0., gen)
        self.assertTrue(torch.equal(state, gen.get_state()))
        b = self.sampler(self.x, 0., torch.Generator().manual_seed(12))
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.isfinite(a).all())
        self.assertEqual(a.shape, self.x.shape)
        self.assertLessEqual(a.abs().max().item(), 1.)

    def test_stochastic_replay_and_seed_sensitivity(self):
        a = self.sampler(self.x, 1., torch.Generator().manual_seed(11))
        b = self.sampler(self.x, 1., torch.Generator().manual_seed(11))
        c = self.sampler(self.x, 1., torch.Generator().manual_seed(12))
        self.assertTrue(torch.equal(a, b))
        self.assertGreater((a-c).abs().max().item(), 1e-5)

    def test_invalid_scales_rejected(self):
        for scale in [-1., float('nan'), float('inf')]:
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                self.sampler(self.x, noise_scale=scale)


class ArtifactTests(unittest.TestCase):
    def test_additional_scales_save_and_replay_with_paired_inputs(self):
        from ablation_sampling import prepare_run, generate, read_images, bank_batch
        sampler = GaussianDiffusionSampler(Epsilon(), 1e-4, .02, 4, img_size=4)
        protocol = {'config': {'img_size': 4}, 'checkpoint_sha256': 'fixture',
                    'batch_size': 2, 'seed_init': 2026, 'seed_reverse': 10001,
                    'noise_scales': [.5, .1], 'purpose': 'extension-test'}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepare_run(root, protocol, 3)
            before = bank_batch(root, 0).clone()
            for scale in [.5, .1]:
                generate(sampler, root, scale, 3, torch.device('cpu'))
                actual = torch.stack([torch.from_numpy(x) for x in read_images(root, scale, 3)])
                expected = (sampler(before, scale, torch.Generator().manual_seed(10001)) + 1) / 2
                self.assertTrue(torch.equal(actual[:2], expected))
                generate(sampler, root, scale, 3, torch.device('cpu'), resume=True)
            self.assertTrue(torch.equal(before, bank_batch(root, 0)))

    def test_preview_rejects_protocol_that_omits_a_required_condition(self):
        from ablation_sampling import validate_preview_scales
        validate_preview_scales([1.,0.])
        with self.assertRaises(ValueError):
            validate_preview_scales([.5])

    def test_sanity_identity_rejects_replaced_checkpoint_or_sampler(self):
        from ablation_sampling import experiment_identity, require_sanity
        meta = {'checkpoint_sha256':'one','checkpoint_step':200000,'config':{'T':1000},
                'environment':{'gpu':'L40S'},'source_sha256':{'model.py':'m','diffusion.py':'s','ablation_sampling.py':'r'}}
        result = {'passed':True,'experiment_identity':experiment_identity(meta)}
        require_sanity(result,meta)
        with self.assertRaises(ValueError):
            require_sanity(result,dict(meta,checkpoint_sha256='two'))
        with self.assertRaises(ValueError):
            require_sanity(result,dict(meta,source_sha256=dict(meta['source_sha256'],**{'diffusion.py':'new'})))

    def test_resume_detects_corruption_and_protocol_mismatch(self):
        from ablation_sampling import prepare_run, generate, read_images
        sampler = GaussianDiffusionSampler(Epsilon(), 1e-4, .02, 4, img_size=4)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol = {'config': {'img_size': 4}, 'checkpoint_sha256': 'fixture',
                        'batch_size': 2, 'seed_init': 2026, 'seed_reverse': 10001,
                        'noise_scales': [1., 0.], 'purpose': 'test'}
            prepare_run(root, protocol, 5, resume=False)
            generate(sampler, root, 1., 5, torch.device('cpu'), resume=False)
            images = torch.stack([torch.from_numpy(x) for x in read_images(root, 1., 5)])
            self.assertEqual(tuple(images.shape), (5, 3, 4, 4))
            before = (root/'lambda_1/manifest.json').read_bytes()
            generate(sampler, root, 1., 5, torch.device('cpu'), resume=True)
            self.assertEqual(before, (root/'lambda_1/manifest.json').read_bytes())
            with self.assertRaises(ValueError):
                prepare_run(root, dict(protocol, batch_size=3), 5, resume=True)
            chunk = root/'lambda_1/batch_000000.pt'
            chunk.write_bytes(b'corrupt')
            with self.assertRaises(ValueError):
                generate(sampler, root, 1., 5, torch.device('cpu'), resume=True)

    def test_extension_preserves_existing_sample_ids(self):
        from ablation_sampling import prepare_run, bank_batch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol = {'config': {'img_size': 4}, 'batch_size': 2, 'seed_init': 3,
                        'seed_reverse': 8, 'noise_scales': [1., 0.]}
            prepare_run(root, protocol, 3, resume=False)
            before = bank_batch(root, 1).clone()
            prepare_run(root, protocol, 5, resume=True)
            after = bank_batch(root, 1)
            self.assertTrue(torch.equal(before, after))


if __name__ == '__main__':
    unittest.main()
