"""
volume_nanogpt_example.py
-------------------------
Worked example: estimate loss-basin volume for a trained nanoGPT
checkpoint (https://github.com/karpathy/nanoGPT).

Run this from the minima-volumes repo root after:
  1. preparing a dataset, e.g. `python nanogpt/data/shakespeare_char/prepare.py`
  2. training a checkpoint with
     `cd nanogpt && python train.py config/train_shakespeare_char.py`

The bridge does three things nanoGPT-specific:
  - reconstructs GPT from the checkpoint and strips the `_orig_mod.` prefix
    that torch.compile leaves on state-dict keys
  - wraps GPT in a thin module so forward(idx) returns logits at every
    position; nanoGPT's own forward returns last-position-only when
    targets is None (an inference-time optimisation that breaks
    cross-entropy over the full sequence)
  - builds a fixed (x, y) batch of non-overlapping windows from val.bin
    so every loss curve is comparable

Adjust the path constants for your layout. Everything else is shared
with volume_example.py.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# nanoGPT is vendored at ./nanogpt/; add it to the import path
NANOGPT_DIR = Path(__file__).resolve().parent / 'nanogpt'
sys.path.insert(0, str(NANOGPT_DIR))

from model import GPTConfig, GPT  # from nanoGPT
from volume import (
    VolumeConfig,
    compute_loss_curves,
    save_curves,
    loss_curves_to_radii,
    log_volume_from_radii,
)


CKPT_PATH = NANOGPT_DIR / 'out-shakespeare-char' / 'ckpt.pt'
DATA_DIR  = NANOGPT_DIR / 'data' / 'shakespeare_char'
CURVES    = Path('curves') / 'shakespeare_char.npz'
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'


class FullLogitsGPT(nn.Module):
    """Run nanoGPT GPT and return logits at every position."""
    def __init__(self, gpt: GPT):
        super().__init__()
        self.gpt = gpt

    def forward(self, idx):
        _, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        tok = self.gpt.transformer.wte(idx)
        pemb = self.gpt.transformer.wpe(pos)
        x = self.gpt.transformer.drop(tok + pemb)
        for block in self.gpt.transformer.h:
            x = block(x)
        x = self.gpt.transformer.ln_f(x)
        return self.gpt.lm_head(x)


def load_model(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = GPTConfig(**ckpt['model_args'])
    model = GPT(cfg)
    sd = ckpt['model']
    prefix = '_orig_mod.'
    for k in list(sd.keys()):
        if k.startswith(prefix):
            sd[k[len(prefix):]] = sd.pop(k)
    model.load_state_dict(sd)
    model.eval().to(DEVICE)
    return model, ckpt['model_args'], ckpt.get('iter_num'), ckpt.get('best_val_loss')


def build_eval_batch(data_dir: Path, block_size: int, n_windows: int):
    val = np.memmap(data_dir / 'val.bin', dtype=np.uint16, mode='r')
    starts = np.arange(n_windows) * block_size
    assert starts[-1] + block_size + 1 <= len(val), 'val.bin too small'
    x = torch.stack([torch.from_numpy(val[s:s + block_size].astype(np.int64)) for s in starts])
    y = torch.stack([torch.from_numpy(val[s + 1:s + 1 + block_size].astype(np.int64)) for s in starts])
    return x, y


def main():
    print('Loading checkpoint...')
    gpt, args, iter_num, best_val = load_model(CKPT_PATH)
    block_size = args['block_size']
    n_params = sum(p.numel() for p in gpt.parameters())
    print(f'  iter={iter_num}  best_val_loss={best_val:.4f}  block_size={block_size}  params={n_params:,}')

    print('Building eval batch from val.bin...')
    x, y = build_eval_batch(DATA_DIR, block_size, n_windows=32)
    x, y = x.to(DEVICE), y.to(DEVICE)
    print(f'  x={tuple(x.shape)}  y={tuple(y.shape)}')

    model = FullLogitsGPT(gpt).to(DEVICE).eval()

    def loss_fn(logits, targets):
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

    with torch.no_grad():
        baseline = loss_fn(model(x), y).item()
    print(f'  baseline loss on this batch: {baseline:.4f}')

    config = VolumeConfig(
        num_directions=20,
        perturbation_seed=1,
        coefficients=np.linspace(0, 1, 100) ** 2,
    )
    print(f'Stage 1: {config.num_directions} directions × {len(config.coefficients)} coeffs...')
    curves = compute_loss_curves(model, x, y, config, loss_fn)
    save_curves(curves, config, CURVES)
    print(f'  saved → {CURVES}')

    print('Stage 2+3: thresholds → radii → log-volume')
    losses = np.stack([c.loss for c in curves])
    print(f'  loss range across (dirs, coeffs): min={losses.min():.3f} max={losses.max():.3f}')
    print(f'  loss at coeff=0 (sanity): mean={losses[:, 0].mean():.4f}')
    print(f'  loss at max coeff:        mean={losses[:, -1].mean():.4f}')

    for thresh in [baseline + 0.25, baseline + 0.5, baseline + 1.0, 2 * baseline]:
        radii = loss_curves_to_radii(curves, thresh, config)
        log_vol = log_volume_from_radii(radii, n_params)
        crossed = len(radii)
        if crossed > 0:
            r_arr = np.array(radii)
            r_summary = f'r in [{r_arr.min():.3f}, {r_arr.max():.3f}], median={np.median(r_arr):.3f}'
        else:
            r_summary = 'no directions crossed'
        print(f'  thresh={thresh:.3f}: {crossed}/{config.num_directions} crossed, '
              f'log_vol={log_vol:.3f}  ({r_summary})')


if __name__ == '__main__':
    main()
