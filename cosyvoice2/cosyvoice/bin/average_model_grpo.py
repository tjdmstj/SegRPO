#!/usr/bin/env python3
"""Average the best GRPO checkpoints, ranked by CV reward.

cosyvoice/bin/average_model.py cannot be used here for two reasons:

  1. It rebuilds the checkpoint path as 'epoch_{epoch}_whole.pt', ignoring the step. GRPO
     saves mid-epoch as 'epoch_{epoch}_step_{step}.pt' (save_per_step), and a run that is
     killed before the epoch ends has no _whole.pt at all -- so it resolves every entry to
     the same non-existent file.
  2. Every mid-epoch checkpoint of a 1-epoch run shares epoch=0, so ranking by epoch alone
     cannot separate them.

This script pairs each yaml with the .pt of the same basename, which is exactly how
save_model() writes them, so the naming scheme never has to be re-derived.

Ranking uses loss ascending, matching average_model.py. GRPO's cv() stores
loss = -mean_reward, so that is the same as reward descending.
"""
import argparse
import glob
import os

import torch
import yaml


def main():
    ap = argparse.ArgumentParser(description='Average top-N GRPO checkpoints by CV reward')
    ap.add_argument('--dst_model', required=True)
    ap.add_argument('--src_path', required=True)
    ap.add_argument('--num', type=int, default=3)
    ap.add_argument('--min_step', type=int, default=0,
                    help='ignore checkpoints below this step (early ones are noisy)')
    args = ap.parse_args()

    cands = []
    for y in glob.glob(os.path.join(args.src_path, '*.yaml')):
        base = os.path.basename(y)
        if base.startswith(('train', 'init')):
            continue
        pt = y[:-len('.yaml')] + '.pt'
        if not os.path.exists(pt):
            continue
        with open(y) as f:
            d = yaml.load(f, Loader=yaml.BaseLoader)
        try:
            loss = float(d['loss_dict']['loss'])
        except (KeyError, TypeError, ValueError):
            # cv() used to dump a pickled tensor here, which BaseLoader returns as a list
            print('skip {}: loss is not a plain number'.format(base))
            continue
        step = int(d.get('step', 0))
        if step < args.min_step:
            continue
        cands.append((loss, int(d.get('epoch', 0)), step, pt))

    if not cands:
        raise SystemExit('no usable checkpoints under {}'.format(args.src_path))
    cands.sort(key=lambda x: x[0])
    picked = cands[:args.num]
    print('candidates: {}  averaging: {}'.format(len(cands), len(picked)))
    for loss, ep, st, pt in picked:
        print('  epoch {:>3} step {:>5} reward {:.4f}  {}'
              .format(ep, st, -loss, os.path.basename(pt)))

    avg, n = {}, len(picked)
    for _loss, _ep, _st, pt in picked:
        states = torch.load(pt, map_location='cpu')
        for k, v in states.items():
            if k in ('step', 'epoch'):
                continue
            if not torch.is_tensor(v) or not v.is_floating_point():
                # ints (e.g. buffers) cannot be averaged; keep the first one as-is
                avg.setdefault(k, v)
                continue
            avg[k] = v.clone().float() if k not in avg else avg[k] + v.float()
    for k, v in avg.items():
        if torch.is_tensor(v) and v.is_floating_point():
            avg[k] = torch.true_divide(v, n)

    os.makedirs(os.path.dirname(os.path.abspath(args.dst_model)), exist_ok=True)
    torch.save(avg, args.dst_model)
    print('saved {}'.format(args.dst_model))


if __name__ == '__main__':
    main()
