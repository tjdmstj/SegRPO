"""Fold the LoRA adapter into the base weights and save a standalone Whisper checkpoint.

Two reasons, and the first is a hard blocker:

1. peft is installed in the `cosyvoice2` env but NOT in `cosyvoice`, which is where the
   GRPO recipe runs. A reward that calls PeftModel.from_pretrained() therefore cannot load
   inside training at all. After merging, the reward loads an ordinary
   WhisperForConditionalGeneration and peft is not needed at run time.

2. Speed. The reward decodes autoregressively for every rollout (rollout_n=4 per prompt),
   and merged weights skip the adapter's extra matmul at every layer.

Run this once, in the env that HAS peft:

    python merge_adapter.py \
        --adapter exp/whisper_lora/adapter_best \
        --out     exp/whisper_lora/merged
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cuda_home  # noqa: F401,E402  (must precede transformers)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='openai/whisper-large-v3')
    p.add_argument('--adapter', default='exp/whisper_lora/adapter_best')
    p.add_argument('--out', default='exp/whisper_lora/merged')
    p.add_argument('--dtype', default='float16',
                   help='float16 halves the 6 GB checkpoint and matches how the reward '
                        'loads it; use float32 to keep full precision on disk')
    args = p.parse_args()

    import peft_compat  # noqa: F401  (peft 0.20 raises on this box's torchao 0.9)
    from peft import PeftModel
    import cuda_compat  # noqa: F401  (must precede transformers)
    from transformers import (WhisperFeatureExtractor,
                              WhisperForConditionalGeneration, WhisperTokenizerFast)

    dt = dict(float16=torch.float16, bfloat16=torch.bfloat16,
              float32=torch.float32)[args.dtype]
    print(f'base    {args.model}')
    model = WhisperForConditionalGeneration.from_pretrained(args.model, torch_dtype=dt)
    print(f'adapter {args.adapter}')
    # train_lal.py adapters also carry the frame-level language head (modules_to_save);
    # the module has to exist on the base before peft can load its weights into it
    import json
    cfg = json.load(open(os.path.join(args.adapter, 'adapter_config.json')))
    has_lid = 'lid_head' in (cfg.get('modules_to_save') or [])
    if has_lid:
        model.lid_head = torch.nn.Linear(model.config.d_model, 3).to(dt)
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    lid_head = None
    if has_lid:
        lid_head = model.lid_head
        del model.lid_head

    # the released checkpoint carries a forced decoder prefix for its own default
    # language; leaving it in would override the per-utterance token the reward forces
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    if lid_head is not None:
        # outside the HF checkpoint: from_pretrained would drop it as an unexpected key
        torch.save(lid_head.state_dict(), os.path.join(args.out, 'lid_head.pt'))
        print(f'LID head -> {os.path.join(args.out, "lid_head.pt")}')
    WhisperTokenizerFast.from_pretrained(args.model).save_pretrained(args.out)
    WhisperFeatureExtractor.from_pretrained(args.model).save_pretrained(args.out)
    size = sum(os.path.getsize(os.path.join(args.out, f)) for f in os.listdir(args.out))
    print(f'\nmerged -> {args.out}  ({size / 2**30:.1f} GB)')
    print('point the reward at it with grpo_conf.mer_model, and leave mer_adapter null:')
    print(f'  mer_model: {os.path.abspath(args.out)}')
    print('  mer_adapter: null')


if __name__ == '__main__':
    main()
