"""Fine-tune whisper-large-v3 on SwitchLingua code-switched speech.

    --method lora   LoRA adapter, ~15.7M trainable of 1.55B          (default)
    --method full   every parameter trained

Follows huggingface.co/blog/fine-tune-whisper, with three departures forced by this
corpus, and they apply to BOTH methods:

1. The language token is PER UTTERANCE. The blog fine-tunes one language, so it calls
   tokenizer.set_prefix_tokens() once. Here each utterance carries its own <|yue|> /
   <|zh|> / <|ja|> / <|ko|>, because the point is to teach the model that a
   matrix-language token does not mean "every word is in that language". large-v3 is
   required for that: it is the first checkpoint with <|yue|>, and Cantonese is 69% of
   this corpus.

2. Features are computed in the collator, not cached by dataset.map(): 128 mel bins x
   3000 frames x 10,854 utterances is ~17 GB of cache to avoid a few ms of work.

3. Eval during training is LOSS only. Seq2SeqTrainer applies one generation_config per
   batch, so with four languages in a batch a generate-based WER would decode most of dev
   under the wrong token. The real metric is a separate pass: infer.py.

full vs lora -- what actually differs:

                        lora                    full
    lr                  1e-3                    1e-5   (the blog's value)
    trainable           1.01%                   100%
    optimiser state     ~0.2 GB                 ~12 GB fp32 AdamW
    grad checkpointing  off (nothing to save)   ON (needed to fit)
    forgetting risk     low, base is frozen     real: 20.5h of 4 languages can damage
                                                the other 95 the checkpoint knows
    artefact            ~60 MB adapter          ~6 GB checkpoint

Full fine-tuning fits on one 48 GB card: fp32 params 6.2 + grads 6.2 + AdamW 12.4 = ~25 GB
before activations, and gradient checkpointing keeps those small. --optim adafactor drops
the optimiser state to near zero if that is still tight, and --freeze_encoder halves the
trainable set -- often enough here, since the failure being fixed is an OUTPUT-side one
(embedded English written in the matrix script).

Single GPU:  python train.py --method full --output_dir exp/whisper_full
Multi GPU:   torchrun --nproc_per_node=4 train.py --method full --output_dir exp/whisper_full
"""
from __future__ import annotations

import argparse
import collections
import math
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cuda_home  # noqa: F401,E402  (must precede transformers)
from cs_data import Collator, CSDataset, read_split           # noqa: E402
from cs_metrics import LANG2WHISPER                            # noqa: E402

DATA_ROOT = ''


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=['lora', 'full'], default='lora')
    p.add_argument('--model', default='openai/whisper-large-v3')
    p.add_argument('--data_root', default=DATA_ROOT)
    p.add_argument('--langs', nargs='+',
                   default=['Cantonese', 'Mandarin', 'Japanese', 'Korean'])
    p.add_argument('--output_dir', default=None,
                   help='default: exp/whisper_<method>')
    # --- LoRA (ignored when --method full). r=32 on q,v is the blog's PEFT companion.
    p.add_argument('--lora_r', type=int, default=32)
    p.add_argument('--lora_alpha', type=int, default=64)
    p.add_argument('--lora_dropout', type=float, default=0.05)
    p.add_argument('--lora_targets', nargs='+', default=['q_proj', 'v_proj'])
    # --- optimisation. lr defaults per method; see the module docstring.
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--epochs', type=float, default=5.0)
    p.add_argument('--max_steps', type=int, default=-1, help='overrides --epochs; for smoke runs')
    p.add_argument('--batch_size', type=int, default=None)
    p.add_argument('--grad_accum', type=int, default=None)
    # the blog warms up 500 of 5000 steps (10%); keep the ratio, not the absolute number
    p.add_argument('--warmup_ratio', type=float, default=0.1)
    p.add_argument('--optim', default=None,
                   help="adamw_torch_fused (default) or adafactor to cut optimiser memory")
    p.add_argument('--freeze_encoder', action='store_true',
                   help='full only: train the decoder alone. Halves the trainable set, and '
                        'the failure being fixed is output-side, so this is often enough')
    p.add_argument('--gradient_checkpointing', action='store_true', default=None)
    p.add_argument('--eval_batch_size', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=4)
    # Language temperature. 1.0 = natural (Cantonese is 69% of this corpus), 0.0 = uniform.
    # On this exact corpus a 69/7.6 split let the dominant language improve while each
    # minority one degraded in its own direction, hidden behind an averaged metric.
    # --- cs-fleurs (off by default: no flag = the SwitchLingua-only recipe, unchanged) ---
    # CS-FLEURS ships ONE training subset and it is SYNTHETIC. From its own README:
    #   Read-Test    17h   read speech      <- the TTS evaluation set. Cannot be trained on.
    #   XTTS-Train  128h   generative TTS   <- the only training audio available
    # So --csf_pairs adds XTTS-synthesised speech to a scorer whose job is to judge
    # synthesised speech. Read the header of run_csf.sh before deciding that is what you
    # want; it is a real trade-off, not a free improvement.
    # DATA_ROOT is .../libritts/cosyvoice2/data, so the sibling is one level up
    p.add_argument('--csf_root', default=os.path.join(
        os.path.dirname(DATA_ROOT), 'data_csfleurs'))
    p.add_argument('--csf_subset', default='xtts_train')
    p.add_argument('--csf_pairs', nargs='*', default=[],
                   help="e.g. cmn-eng jpn-eng kor-eng. Empty = SwitchLingua only.")
    p.add_argument('--csf_dev', action='store_true',
                   help="also add cs-fleurs dev to the eval set. OFF by default: 85% of "
                        "xtts_train/dev's SENTENCES also appear in xtts_train/train, so it "
                        "is not a held-out set and would make eval loss look better than it "
                        "is. SwitchLingua dev stays the honest early-stopping signal.")
    p.add_argument('--balance_key', choices=['lang', 'lang_corpus'], default='lang',
                   help="what --balance_alpha equalises. 'lang' merges both corpora into "
                        "one bucket per language, which is what the Whisper language tag "
                        "cares about. 'lang_corpus' keeps human and synthetic speech apart, "
                        "so a language does not end up represented mostly by TTS audio.")
    p.add_argument('--balance_alpha', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=1986)
    p.add_argument('--resume', default=None)
    a = p.parse_args()

    full = a.method == 'full'
    if a.lr is None:
        a.lr = 1e-5 if full else 1e-3
    if a.batch_size is None:
        # lora: 12.4 GiB at batch 4 with no checkpointing -> 8 lands near 21 GiB.
        # full: params+grads+AdamW already take ~25 GiB, so keep activations small.
        a.batch_size = 4 if full else 8
    if a.grad_accum is None:
        a.grad_accum = 2 if full else 1        # keep the effective batch comparable
    if a.gradient_checkpointing is None:
        a.gradient_checkpointing = full
    if a.optim is None:
        a.optim = 'adamw_torch_fused'
    if a.output_dir is None:
        a.output_dir = f'exp/whisper_{a.method}'
    return a


def balance_languages(rows, alpha, seed, key='lang'):
    """Upsample minority languages toward p(lang) proportional to n_lang ** alpha.

    Row-level repetition rather than a WeightedRandomSampler: under torchrun the Trainer
    wraps the dataset in a DistributedSampler, and a custom sampler would have to
    reimplement the sharding. Nothing is discarded -- the epoch gets longer.
    """
    def kf(r):
        return r['lang'] if key == 'lang' else f"{r['lang']}/{r.get('corpus', 'sl')}"
    by = collections.defaultdict(list)
    for r in rows:
        by[kf(r)].append(r)
    counts = {l: len(v) for l, v in by.items()}
    w = {l: n ** alpha for l, n in counts.items()}
    top = max(counts, key=counts.get)
    rng = random.Random(seed)
    out = []
    print(f'  language balancing, alpha={alpha}, key={key}:')
    for lang, rs in by.items():
        target = int(round(counts[top] * w[lang] / w[top]))
        pool = (rs * math.ceil(target / len(rs)))[:target]
        rng.shuffle(pool)
        out.extend(pool)
        print(f'    {lang:<12} {counts[lang]:>6} -> {len(pool):>6}  (x{len(pool) / counts[lang]:.2f})')
    rng.shuffle(out)
    print(f'    total        {len(rows):>6} -> {len(out):>6}')
    return out


def read_csfleurs(root, subset, pairs, with_dev=False):
    """-> (train_rows, dev_rows) from data_csfleurs/<subset>/<pair>/{train,dev}.

    read_kaldi_dir already resolves the language per utterance -- a cs-fleurs id carries the
    ISO-639-3 code in the middle ('csf_xtts_train_1000_cmn-spk0_sample0'), not at the front
    -- so the Whisper language tag comes out right for every pair without a table here.
    """
    from cs_data import read_kaldi_dir
    tr, dv = [], []
    for pair in pairs:
        for split, bucket in (('train', tr), ('dev', dv if with_dev else None)):
            if bucket is None:
                continue
            d = os.path.join(root, subset, pair, split)
            if not os.path.isdir(d):
                print(f'  cs-fleurs: {d} missing, skipped', flush=True)
                continue
            rows = read_kaldi_dir(d)
            for r in rows:
                r['corpus'] = 'csf'
                r['pair'] = pair
            bucket.extend(rows)
    return tr, dv


def main():
    args = get_args()
    import cuda_compat  # noqa: F401  (must precede transformers)
    from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments,
                              WhisperFeatureExtractor,
                              WhisperForConditionalGeneration, WhisperTokenizerFast)

    fe = WhisperFeatureExtractor.from_pretrained(args.model)
    tok = WhisperTokenizerFast.from_pretrained(args.model)

    train_rows = read_split(args.data_root, args.langs, 'train')
    dev_rows = read_split(args.data_root, args.langs, 'dev')
    for r in train_rows + dev_rows:
        r['corpus'] = 'sl'
    if args.csf_pairs:
        tr, dv = read_csfleurs(args.csf_root, args.csf_subset, args.csf_pairs,
                               with_dev=args.csf_dev)
        train_rows += tr
        dev_rows += dv
    if not train_rows:
        raise SystemExit(f'no training data under {args.data_root}')
    print(f'method={args.method} | train {len(train_rows)} utts | dev {len(dev_rows)} utts',
          flush=True)
    langs_seen = sorted({r['lang'] for r in train_rows})
    print(f"  {'language':<12}{'switchlingua':>14}{'cs-fleurs':>12}{'tag':>6}", flush=True)
    for lang in langs_seen:
        n_sl = sum(1 for r in train_rows if r['lang'] == lang and r.get('corpus') == 'sl')
        n_cf = sum(1 for r in train_rows if r['lang'] == lang and r.get('corpus') == 'csf')
        print(f'  {lang:<12}{n_sl:>14}{n_cf:>12}{"<|" + LANG2WHISPER[lang] + "|>":>8}',
              flush=True)
    n_cf_all = sum(1 for r in train_rows if r.get('corpus') == 'csf')
    if n_cf_all:
        print(f'  cs-fleurs is {n_cf_all / len(train_rows):.1%} of the training set, and it '
              f'is SYNTHETIC (XTTS) speech', flush=True)
    if args.balance_alpha < 1.0:
        train_rows = balance_languages(train_rows, args.balance_alpha, args.seed,
                                       args.balance_key)

    # LoRA freezes the base, so bf16 weights are fine. Full fine-tuning keeps fp32 master
    # weights and lets bf16=True handle mixed precision -- training 1.55B parameters
    # directly in bf16 loses too much precision in the optimiser update.
    dtype = torch.float32 if args.method == 'full' else torch.bfloat16
    model = WhisperForConditionalGeneration.from_pretrained(args.model, torch_dtype=dtype)
    # the checkpoint ships a forced decoder prefix for its own default language, which
    # would override the per-utterance prefix carried by the labels
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.forced_decoder_ids = None

    if args.method == 'full':
        if args.freeze_encoder:
            model.freeze_encoder()
            print('  encoder frozen: training the decoder only')
        if args.gradient_checkpointing and not args.freeze_encoder:
            model.enable_input_require_grads()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f'  trainable params: {trainable:,} || all params: {total:,} || '
              f'trainable%: {100 * trainable / total:.4f}')
    else:
        import peft_compat  # noqa: F401  (must precede peft)
        from peft import LoraConfig, get_peft_model
        if args.gradient_checkpointing:
            # enable_input_require_grads() hooks get_input_embeddings(), which for Whisper
            # is the DECODER token embedding. The encoder takes a mel spectrogram through
            # frozen convs, so with checkpointing on its blocks get inputs with
            # requires_grad=False, torch.utils.checkpoint returns None, and every ENCODER
            # LoRA tensor silently receives zero gradient -- measured: 0 encoder tensors
            # with grad against 256 decoder tensors. Nothing errors; the acoustic half
            # just never adapts.
            model.enable_input_require_grads()
            model.model.encoder.conv2.register_forward_hook(
                lambda _m, _i, out: out.requires_grad_(True))
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            target_modules=args.lora_targets, bias='none'))
        model.print_trainable_parameters()

    targs = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        optim=args.optim,
        gradient_checkpointing=args.gradient_checkpointing,
        bf16=True,
        eval_strategy='epoch' if args.max_steps < 0 else 'no',
        save_strategy='epoch' if args.max_steps < 0 else 'no',
        logging_steps=25,
        report_to=['tensorboard'],
        load_best_model_at_end=args.max_steps < 0,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        save_total_limit=2,
        remove_unused_columns=False,
        label_names=['labels'],
        dataloader_num_workers=args.num_workers,
        seed=args.seed,
        predict_with_generate=False,     # see the module docstring
        ddp_find_unused_parameters=False,
    )

    trainer = Seq2SeqTrainer(model=model, args=targs,
                             train_dataset=CSDataset(train_rows, fe, tok),
                             eval_dataset=CSDataset(dev_rows, fe, tok),
                             data_collator=Collator(fe, tok))
    model.config.use_cache = False
    trainer.train(resume_from_checkpoint=args.resume)

    final = os.path.join(args.output_dir,
                         'adapter_best' if args.method == 'lora' else 'model_best')
    trainer.model.save_pretrained(final)
    tok.save_pretrained(final)
    fe.save_pretrained(final)
    print(f'\n{args.method} artefact -> {final}')
    flag = '--adapter' if args.method == 'lora' else '--model'
    print('now measure it properly (per-utterance language token, real MER):\n'
          f'  python infer.py --split dev  {flag} {final}\n'
          f'  python infer.py --split test {flag} {final}')


if __name__ == '__main__':
    main()
