"""Transcribe and score code-switched speech with the fine-tuned Whisper.

Two jobs, one code path:

  1. How good is the ASR?      --split test        (real recordings, reference from `text`)
  2. How good is a TTS system? --meta_lst ... --wav_dir <synthesised wavs>

(2) is the reason this exists. seed-tts-eval/cal_wer_cs.sh scores TTS by forcing the
matrix-language token into stock whisper-large-v3, and that decoder writes embedded
English in the matrix script: on REAL HUMAN recordings it keeps latin script in only
46.8% of Korean utterances, and the ground-truth MER floor is 29.48%. A TTS system cannot
be measured against a scorer with that much of its own error, which is what this fine-tune
is meant to remove. Run --split test with and without --adapter to see whether it did.

Generation is batched per language, so every utterance is decoded under its OWN
<|yue|>/<|zh|>/<|ja|>/<|ko|> prefix. Mixing languages in one batch would force a single
prefix on all of them, which is the very bug being fixed.

    python infer.py --split test                       # baseline, no adapter
    python infer.py --split test --adapter exp/whisper_lora/adapter_best
    python infer.py --meta_lst .../cs_testset_sft/meta.lst \
        --wav_dir .../results/cs/ground_truth --adapter exp/whisper_lora/adapter_best
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cuda_home  # noqa: F401,E402  (must precede transformers)
from cs_data import load_audio_16k, read_kaldi_dir, read_split           # noqa: E402
from cs_metrics import (LANG2WHISPER, aggregate, english_retention,     # noqa: E402
                        lang_of_utt, mix_error_rate, print_table)

DATA_ROOT = ''


def rows_from_meta(meta_lst, wav_dir):
    """seed-tts meta.lst: utt|prompt_text|prompt_wav|target_text. Audio is wav_dir/utt.wav
    so the same list scores any system's output directory."""
    rows = []
    for line in open(meta_lst, encoding='utf-8'):
        p = line.rstrip('\n').split('|')
        if len(p) < 4:
            continue
        utt, text = p[0], p[3]
        lang = lang_of_utt(utt.split('-')[-1]) or lang_of_utt(utt)
        wav = os.path.join(wav_dir, utt + '.wav')
        if lang and os.path.exists(wav):
            rows.append({'utt_id': utt, 'wav': wav, 'text': text.strip(),
                         'lang': lang, 'whisper_lang': LANG2WHISPER[lang]})
    return rows


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='openai/whisper-large-v3')
    p.add_argument('--adapter', default=None, help='LoRA dir; omit for the stock model')
    p.add_argument('--split', default=None, choices=['train', 'dev', 'test'])
    p.add_argument('--data_root', default=DATA_ROOT)
    p.add_argument('--langs', nargs='+',
                   default=['Cantonese', 'Mandarin', 'Japanese', 'Korean'])
    p.add_argument('--kaldi_dir', nargs='+', default=None,
                   help='any dir(s) with wav.scp+text, e.g. cs-fleurs pairs')
    p.add_argument('--meta_lst', default=None)
    p.add_argument('--wav_dir', default=None)
    p.add_argument('--out', default=None, help='per-utterance records as jsonl')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_beams', type=int, default=1)
    p.add_argument('--max_new_tokens', type=int, default=200)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--limit', type=int, default=None)
    return p.parse_args()


def main():
    args = get_args()
    import cuda_compat  # noqa: F401  (must precede transformers)
    from transformers import (WhisperFeatureExtractor,
                              WhisperForConditionalGeneration, WhisperTokenizerFast)

    if args.kaldi_dir:
        rows = [r for d in args.kaldi_dir for r in read_kaldi_dir(d)]
        tag = ' + '.join(os.path.basename(os.path.dirname(d.rstrip('/')))
                         for d in args.kaldi_dir)
    elif args.meta_lst:
        if not args.wav_dir:
            raise SystemExit('--meta_lst needs --wav_dir')
        rows = rows_from_meta(args.meta_lst, args.wav_dir)
        tag = f'{os.path.basename(args.wav_dir)}'
    elif args.split:
        rows = read_split(args.data_root, args.langs, args.split)
        tag = f'split={args.split}'
    else:
        raise SystemExit('give --split, --kaldi_dir or --meta_lst')
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        raise SystemExit('no utterances found')

    fe = WhisperFeatureExtractor.from_pretrained(args.model)
    tok = WhisperTokenizerFast.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.float16)
    if args.adapter:
        import peft_compat  # noqa: F401  (must precede peft)
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()      # fold LoRA in so generate() is full speed
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    model = model.to(args.device).eval()

    print(f'{tag} | {len(rows)} utts | adapter={args.adapter or "NONE (stock model)"}',
          flush=True)

    # batch within a language so one forced prefix is correct for the whole batch
    by_lang = {}
    for r in rows:
        by_lang.setdefault(r['whisper_lang'], []).append(r)

    records, done = [], 0
    for code, rs in by_lang.items():
        prefix = tok.convert_tokens_to_ids([f'<|{code}|>', '<|transcribe|>',
                                            '<|notimestamps|>'])
        forced = [(i + 1, t) for i, t in enumerate(prefix)]
        for i in range(0, len(rs), args.batch_size):
            chunk = rs[i:i + args.batch_size]
            audio = [load_audio_16k(r['wav']) for r in chunk]
            feats = fe(audio, sampling_rate=16000, return_tensors='pt').input_features
            with torch.inference_mode():
                ids = model.generate(feats.to(args.device, torch.float16),
                                     forced_decoder_ids=forced,
                                     num_beams=args.num_beams,
                                     max_new_tokens=args.max_new_tokens)
            hyps = tok.batch_decode(ids, skip_special_tokens=True)
            for r, hyp in zip(chunk, hyps):
                m = mix_error_rate(hyp, r['text'], code)
                e = english_retention(hyp, r['text'], lang=r['lang'])
                if m is None:
                    continue
                records.append({'utt_id': r['utt_id'], 'lang': r['lang'],
                                'ref': r['text'], 'hyp': hyp.strip(), **m,
                                'retention': e['recall'] if e else None,
                                'lost': e['lost'] if e else None})
            done += len(chunk)
            if done % 100 < args.batch_size:
                print(f'  {done}/{len(rows)}', flush=True)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        print(f'\nper-utterance records -> {args.out}')

    print_table(aggregate(records), title=f'== {tag} | adapter={args.adapter or "none"} ==')
    print('\n"EN lost" is the share of English-bearing utterances whose transcript kept no')
    print('latin word at all -- the embedded span written in the matrix script, dropped, or')
    print('translated. Stock large-v3 scores ~53% on Korean even on real human speech.')


if __name__ == '__main__':
    main()
