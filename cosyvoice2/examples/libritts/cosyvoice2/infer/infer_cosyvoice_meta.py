#!/usr/bin/env python3
# Batch zero-shot TTS inference for CosyVoice2 over a Seed-TTS style meta.lst.
# meta.lst format (pipe separated): utt_id | prompt_text | prompt_wav(rel) | target_text
import os
import sys
import argparse

# make `cosyvoice` importable regardless of CWD
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'third_party/Matcha-TTS'))

import tempfile

import torch
import torchaudio
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

# frontend rejects prompt audio > 30s (speech tokenizer limit); keep a margin.
MAX_PROMPT_SEC = 28.0


def prep_prompt(prompt_wav, tmpdir):
    """Return a prompt path whose duration <= MAX_PROMPT_SEC (truncate if needed)."""
    try:
        info = torchaudio.info(prompt_wav)
        dur = info.num_frames / info.sample_rate
    except Exception:
        return prompt_wav
    if dur <= MAX_PROMPT_SEC:
        return prompt_wav
    wav, sr = torchaudio.load(prompt_wav)
    wav = wav[:, :int(MAX_PROMPT_SEC * sr)]
    tmp = os.path.join(tmpdir, 'prompt_trunc.wav')
    torchaudio.save(tmp, wav, sr)
    return tmp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir', required=True,
                    help='dir with cosyvoice2.yaml + (trained) llm.pt + pretrained flow.pt/hift.pt')
    ap.add_argument('--meta_file', required=True)
    ap.add_argument('--save_dir', required=True)
    ap.add_argument('--prompt_dir', default='',
                    help='base dir for prompt_wav (default: dir of meta_file)')
    ap.add_argument('--fp16', action='store_true')
    ap.add_argument('--text_frontend', action='store_true',
                    help='run CosyVoice text normalization. Off by default: training tokenizes the '
                         'raw parquet text with no normalization, and the frontend routes any '
                         'non-Chinese text (Korean, Arabic, ...) down the English branch, which '
                         'spells out numbers. Enable only to compare against normalized inference.')
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    base = args.prompt_dir or os.path.dirname(os.path.abspath(args.meta_file))

    cosyvoice = CosyVoice2(args.model_dir, load_jit=False, load_trt=False, fp16=args.fp16)
    tmpdir = tempfile.mkdtemp(prefix='cosy_prompt_')

    with open(args.meta_file) as f:
        lines = [l.rstrip('\n') for l in f if l.strip()]

    total = len(lines)
    for idx, line in enumerate(lines, 1):
        parts = line.split('|')
        if len(parts) < 4:
            print('[{}/{}] skip (bad line): {}'.format(idx, total, line))
            continue
        utt_id, prompt_text, prompt_wav_rel, target_text = parts[0], parts[1], parts[2], parts[3]
        out_path = os.path.join(args.save_dir, utt_id + '.wav')
        if os.path.exists(out_path):
            print('[{}/{}] skip (exists): {}'.format(idx, total, utt_id))
            continue
        prompt_wav = prompt_wav_rel if os.path.isabs(prompt_wav_rel) \
            else os.path.join(base, prompt_wav_rel)
        # truncate prompt to <=30s so the speech tokenizer accepts it
        prompt_wav = prep_prompt(prompt_wav, tmpdir)

        # frontend loads prompt_wav internally (24k feat / 16k token / 16k embedding),
        # so pass the file PATH, not a pre-loaded tensor.
        try:
            chunks = [o['tts_speech'] for o in cosyvoice.inference_zero_shot(
                target_text, prompt_text, prompt_wav, stream=False,
                text_frontend=args.text_frontend)]
            wav = torch.cat(chunks, dim=1)
            torchaudio.save(out_path, wav, cosyvoice.sample_rate)
            print('[{}/{}] gen: {}'.format(idx, total, utt_id), flush=True)
        except Exception as e:
            print('[{}/{}] FAIL {}: {}'.format(idx, total, utt_id, e), flush=True)
            continue


if __name__ == '__main__':
    main()
