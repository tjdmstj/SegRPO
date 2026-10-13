#!/usr/bin/env python3
# Build a Seed-TTS style meta.lst from the prepared SwitchLingua kaldi dirs, so the
# existing infer/infer_cosyvoice_meta.py can run zero-shot TTS over the CS test set.
#
# Line format (pipe separated), as consumed by infer_cosyvoice_meta.py:
#   utt_id | prompt_text | prompt_wav | target_text
# where utt_id = "<prompt_utt>-<target_utt>" and the generated wav is saved as <utt_id>.wav
#
# Zero-shot TTS needs a reference (prompt) utterance from the SAME speaker as the target.
# SwitchLingua has no speaker column, so prepare_cs_data.py used the recording-session
# group (`Korean_0` for `Korean_0_*`) as the speaker. Here one utterance per group becomes
# the prompt and the remaining ones become synthesis targets.
import argparse
import glob
import logging
import os
import random
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger()


def read_kv(path, join=False):
    """kaldi two-column file -> dict. join=True keeps the whole remainder as the value."""
    out = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\n').split(' ')
            if len(parts) < 2:
                continue
            out[parts[0]] = ' '.join(parts[1:]) if join else parts[1]
    return out


def pick_prompt(utts, utt2dur, max_prompt_sec):
    """Longest utterance that still fits the tokenizer's prompt limit.

    A longer reference gives a better speaker embedding, but infer_cosyvoice_meta.py
    truncates anything over ~28s, and a truncated prompt no longer matches prompt_text
    (the ASR transcript would drift), so prefer one that fits as-is.
    """
    fits = [u for u in utts if utt2dur.get(u, 0.0) <= max_prompt_sec]
    pool = fits if fits else utts
    return max(pool, key=lambda u: utt2dur.get(u, 0.0))


def build_lang(lang, d, max_prompt_sec, self_prompt=False):
    """lang is just the label used for grouping/reporting; d is the kaldi dir."""
    need = ['wav.scp', 'text', 'utt2spk']
    if not all(os.path.exists(os.path.join(d, n)) for n in need):
        logger.warning('skip %s: %s missing one of %s', lang, d, need)
        return []
    utt2wav = read_kv(os.path.join(d, 'wav.scp'))
    utt2text = read_kv(os.path.join(d, 'text'), join=True)
    utt2spk = read_kv(os.path.join(d, 'utt2spk'))
    dur_path = os.path.join(d, 'utt2dur')
    utt2dur = {k: float(v) for k, v in read_kv(dur_path).items()} if os.path.exists(dur_path) else {}

    spk2utts = defaultdict(list)
    for utt in sorted(utt2wav):
        if utt in utt2text and utt in utt2spk:
            spk2utts[utt2spk[utt]].append(utt)

    if self_prompt:
        # Reconstruction mode: every utterance is its own prompt, so speaker pairing is
        # irrelevant and nothing is dropped -- both the single-utterance speakers and the
        # prompt-only utterances that cross-pairing has to spend become targets. The
        # generated wav and the ground truth then share text AND speaker, so they can be
        # compared directly (WER against the same text, SIM against the same voice).
        rows = []
        for utt in sorted(utt2wav):
            if utt not in utt2text or utt not in utt2spk:
                continue
            rows.append({
                'utt_id': utt,  # 1:1 with the ground truth, so <utt_id>.wav pairs with target_wav
                'prompt_text': utt2text[utt],
                'prompt_wav': utt2wav[utt],
                'target_text': utt2text[utt],
                'target_wav': utt2wav[utt],
                'lang': lang,
                'spk': utt2spk[utt],
            })
        logger.info('[%s] %d speakers, %d lines (self-prompt, full coverage)',
                    lang, len(spk2utts), len(rows))
        return rows

    rows, solo = [], 0
    for spk in sorted(spk2utts):
        utts = spk2utts[spk]
        if len(utts) < 2:
            # nothing from this speaker can serve as a reference for anything else
            solo += 1
            continue
        prompt = pick_prompt(utts, utt2dur, max_prompt_sec)
        for target in utts:
            if target == prompt:
                continue
            rows.append({
                'utt_id': '{}-{}'.format(prompt, target),
                'prompt_text': utt2text[prompt],
                'prompt_wav': utt2wav[prompt],
                'target_text': utt2text[target],
                'target_wav': utt2wav[target],
                'lang': lang,
                'spk': spk,
            })
    logger.info('[%s] %d speakers, %d lines (%d single-utt speakers dropped)',
                lang, len(spk2utts), len(rows), solo)
    return rows


def write_lst(path, rows, with_gt, relative_to):
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            pw, tw = r['prompt_wav'], r['target_wav']
            if relative_to:
                pw = os.path.relpath(pw, relative_to)
                tw = os.path.relpath(tw, relative_to)
            fields = [r['utt_id'], r['prompt_text'], pw, r['target_text']]
            if with_gt:
                fields.append(tw)
            f.write('|'.join(fields) + '\n')


def main():
    ap = argparse.ArgumentParser(description='Build Seed-TTS style meta.lst from SwitchLingua test dirs')
    ap.add_argument('--data_dir', default='data', help='root holding <Lang>/<split>/wav.scp')
    ap.add_argument('--split', default='test')
    ap.add_argument('--langs', nargs='*', default=None, help='default: every <Lang>/<split> found')
    ap.add_argument('--des_dir', default='data/cs_testset')
    ap.add_argument('--max_prompt_sec', type=float, default=20.0,
                    help='prefer a prompt shorter than this (infer truncates over ~28s)')
    ap.add_argument('--per_lang', action='store_true', help='also write <Lang>.lst per language')
    ap.add_argument('--relative_to', default=None,
                    help='write wav paths relative to this dir (default: absolute paths)')
    ap.add_argument('--extra_dirs', nargs='*', default=None,
                    help='glob(s) matching kaldi dirs in any other layout, e.g. '
                         '"data_csfleurs/read_test/*/test". CS-FLEURS nests one level deeper '
                         'than SwitchLingua, so it cannot be reached via --data_dir/--split. '
                         'The label is taken from the dir name above <split>.')
    ap.add_argument('--prefix', default='',
                    help='prepended to the per-language label/filename, e.g. csf_')
    ap.add_argument('--max_per_lang', type=int, default=0,
                    help='cap lines per source (0 = no cap). CS-FLEURS read_test has as few as '
                         'one speaker per pair, so it yields ~10x more lines per language than '
                         'SwitchLingua; cap it to keep the combined set balanced and the '
                         'inference run tractable. Sampling is seeded, so it is reproducible.')
    ap.add_argument('--seed', type=int, default=1986)
    ap.add_argument('--self_prompt', action='store_true',
                    help='use each utterance as its own prompt instead of pairing it with a '
                         'different utterance from the same speaker. Covers every utterance '
                         '(no speaker pairing, nothing dropped) and makes the generated wav '
                         'directly comparable to the ground truth, which has the same text '
                         'and the same voice.')
    args = ap.parse_args()

    # (label, kaldi_dir) pairs, in the order they should appear in the combined lst
    targets = []
    langs = args.langs
    if langs is None:  # bare `--langs` (empty list) means "none", so test `is None`
        langs = sorted({p.split(os.sep)[-3] for p in
                        glob.glob(os.path.join(args.data_dir, '*', args.split, 'wav.scp'))})
    for lang in langs:
        targets.append((args.prefix + lang, os.path.join(args.data_dir, lang, args.split)))
    for pattern in (args.extra_dirs or []):
        for d in sorted(glob.glob(pattern)):
            if os.path.exists(os.path.join(d, 'wav.scp')):
                targets.append((args.prefix + os.path.basename(os.path.dirname(d)), d))
    if not targets:
        raise SystemExit('nothing to do: no <Lang>/{}/wav.scp under {} and no --extra_dirs match'
                         .format(args.split, args.data_dir))
    logger.info('sources: %s', [t[0] for t in targets])

    os.makedirs(args.des_dir, exist_ok=True)
    all_rows = []
    for label, d in targets:
        rows = build_lang(label, d, args.max_prompt_sec, args.self_prompt)
        if args.max_per_lang and len(rows) > args.max_per_lang:
            # sample across the whole language rather than truncating, so the cap does not
            # silently keep only the alphabetically-first speakers
            rows = sorted(random.Random(args.seed).sample(rows, args.max_per_lang),
                          key=lambda r: r['utt_id'])
            logger.info('[%s] capped to %d lines', label, len(rows))
        all_rows.extend(rows)
        if args.per_lang and rows:
            write_lst(os.path.join(args.des_dir, label + '.lst'), rows, False, args.relative_to)

    meta = os.path.join(args.des_dir, 'meta.lst')
    write_lst(meta, all_rows, False, args.relative_to)
    # 5-field variant (same shape as seedtts non_para_reconstruct_meta.lst): the trailing
    # ground-truth path is ignored by infer_cosyvoice_meta.py but is handy for WER/SIM.
    write_lst(os.path.join(args.des_dir, 'meta_with_gt.lst'), all_rows, True, args.relative_to)

    n_spk = len({(r['lang'], r['spk']) for r in all_rows})
    logger.info('wrote %d lines from %d speakers -> %s', len(all_rows), n_spk, meta)


if __name__ == '__main__':
    main()
