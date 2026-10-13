#!/usr/bin/env python3
# Copyright 2025. Prepare SwitchLingua (code-switching) data for CosyVoice2 training.
#
# The raw dataset looks like:
#   ""
#     Korean.csv          <- file_name,text,topic,...,gender,first_language,...
#     Korean/0_0.m4a      <- audio, sometimes WITHOUT the .m4a suffix (e.g. Mandarin/)
#     Mandarin.csv
#     Mandarin/0_1
#     ...
#
# CosyVoice's dataloader / tools use torchaudio with the *soundfile* backend, which
# cannot decode m4a(AAC). So we transcode every utterance to mono PCM wav at
# <sample_rate> Hz first, then emit the usual kaldi-style dirs:
#   wav.scp / text / utt2spk / spk2utt / utt2dur
#
# utt id  = <Lang>_<basename>            e.g. Korean_0_3
# spk id  = <Lang>_<group>               e.g. Korean_0   (group = prefix before '_';
#                                        one recording session == one speaker)
import argparse
import csv
import json
import logging
import os
import random
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from multiprocessing import Pool

from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger()

WS_RE = re.compile(r'\s+')


def find_ffmpeg(user_path):
    """ffmpeg is not installed in the cosyvoice env; fall back to any conda env that has it."""
    if user_path:
        return user_path
    p = shutil.which('ffmpeg')
    if p:
        return p
    # sys.prefix is <conda_root>/envs/cosyvoice -> look for a sibling env that ships ffmpeg
    env_root = os.path.dirname(sys.prefix)
    if os.path.isdir(env_root):
        for env in sorted(os.listdir(env_root)):
            cand = os.path.join(env_root, env, 'bin', 'ffmpeg')
            if os.path.isfile(cand) and os.access(cand, os.X_OK):
                return cand
    raise RuntimeError('ffmpeg not found, pass --ffmpeg /path/to/ffmpeg or `conda install -c conda-forge ffmpeg`')


def normalize_text(text):
    """kaldi `text` is line based, so no newline/tab is allowed."""
    text = str(text).replace('　', ' ')
    text = WS_RE.sub(' ', text).strip()
    return text


def resolve_audio(audio_dir, file_name, listing):
    """csv file_name is usually `0_0.m4a`, but some dirs store it without the suffix
    and a few rows claim `.wav` while the file on disk is still m4a."""
    base = os.path.splitext(file_name)[0]
    for cand in (file_name, base, base + '.m4a'):
        if cand in listing:
            return os.path.join(audio_dir, cand)
    return None


def convert_job(task):
    """(src, dst) -> (dst, duration_in_sec or None)"""
    src, dst, sample_rate, ffmpeg = task
    if not os.path.exists(dst):
        tmp = dst + '.tmp.wav'
        cmd = [ffmpeg, '-y', '-v', 'error', '-i', src,
               '-ac', '1', '-ar', str(sample_rate), '-c:a', 'pcm_s16le', '-f', 'wav', tmp]
        ret = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if ret.returncode != 0 or not os.path.exists(tmp):
            logger.warning('ffmpeg failed on %s: %s', src, ret.stderr.decode('utf-8', 'ignore').strip()[:200])
            if os.path.exists(tmp):
                os.remove(tmp)
            return dst, None
        os.replace(tmp, dst)
    try:
        import soundfile as sf
        info = sf.info(dst)
        if info.frames == 0:
            return dst, None
        return dst, info.frames / info.samplerate
    except Exception as ex:
        logger.warning('cannot read converted wav %s: %s', dst, ex)
        return dst, None


def read_manifest(csv_path):
    """The csv has huge multi-line quoted json fields, so use csv.DictReader (not naive split)."""
    csv.field_size_limit(sys.maxsize)
    rows = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            fn = (row.get('file_name') or '').strip()
            text = normalize_text(row.get('text') or '')
            if not fn or not text:
                continue
            rows.append((fn, text, row))
    return rows


def write_dir(des_dir, utts, utt2wav, utt2text, utt2spk, utt2dur):
    os.makedirs(des_dir, exist_ok=True)
    spk2utt = defaultdict(list)
    for utt in utts:
        spk2utt[utt2spk[utt]].append(utt)
    with open('{}/wav.scp'.format(des_dir), 'w', encoding='utf-8') as f:
        for utt in utts:
            f.write('{} {}\n'.format(utt, utt2wav[utt]))
    with open('{}/text'.format(des_dir), 'w', encoding='utf-8') as f:
        for utt in utts:
            f.write('{} {}\n'.format(utt, utt2text[utt]))
    with open('{}/utt2spk'.format(des_dir), 'w', encoding='utf-8') as f:
        for utt in utts:
            f.write('{} {}\n'.format(utt, utt2spk[utt]))
    with open('{}/spk2utt'.format(des_dir), 'w', encoding='utf-8') as f:
        for spk in sorted(spk2utt):
            f.write('{} {}\n'.format(spk, ' '.join(spk2utt[spk])))
    with open('{}/utt2dur'.format(des_dir), 'w', encoding='utf-8') as f:
        for utt in utts:
            f.write('{} {:.3f}\n'.format(utt, utt2dur[utt]))
    return sum(utt2dur[utt] for utt in utts)


def process_lang(lang, args, ffmpeg):
    csv_path = os.path.join(args.src_dir, lang + '.csv')
    audio_dir = os.path.join(args.src_dir, lang)
    if not os.path.exists(csv_path) or not os.path.isdir(audio_dir):
        logger.warning('skip %s: missing %s or %s', lang, csv_path, audio_dir)
        return
    listing = set(os.listdir(audio_dir))
    rows = read_manifest(csv_path)

    wav_dir = os.path.join(args.wav_dir, lang)
    os.makedirs(wav_dir, exist_ok=True)

    tasks, meta, missing = [], [], []
    seen = set()
    for fn, text, row in rows:
        src = resolve_audio(audio_dir, fn, listing)
        if src is None:
            missing.append(fn)
            continue
        base = os.path.splitext(fn)[0]
        utt = '{}_{}'.format(lang, base)
        if utt in seen:
            continue
        seen.add(utt)
        dst = os.path.join(wav_dir, base + '.wav')
        tasks.append((src, dst, args.sample_rate, ffmpeg))
        meta.append((utt, base, dst, text, row))
    if missing:
        logger.warning('[%s] %d rows have no audio file (e.g. %s)', lang, len(missing), missing[:3])

    logger.info('[%s] transcoding %d utts -> %dHz mono wav', lang, len(tasks), args.sample_rate)
    with Pool(args.num_processes) as pool:
        results = list(tqdm(pool.imap(convert_job, tasks, chunksize=16), total=len(tasks), desc=lang))
    dst2dur = {dst: dur for dst, dur in results}

    utt2wav, utt2text, utt2spk, utt2dur = {}, {}, {}, {}
    dropped = defaultdict(int)
    for utt, base, dst, text, row in meta:
        dur = dst2dur.get(dst)
        if dur is None:
            dropped['decode_error'] += 1
            continue
        if dur < args.min_dur:
            dropped['too_short'] += 1
            continue
        if dur > args.max_dur:
            dropped['too_long'] += 1
            continue
        if args.spk_mode == 'group':
            spk = '{}_{}'.format(lang, base.split('_')[0])
        elif args.spk_mode == 'lang':
            spk = lang
        else:
            spk = utt
        utt2wav[utt] = dst
        utt2text[utt] = text
        utt2spk[utt] = spk
        utt2dur[utt] = dur
    logger.info('[%s] kept %d / %d utts, dropped %s', lang, len(utt2wav), len(meta), dict(dropped))
    if not utt2wav:
        return

    # Split by speaker so that dev/test speakers are unseen during training.
    # Speakers are taken whole until the target *utterance* count is reached, so a
    # low-resource language still gets a usable dev/test set (--min_{dev,test}_utts),
    # while --max_heldout_ratio keeps us from eating a small language alive.
    spk2utts = defaultdict(list)
    for utt in sorted(utt2wav):
        spk2utts[utt2spk[utt]].append(utt)
    spks = sorted(spk2utts)
    rng = random.Random(args.seed)
    rng.shuffle(spks)

    n_total = len(utt2wav)
    budget = int(n_total * args.max_heldout_ratio)

    def target(ratio, floor):
        if ratio <= 0 and floor <= 0:
            return 0
        return min(max(int(n_total * ratio), floor), n_total)

    want_dev = target(args.dev_ratio, args.min_dev_utts)
    want_test = target(args.test_ratio, args.min_test_utts)
    # a low-resource language cannot afford both floors: shrink them proportionally
    # instead of letting whichever runs first swallow the whole budget
    if want_dev + want_test > budget:
        scale = budget / float(want_dev + want_test)
        want_dev = max(1, int(want_dev * scale)) if want_dev else 0
        want_test = max(1, int(want_test * scale)) if want_test else 0

    split_of = {}
    for split, want in (('dev', want_dev), ('test', want_test)):
        got = 0
        while got < want and spks:
            spk = spks.pop(0)
            split_of[spk] = split
            got += len(spk2utts[spk])
    for spk in spks:
        split_of[spk] = 'train'

    buckets = defaultdict(list)
    for utt in sorted(utt2wav):
        buckets[split_of[utt2spk[utt]]].append(utt)

    summary = {}
    for split, utts in buckets.items():
        des_dir = os.path.join(args.des_dir, lang, split)
        total = write_dir(des_dir, utts, utt2wav, utt2text, utt2spk, utt2dur)
        summary[split] = {'utts': len(utts), 'hours': round(total / 3600.0, 3),
                          'spks': len({utt2spk[u] for u in utts})}
        logger.info('[%s/%s] %d utts, %.2f h -> %s', lang, split, len(utts), total / 3600.0, des_dir)
    with open(os.path.join(args.des_dir, lang, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Prepare SwitchLingua code-switching data for CosyVoice2')
    parser.add_argument('--src_dir', type=str, default='', help='raw dataset root (contains <Lang>.csv and <Lang>/)')
    parser.add_argument('--wav_dir', type=str, required=True, help='where the transcoded wav files are written')
    parser.add_argument('--des_dir', type=str, required=True, help='where data/<Lang>/{train,dev,test} are written')
    parser.add_argument('--langs', type=str, nargs='*', default=None, help='default: every <Lang>.csv found in src_dir')
    parser.add_argument('--sample_rate', type=int, default=24000)
    parser.add_argument('--num_processes', type=int, default=16)
    parser.add_argument('--ffmpeg', type=str, default=None)
    parser.add_argument('--min_dur', type=float, default=1.0, help='cosyvoice2.yaml filter min_length=100 (=1s)')
    parser.add_argument('--max_dur', type=float, default=30.0, help='speech tokenizer does not support >30s')
    parser.add_argument('--dev_ratio', type=float, default=0.02, help='per-language fraction of utts held out for dev')
    parser.add_argument('--test_ratio', type=float, default=0.05)
    parser.add_argument('--min_dev_utts', type=int, default=20, help='floor so low-resource langs still get a dev set')
    parser.add_argument('--min_test_utts', type=int, default=100)
    parser.add_argument('--max_heldout_ratio', type=float, default=0.2,
                        help='dev+test never exceed this fraction of a language')
    parser.add_argument('--spk_mode', type=str, default='group', choices=['group', 'lang', 'utt'],
                        help='group: utt prefix before "_" (one recording session); utt: every utt is its own spk')
    parser.add_argument('--seed', type=int, default=1986)
    args = parser.parse_args()

    ffmpeg = find_ffmpeg(args.ffmpeg)
    logger.info('using ffmpeg at %s', ffmpeg)

    langs = args.langs
    if not langs:
        langs = sorted(f[:-4] for f in os.listdir(args.src_dir)
                       if f.endswith('.csv') and os.path.isdir(os.path.join(args.src_dir, f[:-4])))
    logger.info('languages: %s', langs)

    os.makedirs(args.wav_dir, exist_ok=True)
    os.makedirs(args.des_dir, exist_ok=True)
    for lang in langs:
        process_lang(lang, args, ffmpeg)


if __name__ == '__main__':
    main()
