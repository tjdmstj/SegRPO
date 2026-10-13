#!/usr/bin/env python3
# Prepare CS-FLEURS (https://huggingface.co/datasets/byan/cs-fleurs) for CosyVoice2 training.
#
# Source layout:
#   <src_dir>/read/test/{audio/<lang>/*.wav, metadata.jsonl}     real read speech, 17h
#   <src_dir>/xtts/{train,test1,test2}/{audio/..., metadata.jsonl}   XTTS-synthesized
#   <src_dir>/mms/test/{audio/..., metadata.jsonl}               concatenative TTS
#
# metadata.jsonl record:
#   {"id","file_name","text","duration","language":"ara-eng","speaker",["fluency"]}
#
# Unlike SwitchLingua, the audio is already mono PCM wav, so nothing is transcoded --
# the wavs are referenced in place. They are 16 kHz, which is correct for LLM training:
# the speech tokenizer consumes 16 kHz natively, and Qwen2LM.forward only ever reads
# text_token/speech_token, never the 24 kHz mel. Do NOT reuse this data for flow/hifigan
# training, which does need real 24 kHz content.
import argparse
import json
import logging
import os
import random
import re
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger()

# subset -> (relative path, tag used in the output tree, is it a training split)
SUBSETS = {
    'read_test':  ('read/test',  False),
    'xtts_train': ('xtts/train', True),
    'xtts_test1': ('xtts/test1', False),
    'xtts_test2': ('xtts/test2', False),
    'mms_test':   ('mms/test',   False),
}

WS_RE = re.compile(r'\s+')
STAR_RE = re.compile(r'\*+')


def clean_text(text):
    """xtts/train wraps the embedded-language spans in **...** markup (96% of lines).

    The asterisks are annotation, not speech, so they must not reach the tokenizer.
    They are always balanced pairs and never appear as a single '*', so a blanket
    strip is safe.
    """
    text = STAR_RE.sub('', str(text))
    return WS_RE.sub(' ', text).strip()


def write_dir(des_dir, rows):
    os.makedirs(des_dir, exist_ok=True)
    spk2utt = defaultdict(list)
    for r in rows:
        spk2utt[r['spk']].append(r['utt'])
    with open(os.path.join(des_dir, 'wav.scp'), 'w', encoding='utf-8') as f:
        for r in rows:
            f.write('{} {}\n'.format(r['utt'], r['wav']))
    with open(os.path.join(des_dir, 'text'), 'w', encoding='utf-8') as f:
        for r in rows:
            f.write('{} {}\n'.format(r['utt'], r['text']))
    with open(os.path.join(des_dir, 'utt2spk'), 'w', encoding='utf-8') as f:
        for r in rows:
            f.write('{} {}\n'.format(r['utt'], r['spk']))
    with open(os.path.join(des_dir, 'spk2utt'), 'w', encoding='utf-8') as f:
        for spk in sorted(spk2utt):
            f.write('{} {}\n'.format(spk, ' '.join(spk2utt[spk])))
    with open(os.path.join(des_dir, 'utt2dur'), 'w', encoding='utf-8') as f:
        for r in rows:
            f.write('{} {:.3f}\n'.format(r['utt'], r['dur']))
    return sum(r['dur'] for r in rows)


def load_subset(src_dir, tag, args):
    rel, _is_train = SUBSETS[tag]
    base = os.path.join(src_dir, rel)
    meta = os.path.join(base, 'metadata.jsonl')
    if not os.path.exists(meta):
        logger.warning('skip %s: %s not found', tag, meta)
        return {}

    by_pair = defaultdict(list)
    dropped = defaultdict(int)
    seen = set()
    with open(meta, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            pair = r.get('language', 'unk')
            if args.pairs and pair not in args.pairs:
                continue
            dur = float(r.get('duration', 0.0))
            if dur < args.min_dur:
                dropped['too_short'] += 1
                continue
            if dur > args.max_dur:
                dropped['too_long'] += 1
                continue
            text = clean_text(r.get('text', ''))
            if not text:
                dropped['empty_text'] += 1
                continue
            # some file_name values contain a doubled slash
            wav = os.path.normpath(os.path.join(base, r['file_name']))
            if not os.path.exists(wav):
                dropped['missing_wav'] += 1
                continue
            utt = 'csf_{}_{}'.format(tag, r['id'])
            if utt in seen:
                dropped['dup_id'] += 1
                continue
            seen.add(utt)
            by_pair[pair].append({
                'utt': utt,
                'wav': wav,
                'text': text,
                # speaker ids like "BT" are only unique within a pair, so qualify them
                'spk': 'csf_{}_{}'.format(pair, r.get('speaker', 'unk')),
                'dur': dur,
            })
    if dropped:
        logger.info('[%s] dropped %s', tag, dict(dropped))
    return by_pair


def main():
    ap = argparse.ArgumentParser(description='Prepare CS-FLEURS for CosyVoice2')
    ap.add_argument('--src_dir', default='')
    ap.add_argument('--des_dir', default='data_csfleurs')
    ap.add_argument('--subsets', nargs='*', default=['read_test', 'xtts_train'],
                    choices=list(SUBSETS), help='default: the real read-speech test set + the only train split')
    ap.add_argument('--pairs', nargs='*', default=None,
                    help='language pairs to keep, e.g. cmn-eng jpn-eng kor-eng (default: all)')
    ap.add_argument('--min_dur', type=float, default=1.0)
    ap.add_argument('--max_dur', type=float, default=30.0, help='speech tokenizer limit')
    ap.add_argument('--dev_ratio', type=float, default=0.02,
                    help='held out from training subsets only')
    ap.add_argument('--seed', type=int, default=1986)
    args = ap.parse_args()

    if args.pairs:
        args.pairs = set(args.pairs)
    rng = random.Random(args.seed)
    grand = defaultdict(float)

    for tag in args.subsets:
        _rel, is_train = SUBSETS[tag]
        by_pair = load_subset(args.src_dir, tag, args)
        if not by_pair:
            continue
        for pair in sorted(by_pair):
            rows = sorted(by_pair[pair], key=lambda r: r['utt'])
            if is_train:
                # CS-FLEURS gives only ~3 speakers per pair, so a speaker-held-out dev
                # set would cost a third of the data. Split by utterance instead.
                shuffled = rows[:]
                rng.shuffle(shuffled)
                n_dev = int(len(shuffled) * args.dev_ratio)
                splits = {'dev': sorted(shuffled[:n_dev], key=lambda r: r['utt']),
                          'train': sorted(shuffled[n_dev:], key=lambda r: r['utt'])}
            else:
                splits = {'test': rows}
            for split, srows in splits.items():
                if not srows:
                    continue
                des = os.path.join(args.des_dir, tag, pair, split)
                total = write_dir(des, srows)
                grand[(tag, split)] += total
                logger.info('[%s/%s/%s] %d utts, %.2f h -> %s',
                            tag, pair, split, len(srows), total / 3600.0, des)

    print()
    for (tag, split), sec in sorted(grand.items()):
        print('{:12s} {:5s} {:7.2f} h'.format(tag, split, sec / 3600.0))


if __name__ == '__main__':
    main()
