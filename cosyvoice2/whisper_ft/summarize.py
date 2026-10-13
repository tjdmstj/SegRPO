"""Collect the per-utterance jsonl files infer.py writes and print one comparison table.

Each run of infer.py scores ONE system on ONE test set. What anyone actually wants to look
at is every system side by side, so this reads the lot and lines them up, with the
ground-truth row pinned to the top: on this corpus a synthesised system can beat the real
recording on any canonicality-based metric, so an absolute MER means little and the gap to
ground truth means everything.

    python summarize.py exp/whisper_lora/eval/tts            # every jsonl in a dir
    python summarize.py exp/whisper_lora/eval/tts --lang Korean
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cuda_home  # noqa: F401,E402  (must precede transformers)
from cs_metrics import aggregate                                       # noqa: E402


def load(path):
    return [json.loads(l) for l in open(path, encoding='utf-8') if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='+', help='jsonl files, or directories holding them')
    ap.add_argument('--lang', default=None, help='restrict to one language')
    ap.add_argument('--subset', default=None, help='restrict to one test set (file prefix)')
    ap.add_argument('--csv', default=None)
    args = ap.parse_args()

    files = []
    for p in args.paths:
        files.extend(sorted(glob.glob(os.path.join(p, '*.jsonl'))) if os.path.isdir(p) else [p])
    if not files:
        raise SystemExit('no jsonl found')

    rows = []
    for f in files:
        stem = os.path.basename(f)[:-6]
        subset, _, system = stem.partition('__')
        if not system:
            subset, system = '-', stem
        if args.subset and subset != args.subset:
            continue
        recs = load(f)
        if args.lang:
            recs = [r for r in recs if r['lang'] == args.lang]
        if not recs:
            continue
        s = aggregate(recs)
        key = '__macro__' if args.lang is None else args.lang
        rows.append((subset, system, s.get(key, s['__macro__'])))

    # ground truth first, then alphabetically: every other row is read against it
    rows.sort(key=lambda r: (r[0], 'ground_truth' not in r[1], r[1]))

    scope = args.lang or 'MACRO avg over languages'
    print(f'\nscored with the CS-fine-tuned whisper  |  {scope}')
    print(f'{"test set":<16}{"system":<46}{"utts":>6}{"MER":>9}{"MERclip":>9}'
          f'{"ENrecall":>10}{"ENlost":>8}')
    print('-' * 104)
    cur, base = None, None
    for subset, system, s in rows:
        if subset != cur:
            if cur is not None:
                print()
            cur, base = subset, None
        mark = ''
        if 'ground_truth' in system:
            base = s['mer']
        elif base is not None:
            d = s['mer'] - base
            mark = f'  ({d * 100:+.2f}p vs GT)'
        print(f'{subset:<16}{system:<46}{s["n_utts"]:>6}{s["mer"] * 100:>8.2f}%'
              f'{s["mer_clip100"] * 100:>8.2f}%{s["en_recall"] * 100:>9.1f}%'
              f'{s["en_lost"] * 100:>7.1f}%{mark}')

    if args.csv:
        import csv
        with open(args.csv, 'w', newline='', encoding='utf-8') as fh:
            w = csv.writer(fh)
            w.writerow(['test_set', 'system', 'n_utts', 'n_langs', 'mer',
                        'mer_clip100', 'en_recall', 'en_lost', 'n_en_utts'])
            for subset, system, s in rows:
                w.writerow([subset, system, s['n_utts'], s['n'], round(s['mer'], 5),
                            round(s['mer_clip100'], 5), round(s['en_recall'], 5),
                            round(s['en_lost'], 5), s['n_en_utts']])
        print(f'\ncsv -> {args.csv}')
    print('\nMER here is the same mix error rate seed-tts-eval computes -- same punctuation\n'
          'table, same zhconv normalisation, same CJK-char / latin-word tokenisation -- so\n'
          'these numbers sit on the same scale as the ones cal_wer_cs.sh produced. What\n'
          'changed is the transcriber: its own floor on real speech fell 28.74% -> 5.87%.')


if __name__ == '__main__':
    main()
