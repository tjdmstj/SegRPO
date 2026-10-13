"""PIER -- Point-of-Interest Error Rate (Ugan et al., 2025, arXiv:2501.09512) on our MER outputs.

MER counts every error in the sentence. PIER counts only the errors at the POINTS OF
INTEREST -- here the embedded-language (English) tokens of a code-switched sentence -- so it
measures what code-switching TTS is about rather than diluting it with the matrix language.
Transcripts come from whisper_ft/infer.py (the MER pipeline): no ASR is rerun.

The counting (reference implementation, jiwer/process.py: process_pier, get_idsh):
  * word-level Levenshtein alignment of reference and hypothesis (rapidfuzz editops)
  * a substitution or deletion counts if its reference position is a PoI
  * an insertion counts if it is inserted right BEFORE a PoI (editops src_pos = the PoI);
    insertions after the last word count if the last reference word is a PoI
  * PIER = (S + D + I) / N_PoI x 100, pooled over the corpus (N_PoI = H + S + D)
  * sentences without any PoI, or made only of PoIs, are skipped
  * the same counts over the non-PoI (matrix) tokens are reported as the 'rest' rate

--impl authors (default): the authors' own code, used as their README does
(git clone https://github.com/enesyugan/PIER-CodeSwitching-Evaluation into whisper_ft/):
    ref = " ".join(tokenize_for_mer(ref)); hyp = " ".join(tokenize_for_mer(hyp))
    pier(refs, hyps, scd_language="cmn" | "jap")
  * tokenize_for_mer: one token per Han, kana or full-width character (so '，' is a token),
    Latin words as words, digits spelled out in English by inflect ('2024' -> 'two thousand
    and twenty four', which are PoIs then); ASCII punctuation, '。' and '、' dropped; no
    lower-casing, no simplified/traditional folding
  * PoI = a token containing a Latin letter (their tag_words)
  Two things are done around their code:
  * we pass the FIXED POINT of tokenize_for_mer (applied twice). pier() tokenises the
    reference again to find the PoIs but aligns on the whitespace split of what it is given;
    after one pass '2024' is the single item 'two thousand and twenty-four', the second pass
    makes it five tokens, and the PoI positions would be shifted against the alignment.
  * their regex has no Hangul range, so Korean text vanishes. For Korean and kor-eng (marked †)
    the regex gets Hangul syllables added, nothing else; --no_hangul leaves them out.
  --strict drops the fixed point: the README recipe verbatim (one pass), so sentences with
  numbers keep the shifted PoI positions of their code. --strict --no_hangul is their code with
  nothing added at all (Korean and kor-eng then cannot be scored).

--impl mer: the same counting on MER's normalisation and tokens (cs_metrics.normalise /
tokenize_cs) so that PIER and MER are on the same footing: lower-cased, punctuation removed,
Chinese folded to simplified, digits kept as digits (not PoIs), Hangul supported.

PIER CAN EXCEED 100%. A PoI is one English WORD, but an English word realised in kana,
Hangul or Han is several CHARACTER tokens: 'document' -> 'ドキュメント' is 1 substitution +
5 insertions charged to that PoI. That is the paper's definition; for a bounded reading the
script also prints (S + D) / N_PoI -- the share of English words not realised as such --
which ignores insertions and stays within 0-100%.

    python pier.py                                   # the five table systems, CosyVoice2
    python pier.py --strict                          # the authors' code; Hangul added for Korean
    python pier.py --strict --no_hangul              # the authors' code, nothing added
    python pier.py --impl mer
    python pier.py --eval_dir exp/whisper_lora/eval/sparktts --systems sparktts_cs_sft ...
    python pier.py --strict --eval_dir exp/whisper_lora/eval/sparktts \
        --gt_dir exp/whisper_lora/eval/tts --systems SFT=sparktts_cs_sft DPO=sparktts_cs_dpo ...
        # LABEL=folder names a row; --gt_dir adds a GT row read from another eval dir
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cs_metrics import normalise  # noqa: E402

EVAL_DIR = os.path.join(HERE, 'exp/whisper_lora/eval/tts')
LANGS = ['Cantonese', 'Mandarin', 'Japanese', 'Korean', 'cmn-eng', 'jpn-eng', 'kor-eng']
SL, CSF = LANGS[:4], LANGS[4:]
SYSTEMS = {
    'GT': 'ground_truth',
    'SFT': 'cosyvoice2_cs_sft_fleurs_switchlingua_part_best',
    'SFT + DPO': 'cosyvoice2_cs_dpo_mer_utmos_cmi_sft_best',
    'CS3': 'cosyvoice2_cs_sft_cs3_best',
    'SFT + SegRPO': 'cosyvoice2_cs_grpo_kl_segswitch_sft_best',
}
LANG_CODE = {'Cantonese': 'yue', 'Mandarin': 'zh', 'Japanese': 'ja', 'Korean': 'ko',
             'cmn-eng': 'zh', 'jpn-eng': 'ja', 'kor-eng': 'ko'}
LATIN = re.compile(r'[a-zA-Z]')
REPO = os.path.join(HERE, 'PIER-CodeSwitching-Evaluation')
SCD = {'yue': 'cmn', 'zh': 'cmn', 'ja': 'jap', 'ko': 'cmn'}   # their scd_language (cmn = jap here)


def counts(ref_tokens, hyp_tokens):
    """-> dict of S/D/I/H at PoIs and at the rest, or None if the sentence is skipped."""
    from rapidfuzz.distance import Levenshtein
    poi = [i for i, t in enumerate(ref_tokens) if LATIN.search(t)]
    rest = [i for i in range(len(ref_tokens)) if i not in set(poi)]
    if not poi or not rest:
        return None
    # word-level distance: map every distinct word to one character, as jiwer does
    vocab = {}
    for t in ref_tokens + hyp_tokens:
        vocab.setdefault(t, chr(len(vocab) + 0x100))
    ops = Levenshtein.editops(''.join(vocab[t] for t in ref_tokens),
                              ''.join(vocab[t] for t in hyp_tokens))
    n = len(ref_tokens)

    def idsh(idx):
        idx_set = set(idx)
        s = sum(op.tag == 'replace' and op.src_pos in idx_set for op in ops)
        d = sum(op.tag == 'delete' and op.src_pos in idx_set for op in ops)
        i = sum(op.tag == 'insert' and op.src_pos in idx_set for op in ops)
        if idx[-1] == n - 1:                      # trailing edits belong to the last word
            i += sum(op.tag == 'insert' and op.src_pos >= n for op in ops)
        return {'S': s, 'D': d, 'I': i, 'H': len(idx) - (s + d)}

    return {'poi': idsh(poi), 'rest': idsh(rest)}


def authors_pier(repo, hangul=True, strict=False):
    """-> run(refs, hyps, code) giving ({'poi': Counter, 'rest': Counter}, #skipped) from their pier().

    strict: their README recipe verbatim -- ONE tokenize_for_mer pass, so number words can shift
    the PoI positions as in their code. Korean still gets the Hangul range unless hangul=False."""
    for m in [m for m in sys.modules if m == 'jiwer' or m.startswith('jiwer.')]:
        del sys.modules[m]                       # their process.py imports its own jiwer
    sys.path[:0] = [repo, os.path.join(repo, 'jiwer')]
    import measures
    import process
    tokenize_for_mer = process.tokenize_for_mer

    def tokenize_ko(text):
        """their tokenize_for_mer, verbatim but for the Hangul syllable range"""
        reg_range = (r"[一-鿿]|[぀-ゟ]|[゠-ヿ]|[＀-￯]|[가-힣]"
                     r"|[0-9]+|[a-zA-Z]+\'*[a-z]*")
        matches = re.findall(reg_range, text, re.UNICODE)
        p = process.inflect.engine()
        res = []
        for item in matches:
            try:
                temp = (p.number_to_words(item) if (item.isnumeric() and
                        len(process.regex.findall(r'\p{Han}+', item)) == 0) else item)
            except Exception:
                temp = item
            res.append(temp)
        return res

    def run(refs, hyps, code):
        if code == 'ko' and not hangul:
            return None, len(refs)
        tok = process.tokenize_for_mer = tokenize_ko if code == 'ko' else tokenize_for_mer
        scd = SCD[code]
        pairs, skipped = [], 0
        for r, h in zip(refs, hyps):
            if strict:                               # README: " ".join(tokenize_for_mer(text))
                r, h = ' '.join(tok(r)), ' '.join(tok(h or ''))
            else:
                r = ' '.join(tok(' '.join(tok(r))))
                h = ' '.join(tok(' '.join(tok(h or ''))))
                assert tok(r) == r.split(), r        # what pier() tags == what it aligns
            if not r:                                # pier() refuses empty references
                skipped += 1
                continue
            poi, rest = process.extract_indices(r, False, scd, scd, True)
            skipped += not poi or not rest           # pier() skips these itself
            pairs.append((r, h))
        if not pairs:
            return None, skipped
        with contextlib.redirect_stdout(io.StringIO()):   # it prints the matrix language
            out = measures.pier([r for r, _ in pairs], [h for _, h in pairs], scd_language=scd)
        return {part: collections.Counter({
            'S': out[part]['substitutions'], 'D': out[part]['deletions'],
            'I': out[part]['insertions'], 'H': out[part]['hits']}) for part in ('poi', 'rest')}, skipped

    return run


def rate(c, insertions=True):
    n = c['H'] + c['S'] + c['D']
    e = c['S'] + c['D'] + (c['I'] if insertions else 0)
    return 100.0 * e / n if n else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval_dir', default=EVAL_DIR, help='where infer.py wrote <subset>__<system>.jsonl')
    ap.add_argument('--systems', nargs='*', default=None,
                    help='system folder names, or LABEL=folder to name the row '
                         '(default: the five systems of the paper table)')
    ap.add_argument('--gt_dir', default=None,
                    help='also report GT, read as <gt_dir>/<subset>__ground_truth.jsonl -- the '
                         'SparkTTS eval dir has no GT of its own, the CosyVoice one (eval/tts) does')
    ap.add_argument('--csv', default=None, help='also write the numbers here')
    ap.add_argument('--impl', choices=['authors', 'mer'], default='authors',
                    help="authors: their code and tokenisation; mer: MER's normalisation")
    ap.add_argument('--repo', default=REPO, help='clone of PIER-CodeSwitching-Evaluation (--impl authors)')
    ap.add_argument('--no_hangul', action='store_true',
                    help='--impl authors: leave Korean / kor-eng out instead of adding Hangul')
    ap.add_argument('--strict', action='store_true',
                    help='--impl authors: their README recipe verbatim (one tokenisation pass); '
                         'Korean keeps the Hangul range unless --no_hangul')
    args = ap.parse_args()
    systems = (dict(x.split('=', 1) if '=' in x else (x, x) for x in args.systems)
               if args.systems else dict(SYSTEMS))
    src_dir = {name: args.eval_dir for name in systems}
    if args.gt_dir:
        systems = {'GT': 'ground_truth', **{k: v for k, v in systems.items() if k != 'GT'}}
        src_dir['GT'] = args.gt_dir
    run = (authors_pier(args.repo, not args.no_hangul, args.strict)
           if args.impl == 'authors' else None)

    tot = collections.defaultdict(lambda: {k: collections.Counter() for k in ('poi', 'rest')})
    skipped = collections.Counter()
    for name, folder in systems.items():
        for lang in LANGS:
            f = os.path.join(src_dir[name], f'{lang}__{folder}.jsonl')
            if not os.path.exists(f):
                continue
            code = LANG_CODE[lang]
            rows = [json.loads(l) for l in open(f, encoding='utf-8')]
            if run is not None:
                c, skipped[(name, lang)] = run([r['ref'] for r in rows], [r['hyp'] for r in rows], code)
                if c is not None:
                    for part in ('poi', 'rest'):
                        tot[(name, lang)][part].update(c[part])
                continue
            for r in rows:
                ref = normalise(r['ref'], code).split()
                hyp = normalise(r['hyp'] or '', code).split()
                c = counts(ref, hyp)
                if c is None:
                    skipped[(name, lang)] += 1
                    continue
                for part in ('poi', 'rest'):
                    tot[(name, lang)][part].update(c[part])

    cols = LANGS + ['SwitchLingua', 'CS-FLEURS', 'All']
    groups = {'SwitchLingua': SL, 'CS-FLEURS': CSF, 'All': LANGS}

    def pooled(name, col, part):
        c = collections.Counter()
        for lang in groups.get(col, [col]):
            c.update(tot[(name, lang)][part])
        return c

    def n_tokens(c):
        return c['H'] + c['S'] + c['D']

    dagger = args.impl == 'authors' and not args.no_hangul
    heads = [c + '†' if dagger and c in ('Korean', 'kor-eng', 'SwitchLingua', 'CS-FLEURS', 'All')
             else c for c in cols]
    out = [f"implementation: {args.impl}{' (strict: README recipe verbatim)' if args.strict else ''}"]
    for part, ins, title in (
            ('poi', True, 'PIER (%) = (S + D + I) / N_PoI -- errors at the embedded-language (English) tokens'),
            ('poi', False, 'PIER without insertions (%) = (S + D) / N_PoI -- English words not realised'),
            ('rest', True, 'rest (%) -- the PIER count over the matrix-language tokens')):
        out.append(f'\n=== {title} ===')
        out.append(f"{'system':<14}" + ''.join(f'{h:>14}' for h in heads))
        for name in systems:
            out.append(f'{name:<14}' + ''.join(
                f'{rate(pooled(name, c, part), ins):>14.2f}' if n_tokens(pooled(name, c, part))
                else f"{'-':>14}" for c in cols))
    if dagger:
        out.append('\n† Korean text: the authors\' tokenize_for_mer with the Hangul syllable range added')
    first = next(iter(systems))
    out.append('\nPoI tokens per subset (sentences skipped: no PoI, or PoI only):')
    out.append('  ' + '   '.join(
        f"{l} {sum(pooled(first, l, 'poi')[k] for k in 'HSD')} ({skipped[(first, l)]})"
        for l in LANGS))
    print('\n'.join(out))

    if args.csv:
        with open(args.csv, 'w', encoding='utf-8') as f:
            f.write('system,subset,part,S,D,I,H,n_tokens,rate,rate_no_ins\n')
            for name in systems:
                for col in cols:
                    for part in ('poi', 'rest'):
                        c = pooled(name, col, part)
                        n = c['H'] + c['S'] + c['D']
                        if n:
                            f.write(f"{name},{col},{part},{c['S']},{c['D']},{c['I']},{c['H']},"
                                    f'{n},{rate(c):.4f},{rate(c, False):.4f}\n')
        print(f'-> {args.csv}')


if __name__ == '__main__':
    main()
