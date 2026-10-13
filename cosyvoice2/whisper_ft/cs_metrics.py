"""Code-switching scoring, kept byte-compatible with seed-tts-eval/run_wer.py.

The point of fine-tuning Whisper here is to fix a MEASUREMENT, so the metric must not
move at the same time. `mix_error_rate` reproduces process_one_cs() from
seed-tts-eval/run_wer.py exactly -- same punctuation table, same zhconv normalisation,
same CJK-char / latin-word tokenisation -- so a number produced here is directly
comparable to the ones already recorded for the TTS systems.

`english_retention` is the diagnostic the MER hides. Forcing <|ko|> makes Whisper write
embedded English in Hangul ('War strategies' -> '월 스트라디지스'), which MER charges as a
substitution but does not distinguish from a genuinely mangled word. Retention counts, per
utterance, whether ANY of the reference's English survived as latin script. On real human
recordings the baseline model scores 46.8% on Korean -- that is the number this fine-tune
exists to move.
"""
from __future__ import annotations

import re
import string
import warnings

# jiwer 2.x exposes compute_measures(); 3.x replaced it with process_words(). The counts
# are identical, so shim rather than pin -- seed-tts-eval runs on 2.x and this recipe's env
# has 3.x, and the two must produce the same MER.
try:
    from jiwer import compute_measures as _measure

    def _counts(ref, hyp):
        m = _measure(ref, hyp)
        return m['wer'], m['substitutions'], m['deletions'], m['insertions']
except ImportError:
    from jiwer import process_words as _measure

    def _counts(ref, hyp):
        m = _measure(ref, hyp)
        return m.wer, m.substitutions, m.deletions, m.insertions

try:
    from zhon.hanzi import punctuation as _ZH_PUNCT
except ImportError:                                        # pragma: no cover
    _ZH_PUNCT = ('＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､　'
                 '、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·．！？｡。')

try:
    import zhconv
except ImportError:                                        # pragma: no cover
    zhconv = None
    warnings.warn('zhconv is not installed: Mandarin/Cantonese MER will be inflated by '
                  'traditional-vs-simplified mismatches and will NOT match the numbers in '
                  'seed-tts-eval. pip install zhconv')

PUNCT_ALL = _ZH_PUNCT + string.punctuation
CS_PUNCTUATION = PUNCT_ALL + '、。「」『』【】〜～…—–‘’“”·・'

# kana + CJK ideographs (including the extension planes Cantonese draws on) + hangul.
# One token each, so jiwer's word measures come out as a MIX error rate.
CJK_CHAR = re.compile('[぀-ヿ㐀-䶿一-鿿豈-﫿가-힣\U00020000-\U0003134f]')
LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z'’\-]*")

# corpus label -> whisper language token, and the espeak-free language code used in ids.
# large-v3 is the only checkpoint with <|yue|>; on v2 Cantonese has to borrow <|zh|>,
# which is exactly the conflation this recipe is trying to avoid.
LANG2WHISPER = {'Cantonese': 'yue', 'Mandarin': 'zh', 'Japanese': 'ja', 'Korean': 'ko'}
WHISPER2NAME = {'yue': 'cantonese', 'zh': 'chinese', 'ja': 'japanese', 'ko': 'korean'}

# cs-fleurs names its languages by ISO-639-3 inside the utterance id
# ('csf_read_test_kor_1660_EY'), and covers eleven pairs the SwitchLingua fine-tune never
# saw. Those are the interesting ones: they measure whether 20.5h of four languages
# damaged the other ninety the checkpoint already knew.
ISO3 = {'kor': 'Korean', 'jpn': 'Japanese', 'cmn': 'Mandarin', 'yue': 'Cantonese',
        'ara': 'Arabic', 'ces': 'Czech', 'deu': 'German', 'fra': 'French',
        'hin': 'Hindi', 'ita': 'Italian', 'por': 'Portuguese', 'rus': 'Russian',
        'slk': 'Slovak', 'spa': 'Spanish', 'tel': 'Telugu'}
LANG2WHISPER.update({'Arabic': 'ar', 'Czech': 'cs', 'German': 'de', 'French': 'fr',
                     'Hindi': 'hi', 'Italian': 'it', 'Portuguese': 'pt',
                     'Russian': 'ru', 'Slovak': 'sk', 'Spanish': 'es',
                     'Telugu': 'te'})
# cs-fleurs labels a corpus by its PAIR ('kor-eng'), not by the matrix language name, and
# that is what lands in the merged trainset's `language` field. Without these the whole
# cs-fleurs half resolves to no language and is skipped -- silently, because it is dropped
# before anything can fail.
LANG2WHISPER.update({f'{iso}-eng': LANG2WHISPER[name] for iso, name in ISO3.items()
                     if name in LANG2WHISPER})

# Matrix languages written in the latin alphabet. english_retention() cannot work for
# these: it counts latin words, and every word of the matrix language is one too. Reported
# as N/A rather than as a misleading number.
LATIN_MATRIX = {'Czech', 'German', 'French', 'Italian', 'Portuguese', 'Slovak', 'Spanish'}


def lang_of_utt(utt_id):
    """-> 'Korean' | None, for both corpora.

    SwitchLingua: 'Korean_102_0'          -> the prefix is the name
    cs-fleurs:    'csf_read_test_kor_166' -> an ISO-639-3 code sits in the middle
    """
    s = str(utt_id)
    head = s.split('_')[0]
    if head in LANG2WHISPER:
        return head
    # split on '-' too: cs-fleurs ids carry the speaker as 'kor-spk0', so the ISO code is
    # only a sub-part ('csf_xtts_train_1000_kor-spk0_sample0')
    import re as _re
    for part in _re.split(r'[_\-]', s):
        if part in ISO3:
            return ISO3[part]
    return None


def tokenize_cs(text):
    """CJK/hangul characters as one token each, latin runs as words."""
    tokens = []
    for word in text.split():
        buf = ''
        for ch in word:
            if CJK_CHAR.match(ch):
                if buf:
                    tokens.append(buf)
                    buf = ''
                tokens.append(ch)
            else:
                buf += ch
        if buf:
            tokens.append(buf)
    return tokens


def normalise(text, cs_lang):
    for x in CS_PUNCTUATION:
        if x == "'":
            continue                                   # keep English contractions intact
        text = text.replace(x, ' ')
    text = text.replace('’', "'")
    if cs_lang in ('zh', 'yue') and zhconv is not None:
        # whisper emits traditional for yue and mixes both for zh; unify on simplified.
        # NOT applied to japanese, whose kanji would be rewritten into Chinese forms.
        text = zhconv.convert(text, 'zh-cn')
    return ' '.join(tokenize_cs(text.lower()))


def mix_error_rate(hypo, truth, cs_lang):
    """-> dict(mer, sub, del, ins, n_ref). Same computation as run_wer.py."""
    t, h = normalise(truth, cs_lang), normalise(hypo, cs_lang)
    if not t.strip():
        return None
    wer, sub, dele, ins = _counts(t, h)
    n = len(t.split(' '))
    return {'mer': wer, 'sub': sub / n, 'del': dele / n, 'ins': ins / n, 'n_ref': n}


def english_retention(hypo, truth, min_ref_words=3, lang=None):
    """-> dict(recall, lost) for utterances whose reference holds real English, else None.

    `lost` is 1.0 when the hypothesis contains no latin word at all: the whole embedded
    span was written in the matrix script (or dropped, or translated). That is the failure
    mode, and it is invisible in MER.
    """
    if lang in LATIN_MATRIX:
        return None            # the matrix language is latin too; the count means nothing
    ref = [w.lower() for w in LATIN_WORD.findall(truth)]
    if len(ref) < min_ref_words:
        return None
    hyp = [w.lower() for w in LATIN_WORD.findall(hypo)]
    from collections import Counter
    overlap = sum((Counter(ref) & Counter(hyp)).values())
    return {'recall': overlap / len(ref), 'lost': float(not hyp), 'n_ref_en': len(ref)}


def aggregate(records):
    """records: [{lang, mer, ...}] -> per-language and macro/micro summary."""
    import numpy as np
    by = {}
    for r in records:
        by.setdefault(r['lang'], []).append(r)
    out = {}
    for lang, rs in sorted(by.items()):
        mers = [r['mer'] for r in rs if r.get('mer') is not None]
        ret = [r['retention'] for r in rs if r.get('retention') is not None]
        lost = [r['lost'] for r in rs if r.get('lost') is not None]
        out[lang] = {
            'n': len(rs), 'n_utts': len(rs),
            'mer': float(np.mean(mers)) if mers else float('nan'),
            # the mean is dominated by a handful of runaway insertions; clipping at 100%
            # is what makes checkpoints comparable (see the CS analysis in this repo)
            'mer_clip100': float(np.mean([min(m, 1.0) for m in mers])) if mers else float('nan'),
            'en_recall': float(np.mean(ret)) if ret else float('nan'),
            'en_lost': float(np.mean(lost)) if lost else float('nan'),
            'n_en_utts': len(ret),
        }
    allr = [r for rs in by.values() for r in rs]
    mers = [r['mer'] for r in allr if r.get('mer') is not None]
    ret = [r['retention'] for r in allr if r.get('retention') is not None]
    lost = [r['lost'] for r in allr if r.get('lost') is not None]
    out['__micro__'] = {
        'n': len(allr), 'n_utts': len(allr),
        'mer': float(np.mean(mers)) if mers else float('nan'),
        'mer_clip100': float(np.mean([min(m, 1.0) for m in mers])) if mers else float('nan'),
        'en_recall': float(np.mean(ret)) if ret else float('nan'),
        'en_lost': float(np.mean(lost)) if lost else float('nan'),
        'n_en_utts': len(ret),
    }
    langs = [l for l in out if not l.startswith('__')]
    # macro matters here: Cantonese is 57% of the test split, so the micro average is
    # mostly a Cantonese number
    out['__macro__'] = {
        # 'n' is the number of LANGUAGES averaged over -- that is what makes it a macro
        # average. 'n_utts' is how many utterances went in, which is what a report wants.
        'n': len(langs), 'n_utts': len(allr),
        'mer': float(np.mean([out[l]['mer'] for l in langs])),
        'mer_clip100': float(np.mean([out[l]['mer_clip100'] for l in langs])),
        'en_recall': float(np.nanmean([out[l]['en_recall'] for l in langs])),
        'en_lost': float(np.nanmean([out[l]['en_lost'] for l in langs])),
        'n_en_utts': sum(out[l]['n_en_utts'] for l in langs),
    }
    return out


def print_table(summary, title=''):
    if title:
        print(f'\n{title}')
    print(f'{"lang":<12}{"n":>6}{"MER":>9}{"MER clip":>10}{"EN recall":>11}'
          f'{"EN lost":>9}{"(n_en)":>8}')
    for lang, s in summary.items():
        name = {'__micro__': 'MICRO avg', '__macro__': 'MACRO avg'}.get(lang, lang)
        if lang.startswith('__'):
            print('-' * 65)
        print(f'{name:<12}{s["n"]:>6}{s["mer"] * 100:>8.2f}%{s["mer_clip100"] * 100:>9.2f}%'
              f'{s["en_recall"] * 100:>10.1f}%{s["en_lost"] * 100:>8.1f}%{s["n_en_utts"]:>8}')


def char_kinds_local(text):
    """-> one entry per character: 'en' | 'matrix' | None.

    Same rule reward_cs_whisper_ll.char_kinds uses, restated here so whisper_ft/ does not
    have to import the GRPO package: a code switch happens inside a whitespace token too
    ('meeting에서'), so the split is by SCRIPT, not by spaces.
    """
    import re
    LATIN = re.compile(r"[A-Za-z\u00C0-\u024F][A-Za-z\u00C0-\u024F'\u2019\-]*")
    out = [None] * len(text)
    for i, ch in enumerate(text):
        if CJK_CHAR.match(ch) or (ch.isalpha() and not ch.isascii()):
            out[i] = 'matrix'
    for m in re.finditer(r"(?<=[A-Za-z])[\u2019'](?=[A-Za-z])", text):
        out[m.start()] = 'en'
    for m in LATIN.finditer(text):
        for i in range(m.start(), m.end()):
            out[i] = 'en'
    return out
