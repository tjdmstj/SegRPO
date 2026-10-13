"""Take the pronunciation reference from the ORIGINAL AUDIO instead of from espeak.

Why (audio_reference_reward.md §2): espeak reads spelling with English rules, but a
speaker reads the word. It expects `km` as "케이엠" while the Korean speaker says
"킬로미터", `USOC` as "유작" while the speaker says "유에스씨". Roughly 16% of English
tokens are like this -- units, acronyms, non-English proper nouns -- and on those the
reference is simply wrong, so a TTS that pronounces them correctly is punished.

Audio has no such problem. Text->phonemes needs a language argument; audio->phonemes does
not. Reading the reference off the original recording removes the language-selection step
from the pipeline entirely, and whatever the speaker actually said becomes the target.

This script precomputes, per utterance:

    ref_full     phonemes of the whole recording          <- reward when scope=full
    ref_en       phonemes of the English chunks only      <- reward when scope=en
    ref_native   phonemes of the matrix-language chunks   <- never a reward; observed only

ref_native exists because of the open question in §5/§9: is the native score actually
flat across rollouts of one sentence? If it is, scoring the whole utterance is fine and
splitting English out is unnecessary. If it moves as much as English does, whole-utterance
scoring teaches the policy to imitate the reference speaker's PROSODY rather than to
pronounce English, and the split is required. That has only ever been measured on a proxy
(four speakers reading one sentence); training measures it for real.

English spans are located with Whisper cross-attention under forced text -- no
transcription, so it works identically on Japanese and Chinese, which have no spaces.

    python build_audio_refs.py --lst .../Korean.lst --out refs/Korean.pt
    python build_audio_refs.py --kaldi_dir .../data/Korean/train --out refs/Korean_train.pt
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cuda_home  # noqa: F401,E402  (must precede transformers)
from cs_data import load_audio_16k, read_kaldi_dir                     # noqa: E402
from cs_metrics import LANG2WHISPER, char_kinds_local, lang_of_utt     # noqa: E402

PHONEME_MODEL = 'facebook/wav2vec2-lv-60-espeak-cv-ft'


# --------------------------------------------------------------------------- alignment
def median_filter(x, width):
    """Whisper's own smoothing before DTW; without it the attention is too spiky to path."""
    if width <= 1 or x.shape[-1] < width:
        return x
    pad = width // 2
    xp = F.pad(x, (pad, pad), mode='reflect')
    return xp.unfold(-1, width, 1).median(dim=-1).values


def dtw(cost):
    """Monotonic alignment of decoder tokens (rows) to encoder frames (cols)."""
    n, m = cost.shape
    D = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    T = np.zeros((n + 1, m + 1), dtype=np.int8)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c0, c1, c2 = D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]
            k = int(np.argmin((c0, c1, c2)))
            D[i, j] = cost[i - 1, j - 1] + (c0, c1, c2)[k]
            T[i, j] = k
    i, j, path = n, m, []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        k = T[i, j]
        if k == 0:
            i, j = i - 1, j - 1
        elif k == 1:
            i -= 1
        else:
            j -= 1
    return path[::-1]


class Aligner:
    """Forced text -> per-token audio spans, from Whisper cross-attention.

    Nothing is decoded: the reference tokens are teacher-forced and only the attention is
    read, which is why a language with no spaces aligns exactly like one with spaces.
    """

    def __init__(self, model_dir, device='cuda:0', dtype='float16'):
        from transformers import (GenerationConfig, WhisperFeatureExtractor,
                                  WhisperForConditionalGeneration, WhisperTokenizerFast)
        self.device = device
        self.dt = dict(float16=torch.float16, bfloat16=torch.bfloat16,
                       float32=torch.float32)[dtype]
        self.fe = WhisperFeatureExtractor.from_pretrained(model_dir)
        self.tok = WhisperTokenizerFast.from_pretrained(model_dir)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_dir, torch_dtype=self.dt).to(device).eval()
        heads = getattr(self.model.generation_config, 'alignment_heads', None)
        if not heads:
            heads = GenerationConfig.from_pretrained(
                'openai/whisper-large-v3').alignment_heads
        self.heads = [(int(l), int(h)) for l, h in heads]
        # the encoder always sees a 30 s window: 3000 mel frames -> 1500 states, 20 ms each
        self.sec_per_frame = 0.02

    @torch.inference_mode()
    def word_spans(self, audio, text, lang_code, en_spans=None):
        """-> [(surface, kind, t0, t1)] in reading order, or None if the text will not fit."""
        prefix = self.tok.convert_tokens_to_ids(
            ['<|startoftranscript|>', f'<|{lang_code}|>', '<|transcribe|>', '<|notimestamps|>'])
        body = self.tok(text, add_special_tokens=False).input_ids
        eot = self.tok.convert_tokens_to_ids('<|endoftext|>')
        ids = prefix + body + [eot]
        if len(ids) > 440:
            return None
        feats = self.fe(audio, sampling_rate=16000, return_tensors='pt').input_features
        out = self.model(input_features=feats.to(self.device, self.dt),
                         decoder_input_ids=torch.tensor([ids], device=self.device),
                         output_attentions=True)
        # [n_heads, tgt, src] from the heads whisper ships as alignment-bearing
        att = torch.stack([out.cross_attentions[l][0, h] for l, h in self.heads]).float()
        n_frames = min(att.shape[-1], int(len(audio) / 16000 / self.sec_per_frame) + 1)
        att = att[:, :, :n_frames]
        att = (att - att.mean(-1, keepdim=True)) / (att.std(-1, keepdim=True) + 1e-8)
        att = median_filter(att, 7).mean(0)                    # [tgt, src]
        # rows for the forced body only: the prefix and the final eot attend to nothing
        rows = att[len(prefix) - 1:len(prefix) - 1 + len(body)]
        path = dtw(-rows.cpu().numpy().astype(np.float32))
        first, last = {}, {}
        for ti, fi in path:
            first.setdefault(ti, fi)
            last[ti] = fi
        # token -> character span, so tokens can be grouped back into words
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        offs = enc.offset_mapping if list(enc.input_ids) == list(body) else None
        if offs is None:
            return None
        spans = []
        for (surface, kind, a, b) in self._words(text, en_spans):
            sel = [i for i, (ta, tb) in enumerate(offs) if ta < b and tb > a and i in first]
            if not sel:
                continue
            t0 = min(first[i] for i in sel) * self.sec_per_frame
            t1 = (max(last[i] for i in sel) + 1) * self.sec_per_frame
            spans.append([surface, kind, float(t0), float(t1)])
        return spans

    @staticmethod
    def strip_markers(text):
        """'a **b c** d' -> ('a b c d', [(2, 5)]) : plain text plus the English char spans."""
        import re
        out, spans, pos = [], [], 0
        for m in re.finditer(r'\*\*(.+?)\*\*', text):
            out.append(text[pos:m.start()])
            a = sum(len(x) for x in out)
            out.append(m.group(1))
            spans.append((a, a + len(m.group(1))))
            pos = m.end()
        out.append(text[pos:])
        return ''.join(out), spans

    @staticmethod
    def _words(text, en_spans=None):
        """-> [(surface, 'en'|'matrix', char_start, char_end)] in reading order.

        `en_spans` are the **...** ranges when the corpus provides them, which is strictly
        better than guessing by script: cs-fleurs marks the INSERTED English and leaves a
        proper noun that was already in the source sentence unmarked, and no script test
        can recover that distinction. Falls back to script when nothing is marked.
        """
        import re
        if en_spans:
            kinds = [None] * len(text)
            for i, ch in enumerate(text):
                if not ch.isspace() and (ch.isalnum() or ord(ch) > 0x2FF):
                    kinds[i] = 'matrix'
            for a, b in en_spans:
                for i in range(a, min(b, len(text))):
                    if kinds[i] is not None:
                        kinds[i] = 'en'
        else:
            kinds = char_kinds_local(text)
        LATIN = re.compile(r"[A-Za-zÀ-ɏ][A-Za-zÀ-ɏ'’\-]*")
        out, i, n = [], 0, len(text)
        while i < n:
            k = kinds[i]
            if k is None:
                i += 1
                continue
            if k == 'en':
                m = LATIN.match(text, i)
                j = m.end() if m else i + 1
                out.append((text[i:j], 'en', i, j))
                i = j
                continue
            j = i
            while j < n and kinds[j] == 'matrix':
                j += 1
            out.append((text[i:j], 'matrix', i, j))
            i = j
        return out


def chunks_of(spans, kind, pad_guard=True):
    """Merge neighbouring same-language words, then keep each chunk clear of the words
    around it.

    Raw attention spans overlap -- 'climbing' 6.24-7.58 swallowed 'destination' 6.82-7.72,
    so the reference for `climbing` came out as `klaɪmiŋdestineson` and a TTS that said
    `climbing` perfectly still scored badly (§6.2). Clipping to the gap between the
    neighbouring words of the OTHER language fixes it; §6.3 measured the recovered phoneme
    count at exactly 1.00x the espeak expectation afterwards, i.e. nothing extra is
    dragged in.
    """
    out, i = [], 0
    while i < len(spans):
        if spans[i][1] != kind:
            i += 1
            continue
        j = i
        while j < len(spans) and spans[j][1] == kind:
            j += 1
        lo, hi = spans[i][2], spans[j - 1][3]
        if pad_guard:
            if i > 0:
                lo = max(lo, spans[i - 1][3])
            if j < len(spans) and spans[j][2] > lo:
                hi = min(hi, spans[j][2])
        if hi - lo > 0.04:
            out.append((float(lo), float(hi),
                        ' '.join(s[0] for s in spans[i:j])))
        i = j
    return out


# ----------------------------------------------------------------------- phoneme decode
class PhonemeDecoder:
    """Free decoding: the model is asked what it hears, with no target and no language."""

    def __init__(self, device='cuda:0', model=PHONEME_MODEL, dtype='float32'):
        # wav2vec2-espeak or PhoneticXeus behind one interface (phone_backends.py). The
        # reward must load the SAME model, since refs are only meaningful in its inventory.
        from phone_backends import load_phone_model
        self.pm = load_phone_model(model, device=device, dtype=dtype)
        self.name = self.pm.name
        self.vocab, self.id2p, self.blank = self.pm.vocab, self.pm.id2p, self.pm.blank

    @torch.inference_mode()
    def decode(self, audio, with_times=False):
        """Free decoding. with_times also returns each phoneme's second in the audio.

        The timing is what lets one decode of the WHOLE utterance be split by language
        afterwards, instead of decoding every span separately. That matters twice over:
        the spans then really are slices of the full sequence (ref_full == ref_en +
        ref_native in order), and each phoneme is recognised with its neighbours present
        rather than from an isolated cut-out, where the missing coarticulation shifts the
        boundary sounds.
        """
        if len(audio) < 400:
            return ([], []) if with_times else []
        logits = self.pm.logits(torch.as_tensor(audio))
        ids = logits.argmax(-1).tolist()
        spf = len(audio) / 16000 / max(1, len(ids))     # wav2vec2 hops 320 -> 20 ms
        out, times, prev = [], [], None
        for f, i in enumerate(ids):                     # collapse repeats, drop blanks
            if i != prev and i != self.blank:
                ph = self.id2p.get(i, '')
                if ph and not ph.startswith('<'):
                    out.append(ph)
                    times.append((f + 0.5) * spf)
            prev = i
        return (out, times) if with_times else out


def split_by_spans(phones, times, spans):
    """Cut one decoded sequence into the chunks covered by `spans` (seconds).

    A phoneme belongs to the chunk whose interval contains its frame. Chunks keep their
    order, and every phoneme lands in at most one, so the pieces reassemble into the whole.
    """
    out = []
    for lo, hi, *_ in spans:
        piece = [p for p, t in zip(phones, times) if lo <= t < hi]
        if piece:
            out.append(piece)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jsonl', default=None,
                    help='marked trainset from local/build_marked_trainset.py '
                         '(id/text/file_name/language); its **...** spans are used directly')
    ap.add_argument('--lst', default=None, help='seed-tts meta.lst (uses the GT wav)')
    ap.add_argument('--kaldi_dir', nargs='+', default=None)
    ap.add_argument('--wav_dir', default=None,
                    help='with --lst: read <wav_dir>/<utt>.wav instead of the lst path')
    ap.add_argument('--out', required=True)
    ap.add_argument('--whisper', default='')
    ap.add_argument('--phoneme_model', default=PHONEME_MODEL,
                    help='facebook/wav2vec2-lv-60-espeak-cv-ft (default) or '
                         'changelinglab/PhoneticXeus. The reward must use the same one; '
                         'the id is stamped into every entry and checked at load time.')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()

    rows = []
    if args.jsonl:
        import collections as _c
        import json as _json
        skipped = _c.Counter()
        for line in open(args.jsonl, encoding='utf-8'):
            d = _json.loads(line)
            lang = d.get('language')
            code = LANG2WHISPER.get(lang) or LANG2WHISPER.get(lang_of_utt(d['id']) or '')
            if not code:
                skipped[f'no whisper language for {lang!r}'] += 1
                continue
            if not os.path.exists(d['file_name']):
                skipped['audio missing'] += 1
                continue
            rows.append({'utt_id': d['id'], 'wav': d['file_name'], 'text': d['text'],
                         'lang': lang, 'whisper_lang': code})
        # a row dropped HERE never reaches the loop, so it cannot show up as `failed`.
        # Reporting it is the difference between "0 failures" and "0 failures out of the
        # 60% that were even attempted".
        for why, n in skipped.most_common():
            print(f'  SKIPPED {n} rows: {why}', flush=True)
    if args.kaldi_dir:
        for d in args.kaldi_dir:
            rr = read_kaldi_dir(d)
            tm = os.path.join(d, 'text_marked')
            if os.path.exists(tm):     # markers beat a script test; use them when written
                marked = {}
                for line in open(tm, encoding='utf-8'):
                    p2 = line.rstrip('\n').split(' ', 1)
                    if len(p2) == 2:
                        marked[p2[0]] = p2[1]
                for r in rr:
                    if r['utt_id'] in marked:
                        r['text'] = marked[r['utt_id']]
            rows.extend(rr)
    if args.lst:
        for line in open(args.lst, encoding='utf-8'):
            p = line.rstrip('\n').split('|')
            if len(p) < 4:
                continue
            utt, text = p[0], p[3]
            lang = lang_of_utt(utt.split('-')[-1]) or lang_of_utt(utt)
            wav = (os.path.join(args.wav_dir, utt + '.wav') if args.wav_dir
                   else (p[4] if len(p) > 4 else None))
            if lang and wav and os.path.exists(wav):
                rows.append({'utt_id': utt, 'wav': wav, 'text': text.strip(),
                             'lang': lang, 'whisper_lang': LANG2WHISPER[lang]})
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        raise SystemExit('no utterances found')
    print(f'{len(rows)} utterances', flush=True)

    aligner = Aligner(args.whisper, device=args.device)
    dec = PhonemeDecoder(device=args.device, model=args.phoneme_model)
    print(f'phoneme model: {dec.name}  ({len(dec.vocab)} units)', flush=True)

    refs, n_fail = {}, 0
    for k, r in enumerate(rows):
        try:
            audio = load_audio_16k(r['wav'])
            plain, en_spans = Aligner.strip_markers(r['text'])
            spans = aligner.word_spans(audio, plain, r['whisper_lang'], en_spans)
            if spans is None:
                n_fail += 1
                continue
            en = chunks_of(spans, 'en')
            nat = chunks_of(spans, 'matrix')
            # ONE decode of the whole utterance, then sliced by language. Decoding each
            # span separately would cost N+1 forwards and, worse, would leave ref_full and
            # ref_en/ref_native as independent sequences that do not add up.
            phones, times = dec.decode(audio, with_times=True)
            entry = {
                'lang': r['lang'],
                'dur': len(audio) / 16000,
                'ref_full': phones,
                'ref_en': split_by_spans(phones, times, en),
                'ref_native': split_by_spans(phones, times, nat),
                'en_spans': [(a, b, w) for a, b, w in en],
                'native_spans': [(a, b, w) for a, b, w in nat],
                'phoneme_model': dec.name,
            }
            refs[r['utt_id']] = entry
        except Exception as ex:
            n_fail += 1
            if n_fail <= 3:
                print(f'  fail {r["utt_id"]}: {ex}', flush=True)
        if (k + 1) % 50 == 0:
            print(f'  {k + 1}/{len(rows)}  ok={len(refs)}  fail={n_fail}', flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(refs, args.out)
    nen = sum(1 for v in refs.values() if v['ref_en'])
    print(f'\n{len(refs)} refs -> {args.out}   (failed {n_fail})')
    print(f'  with an English chunk: {nen} ({100 * nen / max(1, len(refs)):.0f}%)')
    if refs:
        v = next(iter(refs.values()))
        print(f'  example  full={len(v["ref_full"])} phonemes, '
              f'en chunks={[len(x) for x in v["ref_en"]]}, '
              f'native chunks={[len(x) for x in v["ref_native"]]}')


if __name__ == '__main__':
    main()
